import os
import numpy as np
import cv2

from tqdm import tqdm
import json
import open3d as o3d
import random
from scipy.optimize import least_squares

from preprocess import get_selected_points2d, get_camera_intrinsics
from preprocess import SCENE_GRAPH_FILE, RANSAC_MATCH_DIR, RANSAC_ESSENTIAL_DIR, HAS_BUNDLE_ADJUSTMENT, RESULT_DIR
from bundle_adjustment import compute_ba_residuals


def get_init_image_ids(scene_graph: dict) -> (str, str):
    max_inliers = 0
    max_pair = [None, None]
    
    for image_id1, neighbors in scene_graph.items():
        for image_id2 in neighbors:
            # Load matches and count inliers
            match_id = '_'.join(sorted([image_id1, image_id2]))
            match_file = os.path.join(RANSAC_MATCH_DIR, match_id + '.npy')
            
            if os.path.exists(match_file):
                matches = np.load(match_file)
                num_inliers = matches.shape[0]
                
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    max_pair = [image_id1, image_id2]
    
    image_id1, image_id2 = sorted(max_pair)
    return image_id1, image_id2



def visualize_point_cloud(pts: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw([pcd])


def load_matches(image_id1: str, image_id2: str) -> np.ndarray:
    """ Returns N x 2 indexes of matches  [i,j] where keypoints1[i] at image_id1 corresponds to keypoints2[j]
    for image_id2 """
    sorted_nodes = sorted([image_id1, image_id2])
    match_id = '_'.join(sorted_nodes)
    match_file = os.path.join(RANSAC_MATCH_DIR, match_id + '.npy')
    matches = np.load(match_file)
    if sorted_nodes[0] == image_id2:
        matches = np.flip(matches, axis=1)
    return matches


def get_init_extrinsics(image_id1: str, image_id2: str, intrinsics: np.ndarray) -> (np.ndarray, np.ndarray):
    extrinsics1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # [I|0]

    # Load essential matrix
    match_id = '_'.join([image_id1, image_id2])
    essential_mtx_file = os.path.join(RANSAC_ESSENTIAL_DIR, match_id + '.npy')
    essential_mtx = np.load(essential_mtx_file)

    # Get matching points
    matches = load_matches(image_id1=image_id1, image_id2=image_id2)
    points2d_1 = get_selected_points2d(image_id=image_id1, select_idxs=matches[:, 0])
    points2d_2 = get_selected_points2d(image_id=image_id2, select_idxs=matches[:, 1])

    # Decompose the essential matrix to obtain the possible poses
    _, R, t, _ = cv2.recoverPose(essential_mtx, points2d_1, points2d_2, intrinsics)

    extrinsics2 = np.hstack((R, t))
    return extrinsics1, extrinsics2



def initialize(scene_graph: dict, intrinsics: np.ndarray):
    """
    Performs initialization step.

    Args:
        scene_graph: dict of the scene graph where scene_graph[image_id1] returns the list of neighboring image ids
        of image_id1. scene graph is modelled like an adjacency list.
        intrinsics: 3x3 camera intrinsics

    Returns:
        image_id1: image at the world origin
        image_id2: neighbor of image_id1 where (image_id1, image_id2) has the highest number of RANSAC matches in
        the <scene_graph>
        extrinsics1: [I|0] extrinsic array
        extrinsics2: [R|t] extrinsic array from essential matrix
        points3d: points from triangulation betwene image_id1 and image_id2
        correspondences2d3d: dictionary of correspondences between 2d keypoints and 3d points for each image
            e.g. correspondences2d3d[image_id1][i] = j means that keypoint indexed at i in keypoint file for <image_id1>
            is correspondences to <points3d> indexed at j. Note that correspondences2d3d[image_id1] is a dictionary.
    """
    image_id1, image_id2 = get_init_image_ids(scene_graph)
    extrinsics1, extrinsics2 = get_init_extrinsics(image_id1=image_id1, image_id2=image_id2, intrinsics=intrinsics)
    matches = load_matches(image_id1=image_id1, image_id2=image_id2)
    points3d = triangulate(image_id1=image_id1, image_id2=image_id2, extrinsics1=extrinsics1,
                           extrinsics2=extrinsics2, intrinsics=intrinsics, kp_idxs1=matches[:, 0],
                           kp_idxs2=matches[:, 1])

    num_matches = matches.shape[0]
    correspondences2d3d = {
        image_id1: {matches[i, 0]: i for i in range(num_matches)},
        image_id2: {matches[i, 1]: i for i in range(num_matches)}
    }
    return image_id1, image_id2, extrinsics1, extrinsics2, points3d, correspondences2d3d


def triangulate(image_id1: str, image_id2: str, kp_idxs1: np.ndarray, kp_idxs2: np.ndarray,
                extrinsics1: np.ndarray, extrinsics2: np.ndarray, intrinsics: np.ndarray):
    proj_pts1 = get_selected_points2d(image_id=image_id1, select_idxs=kp_idxs1)
    proj_pts2 = get_selected_points2d(image_id=image_id2, select_idxs=kp_idxs2)

    proj_mtx1 = np.matmul(intrinsics, extrinsics1)
    proj_mtx2 = np.matmul(intrinsics, extrinsics2)

    points3d = cv2.triangulatePoints(projMatr1=proj_mtx1, projMatr2=proj_mtx2,
                                     projPoints1=proj_pts1.transpose(1, 0), projPoints2=proj_pts2.transpose(1, 0))
    points3d = points3d.transpose(1, 0)
    points3d = points3d[:, :3] / points3d[:, 3].reshape(-1, 1)
    return points3d


def get_reprojection_residuals(points2d: np.ndarray, points3d: np.ndarray, intrinsics: np.ndarray,
                               rotation_mtx: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    projected_pts, _ = cv2.projectPoints(points3d, rotation_mtx, tvec, intrinsics, None)
    projected_pts = projected_pts.squeeze()

    # Compute Euclidean distance between original and reprojected points
    residuals = np.linalg.norm(points2d - projected_pts, axis=1)
    return residuals



def solve_pnp(image_id: str, point2d_idxs: np.ndarray, all_points3d: np.ndarray, point3d_idxs: np.ndarray,
              intrinsics: np.ndarray, num_ransac_iterations: int = 200, inlier_threshold: float = 5.0):
    num_pts = point2d_idxs.shape[0]
    assert num_pts >= 6, 'There should be at least 6 points'

    points2d = get_selected_points2d(image_id=image_id, select_idxs=point2d_idxs)
    points3d = all_points3d[point3d_idxs]

    has_valid_solution = False
    max_rotation_mtx, max_tvec, max_is_inlier, max_num_inliers = None, None, None, 0
    for _ in range(num_ransac_iterations):
        selected_idxs = np.random.choice(num_pts, size=6, replace=False).reshape(-1)
        selected_pts2d = points2d[selected_idxs, :]
        selected_pts3d = points3d[selected_idxs, :]

        # Solve PnP with selected points
        success, rvec, tvec = cv2.solvePnP(selected_pts3d, selected_pts2d, intrinsics, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            rotation_mtx, _ = cv2.Rodrigues(rvec)

            # Compute reprojection residuals
            residuals = get_reprojection_residuals(points2d=selected_pts2d, points3d=selected_pts3d,
                                                   intrinsics=intrinsics, rotation_mtx=rotation_mtx, tvec=tvec)

            is_inlier = residuals <= inlier_threshold
            num_inliers = np.sum(is_inlier).item()

            if num_inliers > max_num_inliers:
                max_rotation_mtx = rotation_mtx
                max_tvec = tvec
                max_is_inlier = is_inlier
                max_num_inliers = num_inliers
                has_valid_solution = True

    assert has_valid_solution
    inlier_idxs = np.argwhere(max_is_inlier).reshape(-1)
    return max_rotation_mtx, max_tvec, inlier_idxs



def add_points3d(image_id1: str, image_id2: str, all_extrinsic: dict, intrinsics, points3d: np.ndarray,
                 correspondences2d3d: dict):
    """
    From the image pair (image_id1, image_id2), triangulate to get new points3d. Update the correspondences
    for image_id1 and image_id2 and return the updated points3d as well.

    Args:
        image_id1: new image id
        image_id2: registered image id
        all_extrinsic: dictionary of image extrinsics where all_extrinsic[image_id1] returns the 3x4 extrinsic for
                        image_id1
        intrinsics: 3 x 3 camera intrinsic matrix
        points3d: M x 3 array of old 3d points
        correspondences2d3d: dictionary of correspondences between 2d keypoints and 3d points for each image
            e.g. correspondences2d3d[image_id1][i] = j means that keypoint indexed at i in keypoint file for <image_id1>
            is correspondences to <points3d> indexed at j. Note that correspondences2d3d[image_id1] is a dictionary.

    Returns:
        points3d: updated points3d
        correspondences2d3d: updated correspondences2d3d.
    """
    # Load the matches between the images
    matches = load_matches(image_id1=image_id1, image_id2=image_id2)

    # Identify the points in image_id2 that are not yet registered in correspondences2d3d
    points2d_idxs2 = np.setdiff1d(matches[:, 1], list(correspondences2d3d[image_id2].keys())).reshape(-1)
    if len(points2d_idxs2) == 0:
        return points3d, correspondences2d3d  # No new points to register

    # Filter the matches for the unregistered points in image_id2
    matches_idxs = np.array([np.argwhere(matches[:, 1] == i).reshape(-1)[0] for i in points2d_idxs2])
    matches = matches[matches_idxs, :]

    # Get extrinsics for both images
    extrinsic1 = all_extrinsic[image_id1]  # 3x4 matrix
    extrinsic2 = all_extrinsic[image_id2]  # 3x4 matrix

    # Form the projection matrices for both images
    projection_matrix1 = intrinsics @ extrinsic1  # 3x4 matrix
    projection_matrix2 = intrinsics @ extrinsic2  # 3x4 matrix

    # Initialize an array to hold the new 3D points
    new_points3d = []

    # Triangulate for each match (corresponding 2D points in both images)
    for i in range(matches.shape[0]):
        # Get the 2D points in image_id1 and image_id2
        pt1 = np.array([matches[i, 0], 1.0])  # Homogeneous coordinates
        pt2 = np.array([matches[i, 1], 1.0])  # Homogeneous coordinates

        # Triangulate the points (3D points are returned in homogeneous coordinates)
        point3d_homogeneous = cv2.triangulatePoints(projection_matrix1, projection_matrix2, pt1, pt2)

        # Convert from homogeneous coordinates to 3D (x, y, z)
        point3d = point3d_homogeneous[:3] / point3d_homogeneous[3]

        new_points3d.append(point3d)

    # Convert the list of new 3D points to a NumPy array
    new_points3d = np.array(new_points3d)

    # Update the correspondences2d3d dictionary for image_id1 and image_id2
    num_new_points3d = new_points3d.shape[0]
    new_points3d_idxs = np.arange(num_new_points3d) + points3d.shape[0]

    correspondences2d3d[image_id1] = {matches[i, 0]: new_points3d_idxs[i] for i in range(num_new_points3d)}
    correspondences2d3d[image_id2] = {matches[i, 1]: new_points3d_idxs[i] for i in range(num_new_points3d)}

    # Append the new 3D points to the existing ones
    points3d = np.concatenate([points3d, new_points3d], axis=0)

    return points3d, correspondences2d3d


def get_next_pair(scene_graph: dict, registered_ids: list):
    """
    Finds the next match where one of the images is unregistered while the other is registered. The next image pair
    is the one that has the highest number of inliers.

    Args:
        scene_graph: dict of the scene graph where scene_graph[image_id1] returns the list of neighboring image ids
        of image_id1. scene graph is modeled like an adjacency list.
        registered_ids: list of registered image ids

    Returns:
        new_id: new image id to be registered
        registered_id: registered image id that has the highest number of inliers along with the new_id
    """
    max_new_id, max_registered_id, max_num_inliers = None, None, 0

    # Iterate through the registered images and their neighbors
    for registered_id in registered_ids:
        neighbors = scene_graph.get(registered_id, [])
        
        for new_id in neighbors:
            if new_id not in registered_ids:
                # Load the matches between the registered image and the new image
                matches = load_matches(image_id1=registered_id, image_id2=new_id)

                # Here you would define a function to calculate the number of inliers (e.g., based on a threshold)
                num_inliers = calculate_inliers(matches)

                # Update the pair if this one has more inliers than the previous best
                if num_inliers > max_num_inliers:
                    max_new_id, max_registered_id, max_num_inliers = new_id, registered_id, num_inliers

    return max_new_id, max_registered_id



def get_pnp_2d3d_correspondences(image_id1: str, image_id2: str, correspondences2d3d: dict) -> (np.ndarray, np.ndarray):
    """
    Returns 2d and 3d correspondences for the image_id1 and the current world points. We use the transitive property
    where matches[i, 0] -> matches[i, 1] -> correspondences[image_id2][matches[i,1]], where image_id2 is a registered
    image

    Args:
        image_id1: new image id
        image_id2: registered image id
        correspondences2d3d: dictionary of correspondences

    Returns:
        points2d_idxs: keypoint idxs of first image
        points3d_idxs: points3d idxs
    """
    matches = load_matches(image_id1=image_id1, image_id2=image_id2)
    points2d_idxs2 = np.intersect1d(matches[:, 1], list(correspondences2d3d[image_id2].keys())).reshape(-1)
    match_idxs = [np.argwhere(matches[:, 1] == i).reshape(-1)[0] for i in points2d_idxs2]
    match_idxs = np.array(match_idxs)
    points2d_idxs1 = matches[match_idxs, 0]
    point3d_idxs = np.array([correspondences2d3d[image_id2][i] for i in points2d_idxs2])
    return points2d_idxs1, point3d_idxs


def bundle_adjustment(registered_ids: list, points3d: np.ndarray, correspondences2d3d: np.ndarray,
                      all_extrinsics: dict, intrinsics: np.ndarray, max_nfev: int = 30):
    # create parameters
    parameters = []
    for image_id in registered_ids:
        # convert rotation matrix to Rodriguez vector
        extrinsics = all_extrinsics[image_id]
        rotation_mtx = extrinsics[:3, :3]
        tvec = extrinsics[:, 3].reshape(3)
        rotation_mtx = rotation_mtx.astype(float)
        rvec, _ = cv2.Rodrigues(rotation_mtx)
        rvec = rvec.reshape(3)

        parameters.append(rvec)
        parameters.append(tvec)
    parameters.append(points3d.reshape(-1))
    parameters = np.concatenate(parameters, axis=0)

    # create correspondences
    points2d, camera_idxs, points3d_idxs = [], [], []
    for i, image_id in enumerate(registered_ids):
        correspondence_dict = correspondences2d3d[image_id]
        correspondences = np.array([[k, v] for k, v in correspondence_dict.items()])
        pt2d_idxs = correspondences[:, 0]
        pt3d_idxs = correspondences[:, 1]

        pt2d = get_selected_points2d(image_id=image_id, select_idxs=pt2d_idxs)
        points2d.append(pt2d)
        points3d_idxs.append(pt3d_idxs)
        camera_idxs.append(np.ones(pt2d.shape[0]) * i)

    num_cameras = len(registered_ids)
    points2d = np.concatenate(points2d, axis=0)
    camera_idxs = np.concatenate(camera_idxs, axis=0).astype(int)
    points3d_idxs = np.concatenate(points3d_idxs, axis=0).astype(int)

    # run optimization
    results = least_squares(fun=compute_ba_residuals, x0=parameters, method='trf', max_nfev=max_nfev,
                            args=(intrinsics, num_cameras, points2d, camera_idxs, points3d_idxs), verbose=2)

    updated_parameters = results.x
    camera_parameters = updated_parameters[:num_cameras * 6]
    camera_parameters = camera_parameters.reshape(num_cameras, 6)
    for i, image_id in enumerate(registered_ids):
        params = camera_parameters[i]
        rvec, tvec = params[:3], params[3:]
        rvec = rvec.reshape(1, 3)
        rotation_mtx, _ = cv2.Rodrigues(rvec)
        extrinsics = np.concatenate([rotation_mtx, tvec.reshape(-1, 1)], axis=1)
        all_extrinsics[image_id] = extrinsics
    points3d = updated_parameters[num_cameras * 6:].reshape(-1, 3)
    return all_extrinsics, points3d


def incremental_sfm(registered_ids: list, all_extrinsic: dict, intrinsics: np.ndarray, points3d: np.ndarray,
                    correspondences2d3d: dict, scene_graph: dict, has_bundle_adjustment: bool) -> \
        (np.ndarray, dict, dict, list):
    all_image_ids = list(scene_graph.keys())
    num_steps = len(all_image_ids) - 2
    for _ in tqdm(range(num_steps)):
        # get pose for new image
        new_id, registered_id = get_next_pair(scene_graph=scene_graph, registered_ids=registered_ids)
        points2d_idxs1, points3d_idxs = get_pnp_2d3d_correspondences(image_id1=new_id, image_id2=registered_id,
                                                                     correspondences2d3d=correspondences2d3d)
        rotation_mtx, tvec, inlier_idxs = solve_pnp(image_id=new_id, point2d_idxs=points2d_idxs1,
                                                    all_points3d=points3d, point3d_idxs=points3d_idxs,
                                                    intrinsics=intrinsics)

        # update correspondences
        new_extrinsics = np.concatenate([rotation_mtx, tvec.reshape(-1, 1)], axis=1)
        all_extrinsic[new_id] = new_extrinsics
        correspondences2d3d[new_id] = {points2d_idxs1[i]: points3d_idxs[i] for i in inlier_idxs}

        # create new points + update correspondences
        points3d, correspondences2d3d = add_points3d(image_id1=new_id, image_id2=registered_id,
                                                     all_extrinsic=all_extrinsic,
                                                     intrinsics=intrinsics, points3d=points3d,
                                                     correspondences2d3d=correspondences2d3d)
        registered_ids.append(new_id)

    if has_bundle_adjustment:
        all_extrinsic, points3d = bundle_adjustment(registered_ids=registered_ids, points3d=points3d,
                                                    all_extrinsics=all_extrinsic, intrinsics=intrinsics,
                                                    correspondences2d3d=correspondences2d3d)
    assert len(np.setdiff1d(all_image_ids, registered_ids).reshape(-1)) == 0
    return points3d, all_extrinsic, correspondences2d3d, registered_ids


def main():
    # set seeds
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)

    with open(SCENE_GRAPH_FILE, 'r') as f:
        scene_graph = json.load(f)

    # run initialization step
    intrinsics = get_camera_intrinsics()
    image_id1, image_id2, extrinsic1, extrinsic2, points3d, correspondences2d3d = \
        initialize(scene_graph=scene_graph, intrinsics=intrinsics)
    registered_ids = [image_id1, image_id2]
    all_extrinsic = {
        image_id1: extrinsic1,
        image_id2: extrinsic2
    }

    points3d, all_extrinsic, correspondences2d3d, registered_ids = \
        incremental_sfm(registered_ids=registered_ids, all_extrinsic=all_extrinsic, intrinsics=intrinsics,
                        correspondences2d3d=correspondences2d3d, points3d=points3d, scene_graph=scene_graph,
                        has_bundle_adjustment=HAS_BUNDLE_ADJUSTMENT)

    os.makedirs(RESULT_DIR, exist_ok=True)
    points3d_save_file = os.path.join(RESULT_DIR, 'points3d.npy')
    np.save(points3d_save_file, points3d)

    correspondences2d3d = {a: {int(c): int(d) for c, d in b.items()} for a, b in correspondences2d3d.items()}
    correspondences2d3d_save_file = os.path.join(RESULT_DIR, 'correspondences2d3d.json')
    with open(correspondences2d3d_save_file, 'w') as f:
        json.dump(correspondences2d3d, f, indent=1)

    all_extrinsic = {image_id: [list(row) for row in extrinsic.astype(float)]
                     for image_id, extrinsic in all_extrinsic.items()}
    extrinsic_save_file = os.path.join(RESULT_DIR, 'all-extrinsic.json')
    with open(extrinsic_save_file, 'w') as f:
        json.dump(all_extrinsic, f, indent=1)

    registration_save_file = os.path.join(RESULT_DIR, 'registration-trajectory.txt')
    with open(registration_save_file, 'w') as f:
        for image_id in registered_ids:
            f.write(image_id + '\n')


if __name__ == '__main__':
    main()

