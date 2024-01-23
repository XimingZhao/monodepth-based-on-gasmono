import cv2
import numpy as np
import os
import torch

def estimate_global_pose(img, cameraMatrix, distCoeffs, patternSize, squareSize):
    """
    Estimate the global pose of an image.
    """
    found, corners = cv2.findChessboardCorners(img, patternSize)
    if not found:
        return None, None

    objectPoints = np.zeros((np.prod(patternSize), 3), np.float32)
    objectPoints[:, :2] = np.indices(patternSize).T.reshape(-1, 2)
    objectPoints *= squareSize

    ret, rvec, tvec = cv2.solvePnP(objectPoints, corners, cameraMatrix, distCoeffs)
    if not ret:
        return None, None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec

def create_transformation_matrix(R, t):
    """Create a 4x4 transformation matrix from rotation matrix and translation vector."""
    T = np.zeros((4, 4), dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()
    T[3, 3] = 1
    return T



def compute_relative_pose(R1, t1, R2, t2):
    """
    Compute the relative pose.
    """
    R_rel = np.dot(R2, np.linalg.inv(R1))
    t_rel = t2 - np.dot(R_rel, t1)
    return R_rel, t_rel

def process_image_sequence(input_dir, output_dir, cameraMatrix, distCoeffs, patternSize, squareSize):
    """
    Process the image sequence and compute frame-to-frame poses.
    """
    images = [img for img in sorted(os.listdir(input_dir)) if img.endswith('.jpg')]
    num_images = len(images)

    if num_images < 3:
        print("At least 3 images are required to compute frame-to-frame poses.")
        return

    global_poses = []

    for image in images:
        img = cv2.imread(os.path.join(input_dir, image), cv2.IMREAD_GRAYSCALE)
        R, t = estimate_global_pose(img, cameraMatrix, distCoeffs, patternSize, squareSize)
        #print(R.shape, t.shape)
        global_poses.append((R, t))

    for i in range(1, num_images - 1):
        prev_pose = global_poses[i - 1]
        curr_pose = global_poses[i]
        next_pose = global_poses[i + 1]

        if all(elem is not None for elem in curr_pose) and all(elem is not None for elem in prev_pose):
            R_rel_prev, t_rel_prev = compute_relative_pose(prev_pose[0], prev_pose[1], curr_pose[0], curr_pose[1])

        if all(elem is not None for elem in curr_pose) and all(elem is not None for elem in next_pose):
            R_rel_next, t_rel_next = compute_relative_pose(next_pose[0], next_pose[1], curr_pose[0], curr_pose[1])

        T_rel_prev = create_transformation_matrix(R_rel_prev, t_rel_prev)
        T_rel_next = create_transformation_matrix(R_rel_next, t_rel_next)
        #print(T_rel_next.shape, T_rel_prev.shape)
        transformations = np.array([T_rel_prev, T_rel_next])

        image_index = int(os.path.splitext(images[i])[0])
        output_filename = f"{image_index:04d}.npy"
        #print(transformations.shape)
        np.save(os.path.join(output_dir, output_filename), transformations)

# Load camera intrinsics
cameraMatrix = np.array([[934.5331060961297, 0, 646.162249922997],
                          [0, 934.6172960887559, 358.8661673371651],
                          [0, 0, 1]])
distCoeffs = np.array([-0.000752406145230268, 0.32913244497159155, 0.0001263833868642026, 0.0007148324123980305, -0.7087706474002451])

# Chessboard parameters
patternSize = (11, 8)  # Adjust according to your chessboard size
squareSize = 1.0      # Adjust according to your chessboard square size

# Example usage
input_directory = '../../../datasets/20231204/training/5'
output_directory = './cpose/5'
process_image_sequence(input_directory, output_directory, cameraMatrix, distCoeffs, patternSize, squareSize)
