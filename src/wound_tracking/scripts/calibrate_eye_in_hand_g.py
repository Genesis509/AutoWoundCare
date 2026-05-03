#!/usr/bin/env python3
"""
Offline Eye-in-Hand Recalculation (Fixed ROS Inversion)
Uses SQPnP to bypass planar ambiguity without manual filtering,
and mathematically inverts the ROS TF matrices to satisfy OpenCV's solver.
"""
import json
import os
import cv2
import numpy as np

BASE_DIR   = os.path.expanduser('~/Vision_Guided_Autonomous_Wound_Treatment_System/calibration')
INTR_FILE  = os.path.join(BASE_DIR, 'rgb_intrinsics.json')
DATA_DIR   = os.path.join(BASE_DIR, 'eye_in_hand_dataset')
POSES_FILE = os.path.join(DATA_DIR, 'poses.json')
OUT_FILE   = os.path.join(BASE_DIR, 'eye_in_hand_fixed.json')

BOARD_W, BOARD_H = 9, 6
SQUARE_M = 0.025
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def main():
    with open(INTR_FILE) as f: intr = json.load(f)
    K = np.array(intr['rgb_K'], dtype=np.float64)
    dist = np.array(intr['rgb_dist'], dtype=np.float64)

    with open(POSES_FILE) as f: poses_data = json.load(f)

    objp = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_M

    R_b2g_list, t_b2g_list = [], []
    R_t2c_list, t_t2c_list = [], []
    valid_count = 0

    print("Re-evaluating images with SQPnP and Matrix Inversion...")

    for item in poses_data:
        img_path = os.path.join(DATA_DIR, item['image_file'])
        if not os.path.exists(img_path): continue
        
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        found, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), 
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)

        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            
            # SQPnP handles planar targets optimally without needing the manual normal filter
            ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_SQPNP)
            
            if ok:
                R_board, _ = cv2.Rodrigues(rvec)
                R_t2c_list.append(R_board)
                t_t2c_list.append(tvec)

                # 1. Load ROS standard TF (tool0 -> base_link)
                R_g2b = np.array(item['R_gripper2base'])
                t_g2b = np.array(item['t_gripper2base']).reshape(3, 1)
                
                T_g2b = np.eye(4)
                T_g2b[:3, :3] = R_g2b
                T_g2b[:3, 3] = t_g2b.ravel()
                
                # 2. INVERT THE MATRIX for OpenCV (base_link -> tool0)
                T_b2g = np.linalg.inv(T_g2b)
                
                R_b2g_list.append(T_b2g[:3, :3])
                t_b2g_list.append(T_b2g[:3, 3].reshape(3, 1))
                valid_count += 1

    print(f"Successfully extracted {valid_count} poses.")
    if valid_count < 4: return

    # Compute Eye-in-Hand Transformation
    R_cam2tool, t_cam2tool = cv2.calibrateHandEye(
        R_b2g_list, t_b2g_list, R_t2c_list, t_t2c_list, method=cv2.CALIB_HAND_EYE_PARK
    )

    T_c2g = np.eye(4)
    T_c2g[:3, :3] = R_cam2tool
    T_c2g[:3, 3] = t_cam2tool.ravel()
    rvec_out, _ = cv2.Rodrigues(R_cam2tool)

    out_data = {
        'solver': 'PARK_SQPNP_INVERTED',
        'n_poses_used': valid_count,
        'T_cam_to_tool0': T_c2g.tolist(),
        'R_cam_to_tool0': R_cam2tool.tolist(),
        't_cam_to_tool0_m': t_cam2tool.ravel().tolist(),
        'rodrigues_cam_to_tool0': rvec_out.ravel().tolist(),
    }

    with open(OUT_FILE, 'w') as f: json.dump(out_data, f, indent=2)
    print(f"\nCalibration fixed and saved to {OUT_FILE}")
    print(f"Translation (X, Y, Z in meters): {t_cam2tool.ravel().round(4)}")

if __name__ == '__main__':
    main()