#!/usr/bin/env python3
"""Offline eye-in-hand re-solve + outlier filtering — no image access.

Trusts calibration_data.npz (stored rvec/tvec/R_g2b/t_g2b assumed consistent).
Reports per-pose residuals, drops outliers, re-solves on inlier subset,
writes best calibration back to eye_in_hand.json.
"""
import json, os, sys, numpy as np, cv2

BASE = os.path.expanduser('~/Vision_Guided_Autonomous_Wound_Treatment_System/calibration')
NPZ  = os.path.join(BASE, 'eye_in_hand_dataset', 'calibration_data.npz')
OUT  = os.path.join(BASE, 'eye_in_hand.json')

SOLVERS = {
    'TSAI':       cv2.CALIB_HAND_EYE_TSAI,
    'PARK':       cv2.CALIB_HAND_EYE_PARK,
    'HORAUD':     cv2.CALIB_HAND_EYE_HORAUD,
    'ANDREFF':    cv2.CALIB_HAND_EYE_ANDREFF,
    'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def ang(R1, R2):
    c = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def make_T(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t).ravel()
    return T


def rmat2rpy_zyx(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        r = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        p = np.degrees(np.arctan2(-R[2, 0], sy))
        y = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        r = np.degrees(np.arctan2(-R[1, 2], R[1, 1])); p = np.degrees(np.arctan2(-R[2, 0], sy)); y = 0
    return r, p, y


def board_in_base_stats(R_c2g, t_c2g, R_g2b, t_g2b, R_t2c, t_t2c):
    X = make_T(R_c2g, t_c2g)
    origins = []
    axes_z = []  # board Z-axis in base (should also be constant)
    for i in range(len(R_g2b)):
        T1 = make_T(R_g2b[i], t_g2b[i])
        T2 = make_T(R_t2c[i], t_t2c[i])
        Tb = T1 @ X @ T2
        origins.append(Tb[:3, 3])
        axes_z.append(Tb[:3, 2])
    origins = np.array(origins)
    axes_z = np.array(axes_z)
    ctr = origins.mean(0)
    dists_mm = np.linalg.norm(origins - ctr, axis=1) * 1000
    axis_mean = axes_z.mean(0)
    axis_mean /= np.linalg.norm(axis_mean) + 1e-9
    axis_dev_deg = np.degrees(np.arccos(np.clip(axes_z @ axis_mean, -1, 1)))
    return origins, dists_mm, axis_dev_deg, axis_mean


def check_board_normal_consistency(R_t2c_list, t_t2c_list):
    """For each pose, check if the board normal points back toward the camera.
    Board local Z-axis in cam = R_t2c[:,2]. Camera looks at +Z. Normal should
    point back at camera, so R_t2c[:,2].z should be negative (points toward cam).
    Also check angle between (cam→board) and (board normal into cam): should be
    close to 180° (normals antialigned — board faces camera)."""
    bad = []
    for i, (R, t) in enumerate(zip(R_t2c_list, t_t2c_list)):
        board_z_in_cam = R[:, 2]
        to_board = t.ravel() / (np.linalg.norm(t) + 1e-9)
        # board normal pointing away from surface plus cam→board vec: dot should be ≈ +1 for facing-camera view
        # (board_z_in_cam ≈ -to_board when board faces camera squarely)
        cos_a = float(np.dot(board_z_in_cam, -to_board))
        if cos_a < 0.2:
            bad.append((i, cos_a))
    return bad


def run_solver(name, method, R_g2b, t_g2b, R_t2c, t_t2c):
    R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    origins, dists_mm, ax_dev_deg, _ = board_in_base_stats(
        R_c2g, t_c2g, R_g2b, t_g2b, R_t2c, t_t2c)
    return {
        'R': R_c2g,
        't': t_c2g,
        'origins': origins,
        'board_spread_mm': float(np.linalg.norm(origins.max(0) - origins.min(0)) * 1000),
        'board_origin_std_mm': (origins.std(0) * 1000).tolist(),
        'board_origin_median_m': np.median(origins, axis=0).tolist(),
        'per_pose_dist_mm': dists_mm,
        'per_pose_axis_deg': ax_dev_deg,
        'axis_dev_mean_deg': float(ax_dev_deg.mean()),
        'axis_dev_max_deg': float(ax_dev_deg.max()),
    }


def leave_one_out_scores(R_g2b, t_g2b, R_t2c, t_t2c, method):
    """For each pose i, run calib on remaining poses and measure board spread.
    Poses whose removal drops spread most are the outliers."""
    n = len(R_g2b)
    baseline = run_solver('base', method, R_g2b, t_g2b, R_t2c, t_t2c)['board_spread_mm']
    scores = []
    for i in range(n):
        rg = [R_g2b[k] for k in range(n) if k != i]
        tg = [t_g2b[k] for k in range(n) if k != i]
        rt = [R_t2c[k] for k in range(n) if k != i]
        tt = [t_t2c[k] for k in range(n) if k != i]
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(rg, tg, rt, tt, method=method)
            origins, _, _, _ = board_in_base_stats(R_c2g, t_c2g, rg, tg, rt, tt)
            sp = float(np.linalg.norm(origins.max(0) - origins.min(0)) * 1000)
        except Exception:
            sp = 1e9
        scores.append(baseline - sp)   # >0 means removing i IMPROVED fit
    return np.array(scores)


def greedy_inlier_selection(R_g2b, t_g2b, R_t2c, t_t2c, method, target_spread_mm=50.0, min_keep=12):
    """Iteratively drop the worst pose (by per-pose board distance from median)
    until spread <= target or only min_keep remain."""
    n = len(R_g2b)
    keep = list(range(n))
    history = []
    while len(keep) > min_keep:
        rg = [R_g2b[i] for i in keep]
        tg = [t_g2b[i] for i in keep]
        rt = [R_t2c[i] for i in keep]
        tt = [t_t2c[i] for i in keep]
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(rg, tg, rt, tt, method=method)
        except Exception:
            break
        origins, dists_mm, ax_dev_deg, _ = board_in_base_stats(
            R_c2g, t_c2g, rg, tg, rt, tt)
        sp = float(np.linalg.norm(origins.max(0) - origins.min(0)) * 1000)
        history.append({
            'n': len(keep), 'spread_mm': sp,
            'axis_dev_max_deg': float(ax_dev_deg.max()),
            'dropped_idx': None,
        })
        if sp <= target_spread_mm:
            break
        # drop the pose with the largest per-pose board distance from median
        med = np.median(origins, axis=0)
        dd = np.linalg.norm(origins - med, axis=1)
        worst_local = int(np.argmax(dd))
        worst_global = keep[worst_local]
        history[-1]['dropped_idx'] = worst_global
        keep.remove(worst_global)
    return keep, history


def main():
    if not os.path.isfile(NPZ):
        print(f'NO DATASET at {NPZ}'); sys.exit(1)
    d = np.load(NPZ)
    R_g2b = [d['R_g2b'][i] for i in range(len(d['R_g2b']))]
    t_g2b = [d['t_g2b'][i] for i in range(len(d['t_g2b']))]
    R_t2c = [d['R_t2c'][i] for i in range(len(d['R_t2c']))]
    t_t2c = [d['t_t2c'][i] for i in range(len(d['t_t2c']))]
    K, dist = d['K'], d['dist']
    n = len(R_g2b)
    print(f'Dataset: {n} poses   K_fx={K[0,0]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}\n')
    print('── BOARD-NORMAL CONSISTENCY (cheap sanity) ──────────────────')
    bad = check_board_normal_consistency(R_t2c, t_t2c)
    if bad:
        print(f'  {len(bad)}/{n} poses with board normal NOT facing camera:')
        for i, c in bad:
            print(f'    pose {i:3d}  cos(angle)={c:+.3f}  (negative = board back-facing → solvePnP picked wrong planar sol)')
    else:
        print(f'  all {n} poses have board facing camera — planar-ambiguity unlikely')
    print()
    print('── ALL SOLVERS (full dataset) ──────────────────────────────────')
    print(f'{"solver":<12}{"tx(mm)":>9}{"ty(mm)":>9}{"tz(mm)":>9}'
          f'{"roll":>8}{"pitch":>8}{"yaw":>8}{"spread_mm":>12}{"axdev_max":>12}')
    full_results = {}
    for name, m in SOLVERS.items():
        try:
            r = run_solver(name, m, R_g2b, t_g2b, R_t2c, t_t2c)
        except Exception as e:
            print(f'  {name:<12} FAIL: {e}'); continue
        roll, pitch, yaw = rmat2rpy_zyx(r['R'])
        tx, ty, tz = r['t'].ravel() * 1000
        print(f'  {name:<12}{tx:9.1f}{ty:9.1f}{tz:9.1f}'
              f'{roll:8.1f}{pitch:8.1f}{yaw:8.1f}{r["board_spread_mm"]:12.1f}{r["axis_dev_max_deg"]:12.1f}')
        full_results[name] = r
    print()
    best_name = min(full_results, key=lambda k: full_results[k]['board_spread_mm'])
    print(f'Best full-data solver: {best_name} — {full_results[best_name]["board_spread_mm"]:.1f} mm spread\n')
    print('── LEAVE-ONE-OUT (improvement if pose removed, mm) ─────────────')
    scores = leave_one_out_scores(R_g2b, t_g2b, R_t2c, t_t2c, SOLVERS[best_name])
    order = np.argsort(scores)[::-1]
    for k in order[:8]:
        marker = '!!! OUTLIER' if scores[k] > 100 else '' if scores[k] < 20 else '    marginal'
        print(f'  pose {k:3d}  drop improves spread by {scores[k]:+8.1f} mm  {marker}')
    print()
    print('── GREEDY INLIER SELECTION (drop worst until spread ≤50 mm or n≤12) ──')
    best_overall = None
    for name, m in SOLVERS.items():
        try:
            keep, hist = greedy_inlier_selection(R_g2b, t_g2b, R_t2c, t_t2c, m,
                                                 target_spread_mm=50.0, min_keep=12)
        except Exception as e:
            print(f'  {name:<12} FAIL: {e}'); continue
        rg = [R_g2b[i] for i in keep]
        tg = [t_g2b[i] for i in keep]
        rt = [R_t2c[i] for i in keep]
        tt = [t_t2c[i] for i in keep]
        r = run_solver(name, m, rg, tg, rt, tt)
        roll, pitch, yaw = rmat2rpy_zyx(r['R'])
        tx, ty, tz = r['t'].ravel() * 1000
        dropped = [h['dropped_idx'] for h in hist if h['dropped_idx'] is not None]
        print(f'  {name:<12}  kept={len(keep):2d}  spread={r["board_spread_mm"]:6.1f} mm  '
              f'axdev_max={r["axis_dev_max_deg"]:5.1f}°  '
              f't=[{tx:+7.1f},{ty:+7.1f},{tz:+7.1f}]  '
              f'rpy=[{roll:+6.1f},{pitch:+6.1f},{yaw:+6.1f}]  dropped={dropped}')
        if best_overall is None or r['board_spread_mm'] < best_overall['spread']:
            best_overall = {
                'solver': name, 'keep': keep, 'R': r['R'], 't': r['t'],
                'spread': r['board_spread_mm'], 'axdev_max': r['axis_dev_max_deg'],
                'origin_std_mm': r['board_origin_std_mm'],
                'rpy': (roll, pitch, yaw),
                'dropped': dropped,
            }
    print()
    if best_overall is None:
        print('Nothing converged.'); return
    print('── SELECTED CALIBRATION ────────────────────────────────────────')
    R = best_overall['R']; t = best_overall['t'].ravel()
    print(f'  solver     {best_overall["solver"]}')
    print(f'  n_poses    {len(best_overall["keep"])} (dropped {best_overall["dropped"]})')
    print(f'  spread     {best_overall["spread"]:.2f} mm')
    print(f'  axdev_max  {best_overall["axdev_max"]:.2f}°')
    print(f'  origin_std {np.round(best_overall["origin_std_mm"],2).tolist()} mm')
    print(f'  t (m)      {t.round(4).tolist()}')
    print(f'  rpy (deg)  ({best_overall["rpy"][0]:.3f}, {best_overall["rpy"][1]:.3f}, {best_overall["rpy"][2]:.3f})')
    print(f'  R =\n{np.round(R, 4)}')

    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    rvec, _ = cv2.Rodrigues(R)
    data = {
        'solver': f'{best_overall["solver"]}_inlier_filtered',
        'consistency': float(best_overall['spread']),
        'n_poses_used': len(best_overall['keep']),
        'n_poses_raw': n,
        'dropped_poses': best_overall['dropped'],
        'notes': (f'Offline re-solve on existing dataset (no image re-analysis). '
                  f'Board square=23mm assumed internally consistent with stored rvec/tvec. '
                  f'Board spread {best_overall["spread"]:.1f} mm on inlier set. '
                  f'axis_dev_max {best_overall["axdev_max"]:.2f}°'),
        'T_cam_to_tool0': T.tolist(),
        'R_cam_to_tool0': R.tolist(),
        't_cam_to_tool0_m': t.tolist(),
        'rodrigues_cam_to_tool0': rvec.ravel().tolist(),
    }
    out_path = os.path.join(BASE, 'eye_in_hand_offline_v2.json')
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\nWrote {out_path}')
    print('(Not overwriting eye_in_hand.json — inspect, then copy if you accept.)')


if __name__ == '__main__':
    main()
