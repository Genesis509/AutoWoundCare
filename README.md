# Vision-Guided Autonomous Wound Treatment System

A ROS 2 system that uses a UR16e robot arm with an eye-in-hand camera to autonomously detect and approach wounds on a patient (or mannequin). The pipeline detects the target via RGB-D perception, estimates a 6-DOF approach pose, and executes motion via MoveIt 2.

**Full project report:** https://drive.google.com/file/d/1BEdPOoK6qvDuHBnmoPhtkX-xuHr8MZbR/view?usp=sharing

**Demo Hardware scanning** : https://drive.google.com/file/d/1if5IiHX_DS2_YyPrlev_uhzaMi7W8HCi/view?usp=sharing

**Demo Sim Treatement** : https://drive.google.com/file/d/1fNuGj6bstz0FDGGH4PEUx_nsxKha3ppO/view?usp=sharing

---

### System Architecture

* **Perception:** A `Camera` (Kinect 2/Sim) feeds into `yolo_wound_detector` (isolates wound) and `yolo_pose` (SVD plane fit for 6-DOF pose). A `skeleton_tracker` (MediaPipe) monitors patient anatomy.
* **Scanning:** Using tracking data, `scan_pose_publisher` calculates 3 optimal viewpoints. The `scan_loop` executes this trajectory sequence via MoveIt.
* **Reporting:** The `wound_report_collector` captures and deduplicates images during the scan, passing data to `generate_wound_report` to output a final PDF.
* **Orchestration:** A high-level FSM (`IDLE` → `APPROACH` → `LOCKING`) governs behavior. The `approach_node` navigates the robot to the target, while the `locking_node` maintains position and tracks.

---

## Hardware Setup (what was used)

- **Robot:** Universal Robots UR16e
- **Camera:** Microsoft Kinect 2 (eye-in-hand, mounted on tool0)
- **Host OS:** Windows 11 + WSL2 (Ubuntu 24)
- **ROS 2:** Jazzy
- **MoveIt 2:** Jazzy

### Why ZMQ?

The Kinect 2 SDK only runs on Windows. The solution is a two-part bridge:

1. **Windows side** : run `tools/kinect_bridge.py` (requires `pykinect2`, `zmq`, `opencv-python`).  
   It captures RGB, depth, and registered-depth frames and publishes them over ZMQ on port 5555.

2. **WSL2 side**: `kinect2_driver_node` connects to `127.0.0.1:5555` (WSL2 mirrored-networking mode) and republishes as standard ROS 2 camera topics.

```
Windows: kinect_bridge.py  ──ZMQ──►  WSL2: kinect2_driver_node ──► /camera/...
```

### Using a Different Camera

If you are **not** using Kinect 2 / Windows, replace `kinect2_driver_node` with your own driver. The rest of the pipeline only requires these three topics:

| Topic | Type | Notes |
|-------|------|-------|
| `/camera/color/image_raw` | `sensor_msgs/Image` (BGR8) | 960×540 |
| `/camera/depth_registered/image_raw` | `sensor_msgs/Image` (16UC1, mm) | aligned to colour |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | intrinsics |

Update `calibration/rgb_intrinsics.json` with your camera's intrinsics and re-run the eye-in-hand calibration (`scripts/calibrate_eye_in_hand.py`).

---

## Required Model Files

Place these in `src/wound_tracking/models/` before building:

| File | Purpose |
|------|---------|
| `best.pt` | YOLO wound detection model (trained on wound dataset) |
| `pose_landmarker.task` | MediaPipe pose landmarker (download from MediaPipe) |

---

## Build

```bash
cd ~/Vision_Guided_Autonomous_Wound_Treatment_System
colcon build --packages-select wound_tracking
source install/setup.bash
```

---

## Simulation - Mannequin Scene

Launches Gazebo with a humanoid mannequin on a table, a red wound patch on the chest, UR16e arm, MoveIt 2, and the full perception + FSM pipeline.

```bash
# Build first (see above), then:
ros2 launch wound_tracking sim_mannequin.launch.py
```

What starts automatically:
- Gazebo Harmonic with mannequin world
- UR16e ros2_control + joint trajectory controller
- MoveIt 2 move_group + RViz
- Camera bridge (Gz topics → ROS 2)
- `feature_detector` + `pose_estimation` (delayed 8 s for Gz to stabilise)
- `orchestrator` + `approach_node` + `locking_node`

The robot will idle until a red wound is detected in the camera view, buffer 7 detections, then plan and execute an approach.

**Other simulation launches:**

```bash
# Plain red cube instead of mannequin
ros2 launch wound_tracking sim.launch.py

# Motion-only test with hardcoded scan poses (no perception)
ros2 launch wound_tracking sim_motion.launch.py

# Preview scan arc targets in RViz (no motion)
ros2 launch wound_tracking scan_arc_preview.launch.py
```

---

## Real Hardware - Full Scan Pipeline

```bash
# Terminal 1 - Windows (PowerShell or CMD)
python tools/kinect_bridge.py

# Terminal 2 - WSL2: UR driver
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur16e robot_ip:=<ROBOT_IP>

# Terminal 3 - WSL2: MoveIt 2
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur16e launch_rviz:=true

# Terminal 4 - WSL2: full perception + scan + report
ros2 launch wound_tracking full_pipeline.launch.py
```

The scan sequence:
1. Robot starts at home pose facing the patient
2. `skeleton_tracker` detects the patient and computes 3 viewpoints (left, centre, right)
3. `scan_loop` moves to each viewpoint in order, pausing 5 s at each
4. `wound_report_collector` captures YOLO detections at each viewpoint, deduplicates across views
5. A PDF report is written to `~/wound_reports/<session_id>/report.pdf`

**Other real-hardware launches:**

```bash
# Perception only (no motion) - good for tuning detection
ros2 launch wound_tracking pose_detection.launch.py robot_ip:=<IP>

# YOLO locking pipeline (approach + hold on wound)
ros2 launch wound_tracking real_mannequin.launch.py

# Full real launch with HSV detector (no YOLO)
ros2 launch wound_tracking real.launch.py robot_ip:=<IP>
```

---

## Calibration Files

| File | Description |
|------|-------------|
| `calibration/eye_in_hand.json` | Hand-eye transform: tool0 → camera_color_optical_frame |
| `calibration/rgb_intrinsics.json` | Kinect 2 RGB intrinsics (fx, fy, cx, cy, dist) |

Run `scripts/calibrate_eye_in_hand.py` to regenerate `eye_in_hand.json`.  
`calibrated_tf_publisher_node` loads this at startup and publishes the static TF. If the file is missing it falls back to approximate values with a warning.

---

## Dependencies

```bash
# ROS 2 Humble packages
sudo apt install ros-humble-ur ros-humble-ur-moveit-config \
    ros-humble-moveit ros-humble-ros-gz-bridge

# Python
pip install ultralytics mediapipe reportlab pyzmq opencv-python numpy
```
