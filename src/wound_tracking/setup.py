from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'wound_tracking'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'),   glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('lib', package_name, 'models'), ['models/best.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='genesis',
    maintainer_email='todo@todo.com',
    description='Vision-based target tracking with UR16e — sim and real',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_driver      = wound_tracking.camera_driver_node:main',
            'feature_detector   = wound_tracking.feature_detector_node:main',
            'pose_estimation    = wound_tracking.pose_estimation_node:main',
            'motion_controller  = wound_tracking.motion_controller_node:main',
            'orchestrator       = wound_tracking.orchestrator_node:main',
            'approach_node      = wound_tracking.approach_node:main',
            'locking_node       = wound_tracking.locking_node:main',
            'view_detection     = wound_tracking.view_detection:main',
            'isaac_trajectory_bridge = wound_tracking.isaac_trajectory_bridge:main',
            'kinect2_driver          = wound_tracking.kinect2_driver_node:main',
            'calibrated_tf_publisher = wound_tracking.calibrated_tf_publisher_node:main',
            'tune_eye_in_hand        = wound_tracking.tune_eye_in_hand:main',
            'skeleton_tracker = wound_tracking.skeleton_tracker:main',
            'yolo_detect = wound_tracking.yolo_detect:main',
            'yolo_pose              = wound_tracking.yolo_pose_node:main',
            'yolo_wound_detector    = wound_tracking.yolo_wound_detector_node:main',
            'wound_report_collector = wound_tracking.wound_report_collector_node:main',
            'scan_pose_publisher    = wound_tracking.scan_pose_publisher_node:main',
            'scan_pose_arc          = wound_tracking.scan_pose_arc_node:main',
            'scan_loop              = wound_tracking.scan_loop_node:main',
            'fake_scan_targets      = wound_tracking.fake_scan_targets_node:main',
        ],
    },
)
