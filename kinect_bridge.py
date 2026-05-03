import ctypes
import numpy as np
import cv2
import zmq
from pykinect2 import PyKinectV2, PyKinectRuntime

kinect = PyKinectRuntime.PyKinectRuntime(
    PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
)

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.bind("tcp://*:5555")

print("Bridge running on port 5555. Ctrl+C to stop.")

rgb_count = depth_count = 0
once_rgb = True
once_depth = True
last_depth = None  # latest raw depth frame (512*424 uint16 1-D)

while True:

    if kinect.has_new_depth_frame():
        frame = kinect.get_last_depth_frame()
        depth = frame.reshape((424, 512)).astype(np.uint16)
        last_depth = frame  # keep for colour→depth mapping
        sock.send_multipart([b"depth", depth.tobytes()])
        depth_count += 1
        if once_depth:
            print(f"Depth frames sent: {depth_count}")
            once_depth = False

    if kinect.has_new_color_frame():
        frame = kinect.get_last_color_frame()
        rgb = frame.reshape((1080, 1920, 4))[:, :, :3].astype(np.uint8)
        rgb = cv2.resize(rgb, (960, 540))
        _, buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        sock.send_multipart([b"rgb", buf.tobytes()])
        rgb_count += 1
        if once_rgb:
            print(f"RGB frames sent: {rgb_count}")
            once_rgb = False

        if last_depth is not None:
            color_count = 1920 * 1080
            depth_space_pts = (PyKinectV2._DepthSpacePoint * color_count)()

            kinect._mapper.MapColorFrameToDepthSpace(
                ctypes.c_uint(512 * 424),
                last_depth.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_uint(color_count),
                depth_space_pts
            )

            # (1080, 1920, 2)  — x=col, y=row in depth space; invalid = very negative
            pts = np.frombuffer(depth_space_pts, dtype=np.float32).reshape(1080, 1920, 2)

            x_d = np.round(pts[:, :, 0]).astype(np.int32)
            y_d = np.round(pts[:, :, 1]).astype(np.int32)
            valid = (x_d >= 0) & (x_d < 512) & (y_d >= 0) & (y_d < 424)

            depth_img = last_depth.reshape(424, 512)
            reg = np.zeros((1080, 1920), dtype=np.uint16)
            reg[valid] = depth_img[y_d[valid], x_d[valid]]

            # Resize to 960×540 to match RGB - NEAREST to preserve depth values
            reg_small = cv2.resize(reg.astype(np.float32), (960, 540),
                                   interpolation=cv2.INTER_NEAREST).astype(np.uint16)
            sock.send_multipart([b"depth_reg", reg_small.tobytes()])
