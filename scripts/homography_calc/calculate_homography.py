import cv2
import cv2.aruco as aruco
import numpy as np
from HomographyUtils import OpencvCam, ArucoDetector
from HomographyUtils import read_board_config
from RosCamera import RosCamera
import rospy


def main():
    aruco_detector = ArucoDetector()

    # # Opencv Cam
    # cam = OpencvCam()
    # frame = cam.get_frame()
    # cam.close()

    # Ros Cam
    rospy.init_node("aruco_detector")
    cam = RosCamera()
    frame = cam.wait_until_first_img()
    mtx = cam.camera_instrinsic
    dist = cam.camera_distortion

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_detector.detect_markers(gray)
    corners, ids, rejectedImgPoints = aruco_detector.get_last_detected_markers()

    corners_pixel_space = aruco_detector.get_corners_dict()
    corners_world_space = read_board_config("./board_config_files/marker_board2.yaml")

    pts_src = []  # Points in pixel space
    pts_dst = []  # Points in world space

    for key, item in corners_world_space.items():
        pts_dst.append(item)
        pts_src.append(corners_pixel_space[key])

    pts_src = np.concatenate(pts_src, axis=1).squeeze()
    pts_dst = np.concatenate(pts_dst, axis=1).squeeze()

    # The function cv2.undistortPoints() will not generate correct results.
    # You need to use cv2.undistortImagePoints() instead.
    # pts_src_undist = cv2.undistortImagePoints(pts_src, mtx, dist).squeeze()

    # Using undistorted image points should lead to a more accurate homography
    h, status = cv2.findHomography(pts_src, pts_dst) 
    # h, status = cv2.findHomography(pts_src_undist, pts_dst)

    ## Calculate the homography error
    pts_src_homogeneous = np.concatenate(
        [pts_src, np.ones((pts_src.shape[0], 1))], axis=1
    )
    pred = h @ pts_src_homogeneous.T
    pred = pred / pred[-1, :]
    pred = pred[:-1, :]

    residual_error = np.linalg.norm(pred - pts_dst.T, axis=0)
    mean_error = np.mean(residual_error) * 1000
    std_error = np.std(residual_error) * 1000
    print(f"Homography error: {mean_error:.4f} \u00b1 {std_error:.3f} mm")
    
    print("residual error")
    print(residual_error * 1000)
    print("Predicted points")
    print(np.array2string(pred * 1000, separator=",", precision=3, suppress_small=True))
    print("ground truth points")
    print(
        np.array2string(
            pts_dst.T * 1000, separator=",", precision=3, suppress_small=True
        )
    )

    # Optionally, display the frame with detected markers for visual confirmation
    frame_with_markers = aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow("Frame with ArUco markers", frame_with_markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
