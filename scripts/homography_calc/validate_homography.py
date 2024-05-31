import cv2
import cv2
import cv2.aruco as aruco
import numpy as np
from HomographyUtils import OpencvCam, ArucoDetector
from HomographyUtils import read_board_config
from RosCamera import RosCamera
import rospy
from dataclasses import dataclass

marker_offset = np.array([[1, 0, 0,-1.5/100],
                          [0,-1, 0, 1.5/100],
                          [0, 0,-1, 0.0],
                          [0, 0, 0, 1.0]])

@dataclass
class ImageAnnotator:
    img: np.ndarray

    def __post_init__(self):
       cv2.imshow("image", self.img) 
       cv2.setMouseCallback("image", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at pixel location: ({x}, {y})")
            cv2.circle(self.img, (x, y), 5, (0, 0, 255), -1)  # Draw a red circle at the clicked point
            cv2.imshow("image", self.img)

            self.pixel_hom = np.array([[x,y,1]]).T

def main():

    aruco_detector = ArucoDetector()
    rospy.init_node("aruco_detector")
    cam = RosCamera()
    frame = cam.wait_until_first_img()
    mtx = cam.camera_instrinsic
    dist = cam.camera_distortion
    marker_length = 3/100

    # Image annotator
    img_annotator = ImageAnnotator(frame)
    homography = np.load("homography.npy")


    # Get pose of first marker
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_detector.detect_markers(gray)
    corners_pixel_space = aruco_detector.get_corners_dict()
    first_marker = corners_pixel_space[0]

    corners, ids, rejectedImgPoints = aruco_detector.get_last_detected_markers()
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(first_marker, marker_length, mtx, dist)


    # Wait for the user to press any key to exit
    cv2.waitKey(0)

    print(img_annotator.pixel_hom)
    print(img_annotator.pixel_hom.shape)
    pred = homography @ img_annotator.pixel_hom
    pred = pred / pred[-1, :]
    pred = pred[:-1, :]
    print(pred * 100)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # TODO: Send point to camera frame, reproject to img and measure reprojection error.

if __name__ == "__main__":
    main()