import cv2
import cv2.aruco as aruco
from HomographyUtils import OpencvCam


def main():
    cam = OpencvCam()
    frame = cam.get_frame()
    cam.close()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the predefined dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

    # Initialize the detector parameters using default values
    parameters = aruco.DetectorParameters()

    # Detect the markers in the image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )

    # Check if any markers were detected
    if ids is not None:
        # Print the corners of each detected marker
        for marker_corners, marker_id in zip(corners, ids):
            print(f"Marker ID: {marker_id[0]}")
            for corner in marker_corners:
                for point in corner:
                    print(f"Corner: {point}")
    else:
        print("No ArUco markers detected.")

    # Optionally, display the frame with detected markers for visual confirmation
    frame_with_markers = aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow("Frame with ArUco markers", frame_with_markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
