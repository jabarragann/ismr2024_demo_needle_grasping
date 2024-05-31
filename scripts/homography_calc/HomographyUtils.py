from typing import Dict, List, Tuple
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
import cv2
from cv2 import aruco
from packaging import version


@dataclass
class OpencvCam:
    device_id: int = field(default=0)

    def __post_init__(self):
        # Initialize the camera
        self.cap = cv2.VideoCapture(self.device_id)  # 0 is the default camera

        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            exit(3) 

    def get_frame(self):
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            print("Error: Could not read frame.")
            return

        return self.frame

    def close(self):
        self.cap.release()


@dataclass
class ArucoDetector:
    current_corners: Tuple[np.ndarray] = field(default=None, init=False)
    current_ids: np.ndarray = field(default=None, init=False)
    current_rejected: Tuple[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        # aruco_dict_type = aruco.DICT_4X4_250
        aruco_dict_type = aruco.DICT_6X6_250
        if version.parse(cv2.__version__) >= version.parse("4.7.0"):
            self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
            self.parameters = aruco.DetectorParameters()
        else:
            self.aruco_dict = aruco.Dictionary_get(aruco_dict_type)
            self.parameters = aruco.DetectorParameters_create()

    def detect_markers(self, gray: np.ndarray):
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )
        self.current_corners = corners
        self.current_ids = ids
        self.current_rejected = rejectedImgPoints

    def get_corners_dict(self) -> Dict[int, List[float]]:
        corners_dict = {}
        for marker_corners, marker_id in zip(self.current_corners, self.current_ids):
            corners_dict[marker_id[0]] = marker_corners
        return corners_dict

    def get_last_detected_markers(self):
        return self.current_corners, self.current_ids, self.current_rejected


def read_board_config(board_config: Path) -> Dict[int, List[float]]:

    with open(board_config, "r") as file:
        board = yaml.safe_load(file)

    id_list = board["ids"]

    board_config = {}
    for id in id_list:
        corners = board[id]
        # Use the same format as the aruco.detectMarkers function
        corners = np.expand_dims(np.array(corners, dtype=np.float32), 0)
        board_config[id] = corners / 100 # Convert from cm to m

    return board_config


def test_read_board_config():
    board_config = read_board_config("./board_config_files/marker_board1.yaml")
    print(board_config)


if __name__ == "__main__":
    test_read_board_config()
