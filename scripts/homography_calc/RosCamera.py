from typing import List
import rospy
from dataclasses import dataclass, field
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge
import numpy as np
import tf_conversions.posemath as pm


@dataclass
class OpencvWindow:
    win_name: str

    def __post_init__(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

    def show_img(self, img):
        cv2.imshow(self.win_name, img)
        k = cv2.waitKey(1)

        if k == ord("q") or k == 27:
            rospy.signal_shutdown("User exit")


@dataclass
class RosCamera:
    namespace: str = field(default="/depstech/")
    display_img: bool = False
    image_sub: rospy.Subscriber = None
    img: np.ndarray = field(init=False, default=None)
    camera_instrinsic: np.ndarray = field(init=False, default=None)
    camera_distortion: np.ndarray = field(init=False, default=None)
    camera_frame: str = field(init=False, default=None)

    def __post_init__(self):
        self.image_topic = self.namespace + "image_raw"
        # self.image_topic = self.namespace + "image_rect_color"
        self.camera_info_topic = self.namespace + "camera_info"
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self._image_callback)
        self.cv_bridge = CvBridge()

        self.info_subscriber = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self._info_callback
        )

    def _image_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.img = cv_image
        except Exception as e:
            pass

    def _info_callback(self, info_msg):
        self.camera_instrinsic = np.array(info_msg.K).reshape((3, 3))
        self.camera_distortion = np.array(info_msg.D).reshape((-1, 1))

        self.camera_frame = info_msg.header.frame_id

    def print_camera_info(self):
        print(f"Streaming image from: {self.image_topic}")
        print(f"Camera instrinsic: {self.camera_instrinsic}")
        print(f"Camera distortion: {self.camera_distortion}")
        print(f"Camera frame: {self.camera_frame}")

    def wait_until_first_img(self, timeout=5.0):
        start_time = rospy.Time.now()
        while self.img is None:
            rospy.sleep(0.1)

            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logerr("Timeout waiting for first image")
                exit(1)

        return self.img
