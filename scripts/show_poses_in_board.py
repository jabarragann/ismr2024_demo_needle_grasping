#!/usr/bin/env python3

import rospy
from dataclasses import dataclass, field
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


@dataclass
class ImageSubscriber:

    image_topic: str = field(default="/usb_cam/image_raw")
    display_img: bool = False
    image_sub: rospy.Subscriber = None

    def __post_init__(self):
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.cv_bridge = CvBridge()
        self.win_name = f"overlay {self.image_topic}" 

        if self.display_img:
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            # cv2.startWindowThread()

    def image_callback(self, msg):

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.display_img:
                cv2.imshow(self.win_name, cv_image)
                k = cv2.waitKey(1)

                ## This is not working in my PC 
                if k == ord("q") or k == 27:
                    print("user exit")
                    rospy.signal_shutdown("User exit")

        except Exception as e:
            pass


def main():
    rospy.init_node("juan_subscriber")

    image_sub = ImageSubscriber(display_img=True)

    while not rospy.is_shutdown():
        rospy.spin()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
