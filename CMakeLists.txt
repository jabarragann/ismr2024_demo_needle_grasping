cmake_minimum_required(VERSION 3.0.2)
project(ismr2024_demo_needle_grasping)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
