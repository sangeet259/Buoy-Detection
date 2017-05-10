cmake_minimum_required(VERSION 2.8.3)
project(opencv)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  std_msgs
  sensor_msgs
)

find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS cv_bridge image_transport roscpp std_msgs sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(denoise src/denoise.cpp)
target_link_libraries( denoise ${OpenCV_LIBS})

add_executable(kalmanfilter src/kalmanfilter.cpp)
target_link_libraries(kalmanfilter ${OpenCV_LIBS})

add_executable(cfarAlgo src/cfarAlgo.cpp)
target_link_libraries(cfarAlgo ${OpenCV_LIBS})

add_executable(CFARnKF src/CFARnKF.cpp)
target_link_libraries(CFARnKF ${OpenCV_LIBS})

add_executable(eliminate src/eliminate.cpp)
target_link_libraries(eliminate ${OpenCV_LIBS})

add_executable(detect src/detect.cpp)
target_link_libraries(detect  ${OpenCV_LIBRARIES}  ${catkin_LIBRARIES})

add_executable(publisher src/publisher.cpp)
target_link_libraries(publisher  ${OpenCV_LIBRARIES}  ${catkin_LIBRARIES})

add_executable(subscriber src/subscriber.cpp)
target_link_libraries(subscriber  ${OpenCV_LIBRARIES}  ${catkin_LIBRARIES})
