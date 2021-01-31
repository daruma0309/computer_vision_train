cmake_minimum_required(VERSION 3.0.2)
project(testF CXX)
add_compile_options(-std=c++11)

add_executable(testF src/main.cpp)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

message(STATUS ${OpenCV_INCLUDE_DIRS})

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)
