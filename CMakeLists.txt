cmake_minimum_required(VERSION 2.8)
project(ransac_homography C CXX)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

set(HEADERS
    ransac.h)

set(SOURCES
    main.cpp)

set(LIBRARIES
    ${OpenCV_LIBS}
    Eigen3::Eigen)

include_directories(${EIGEN_3_INCLUDE_DIR})
message("${CMAKE_MODULE_PATH}")
message("${EIGEN3_INCLUDE_DIR}")

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})


