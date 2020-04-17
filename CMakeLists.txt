cmake_minimum_required(VERSION 3.10)
project(ransac_homography C CXX)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

add_subdirectory(lib/googletest)
add_subdirectory(src)

set(SOURCES
    main.cpp)

set(LIBRARIES
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${PROJECT_NAME}_lib  #define at src
    )

include_directories(${EIGEN_3_INCLUDE_DIR})
message("${CMAKE_MODULE_PATH}")
message("${EIGEN3_INCLUDE_DIR}")

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE src)

