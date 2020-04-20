
set(SOURCES
    utils.hpp
    ransac.hpp
    ransac.cpp)

set(LIBRARIES
    ${OpenCV_LIBS}
    Eigen3::Eigen
    )

add_library(${CMAKE_PROJECT_NAME}_lib ${SOURCES})  # make math into a library
target_link_libraries(${CMAKE_PROJECT_NAME}_lib PRIVATE ${LIBRARIES})
