set(SOURCES
    main.cpp
    ransacHomographyTest.cpp
)

set(LIBRARIES
    gtest
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${CMAKE_PROJECT_NAME}_lib  #link to ransac_lib define in src
)

add_executable(${CMAKE_PROJECT_NAME}_tests ${SOURCES})
add_test(NAME ${CMAKE_PROJECT_NAME}_tests COMMAND ${CMAKE_PROJECT_NAME}_tests)
target_link_libraries(${CMAKE_PROJECT_NAME}_tests PUBLIC ${LIBRARIES}) 
target_include_directories(${CMAKE_PROJECT_NAME}_tests PRIVATE ../src) # header visibility