cmake_minimum_required(VERSION 3.14)

# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-std=c++14")

#-DEIGEN_USE_MKL_ALL")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
# for debug program
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -g")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)
find_package(GTest REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
find_package(SuiteSparse REQUIRED)


include_directories( PUBLIC
                    include
                    ${EIGEN3_INCLUDE_DIR}
                    ${GTEST_INCLUDE_DIRS}
                    ${Pangolin_INCLUDE_DIRS}
                    ${Sophus_INCLUDE_DIRS}
                    ${SUITESPARSE_INCLUDE_DIRS}
)

add_executable(camera_calibparam_test 
                gtest/camera_calibparam_test.cpp
                src/frame.cpp 
                src/camera.cpp)

target_link_libraries(camera_calibparam_test ${OpenCV_LIBS} ${GLOG_LIBRARIES} ${GTEST_BOTH_LIBRARIES}
                        ${CMAKE_THREAD_LIBS_INIT}
                        ${Pangolin_LIBRARIES})

add_executable(img_processor 
                src/frame.cpp 
                src/camera.cpp
                src/imu_measure.cpp
                src/map_feature.cpp
                src/img_processor.cpp
                src/msckf_filter.cpp
                gtest/img_processor_test.cpp)

target_link_libraries(img_processor ${OpenCV_LIBS} ${GLOG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${GTEST_BOTH_LIBRARIES}
                         ${Pangolin_LIBRARIES}   ${SUITESPARSE_LIBRARIES})


add_executable(msckf_filter 
    src/frame.cpp 
    src/camera.cpp
    src/imu_measure.cpp
    src/map_feature.cpp
    src/img_processor.cpp
    src/msckf_filter.cpp
    gtest/msckf_filter_test.cpp)

target_link_libraries(msckf_filter ${OpenCV_LIBS} ${GLOG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${GTEST_BOTH_LIBRARIES}
                         ${Pangolin_LIBRARIES} ${SUITESPARSE_LIBRARIES} pthread)


add_executable(msckf_filter_func_test 
    src/frame.cpp 
    src/camera.cpp
    src/imu_measure.cpp
    src/map_feature.cpp
    src/img_processor.cpp
    src/msckf_filter.cpp
    gtest/msckf_filter_func_test.cpp)

target_link_libraries(msckf_filter_func_test ${OpenCV_LIBS} ${GLOG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${GTEST_BOTH_LIBRARIES}
                         ${Pangolin_LIBRARIES}   ${SUITESPARSE_LIBRARIES} pthread)