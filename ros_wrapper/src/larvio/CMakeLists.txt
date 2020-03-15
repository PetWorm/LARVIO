cmake_minimum_required(VERSION 2.8.12)
project(larvio)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++11)

# Modify cmake module path if new .cmake files are required
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/../../../cmake")
set(LARVIO_SRC "${CMAKE_CURRENT_LIST_DIR}/../../../src")
set(LARVIO_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/../../../include")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    tf
    nav_msgs
    sensor_msgs
    geometry_msgs
    eigen_conversions
    tf_conversions
    nodelet
    image_transport
    cv_bridge
    message_filters
    pcl_conversions
    pcl_ros
    std_srvs
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(Ceres REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
    INCLUDE_DIRS include ${LARVIO_INCLUDE}
    LIBRARIES system_manager estimator image_processor
    CATKIN_DEPENDS
        roscpp std_msgs tf nav_msgs sensor_msgs geometry_msgs
        eigen_conversions tf_conversions message_runtime
        image_transport cv_bridge message_filters pcl_conversions
        pcl_ros std_srvs
    DEPENDS Boost EIGEN3 OpenCV SUITESPARSE
)

###########
## Build ##
###########

include_directories(
    include
    ${LARVIO_INCLUDE}
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${SUITESPARSE_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

# Image processor
add_library(image_processor
    ${LARVIO_SRC}/image_processor.cpp
    ${LARVIO_SRC}/ORBDescriptor.cpp
)
target_link_libraries(image_processor
    ${OpenCV_LIBRARIES}
)

# Initializer
# -- Static Initializer
add_library(staticInitializer
    ${LARVIO_SRC}/StaticInitializer.cpp
)
target_link_libraries(staticInitializer
    ${SUITESPARSE_LIBRARIES}
)
# -- Dynamic Initializer
add_library(dynamicInitializer
    ${LARVIO_SRC}/StaticInitializer.cpp
    ${LARVIO_SRC}/DynamicInitializer.cpp
    ${LARVIO_SRC}/feature_manager.cpp
    ${LARVIO_SRC}/initial_alignment.cpp
    ${LARVIO_SRC}/initial_sfm.cpp
    ${LARVIO_SRC}/solve_5pts.cpp
)
target_link_libraries(dynamicInitializer
    ${SUITESPARSE_LIBRARIES}
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
# -- Flexible Initializer
add_library(flexibleInitializer
    ${LARVIO_SRC}/FlexibleInitializer.cpp
)
target_link_libraries(flexibleInitializer
    staticInitializer
    dynamicInitializer
)

# Estimator
add_library(estimator
    ${LARVIO_SRC}/larvio.cpp
)
target_link_libraries(estimator
    flexibleInitializer
    ${SUITESPARSE_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

# System Manger
add_library(system_manager
    src/System.cpp
    ${LARVIO_SRC}/larvio.cpp
    ${LARVIO_SRC}/image_processor.cpp
)
add_dependencies(system_manager
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(system_manager
    image_processor estimator
    ${catkin_LIBRARIES}
)

# System Manager nodelet
add_library(system_manager_nodelet
    src/System_nodelet.cpp
)
add_dependencies(system_manager_nodelet
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(system_manager_nodelet
    system_manager
    ${catkin_LIBRARIES}
)

#############
## Install ##
#############

install(TARGETS
  image_processor estimator system_manager system_manager_nodelet
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN "*_nodelet.h" EXCLUDE
)

install(FILES nodelets.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
