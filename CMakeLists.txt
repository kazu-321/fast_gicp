cmake_minimum_required(VERSION 3.10.2)
project(fast_gicp)

option(BUILD_VGICP_CUDA "Build GPU-powered VGICP" ON)
option(BUILD_apps "Build application programs" ON)
option(BUILD_test "Build test programs" OFF)
option(BUILD_PYTHON_BINDINGS "Build python bindings" OFF)

set(CMAKE_BUILD_TYPE "Release")

find_package(ament_cmake REQUIRED)
find_package(PCL REQUIRED COMPONENTS io registration)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(autoware_localization_util REQUIRED)
find_package(autoware_internal_debug_msgs REQUIRED)

if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# if(BUILD_VGICP_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  link_directories(${CUDA_LIBRARY_DIRS})
# endif()

add_library(fast_gicp SHARED
  src/fast_gicp/gicp/lsq_registration.cpp
  src/fast_gicp/gicp/fast_gicp.cpp
  src/fast_gicp/gicp/fast_gicp_st.cpp
  src/fast_gicp/gicp/fast_vgicp.cpp
  src/fast_gicp/gicp/fast_vgicp_cuda.cpp
)
target_link_libraries(fast_gicp
  ${PCL_LIBRARIES}
)
ament_target_dependencies(fast_gicp tf2)
ament_target_dependencies(fast_gicp tf2_ros)
ament_target_dependencies(fast_gicp sensor_msgs)
ament_target_dependencies(fast_gicp pcl_conversions)
ament_target_dependencies(fast_gicp autoware_localization_util)
ament_target_dependencies(fast_gicp autoware_internal_debug_msgs)
if (OPENMP_FOUND)
    if (TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(fast_gicp OpenMP::OpenMP_CXX)
    endif ()
endif ()
target_include_directories(fast_gicp PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${geometry_msgs_INCLUDE_DIRS}
)

add_executable(gicp_align src/align.cpp)
target_link_libraries(gicp_align
  ${PCL_LIBRARIES}
  fast_gicp
  fast_vgicp_cuda
)
ament_target_dependencies(gicp_align tf2 tf2_ros sensor_msgs pcl_conversions autoware_localization_util autoware_internal_debug_msgs)

if(BUILD_PYTHON_BINDINGS)
  find_package(pybind11 CONFIG)
  pybind11_add_module(pygicp
    src/python/main.cpp
  )
  target_include_directories(pygicp PUBLIC
    include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )
  target_link_libraries(pygicp PRIVATE
    fast_gicp
  )
endif()

# if(BUILD_VGICP_CUDA)
  set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
  add_definitions(-DUSE_VGICP_CUDA)

  cuda_add_library(fast_vgicp_cuda SHARED
    src/fast_gicp/cuda/fast_vgicp_cuda.cu
    src/fast_gicp/cuda/brute_force_knn.cu
    src/fast_gicp/cuda/covariance_estimation.cu
    src/fast_gicp/cuda/covariance_estimation_rbf.cu
    src/fast_gicp/cuda/covariance_regularization.cu
    src/fast_gicp/cuda/gaussian_voxelmap.cu
    src/fast_gicp/cuda/find_voxel_correspondences.cu
    src/fast_gicp/cuda/compute_derivatives.cu
    src/fast_gicp/cuda/compute_mahalanobis.cu
    src/fast_gicp/cuda/ndt_cuda.cu
    src/fast_gicp/cuda/ndt_compute_derivatives.cu
    src/fast_gicp/gicp/fast_vgicp_cuda.cpp
  )
  target_include_directories(fast_vgicp_cuda PRIVATE
    include
    thirdparty/Eigen
    thirdparty/nvbio
  )
  target_link_libraries(fast_vgicp_cuda
    fast_gicp
  )
# endif()

if(BUILD_test)
  find_package(GTest REQUIRED)

  add_executable(gicp_test src/test/gicp_test.cpp)
  target_link_libraries(gicp_test ${GTEST_LIBRARIES} ${PCL_LIBRARIES} fast_gicp)
endif()

install(TARGETS fast_gicp
  LIBRARY DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp"
)

ament_export_include_directories(include)
ament_export_libraries(fast_gicp)
ament_package()

install(TARGETS gicp_align
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
