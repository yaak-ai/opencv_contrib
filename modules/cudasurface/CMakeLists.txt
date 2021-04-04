if(IOS OR WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudasurface)
endif()

set(the_description "Building cv::cuda::GpuMat from pre-allocated cuda device pointer")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow)

ocv_add_module(cudasurface opencv_core OPTIONAL opencv_cudev WRAP python)

ocv_module_include_directories()
ocv_glob_module_sources()

set(extra_libs "")

if(HAVE_CUBLAS)
  list(APPEND extra_libs ${CUDA_cublas_LIBRARY})
endif()

if(HAVE_CUFFT)
  list(APPEND extra_libs ${CUDA_cufft_LIBRARY})
endif()

ocv_create_module(${extra_libs})
