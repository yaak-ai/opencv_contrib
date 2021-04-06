/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CUDASURFACE_HPP
#define OPENCV_CUDASURFACE_HPP

#ifndef __cplusplus
#  error cudasurface.hpp header must be compiled as C++
#endif

#include "opencv2/core/cuda.hpp"

/**
  @addtogroup cuda
  @{
    @defgroup cudasurface
  @}
 */

namespace cv { namespace cuda {

//! @addtogroup cudasurface
//! @{

/*
Build cv::cuda::GpuMat with pre-allocated data (gpu pointer)
@sa ptr2mat
 */
CV_EXPORTS_W void ptr2mat(size_t ptr, InputOutputArray src, int height, int width, int step, int type, Stream& stream = Stream::Null());

/*
Cuda Device Pointer from cv::cuda::GpuMat
@sa mat2ptr
 */
// CV_EXPORTS_W CUdeviceptr mat2ptr(InputArray src);

}} // namespace cv { namespace cuda {

#endif /* OPENCV_CUDASURFACE_HPP */
