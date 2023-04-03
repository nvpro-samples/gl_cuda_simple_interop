/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute.cuh"
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265359f
#endif

inline __host__ __device__ void operator*=(float2 &a, float b) {
  a.x *= b;
  a.y *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}

inline __host__ __device__ void operator*=(float4 &a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}

inline __host__ __device__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}

__global__ void compute_kernel(const int frame_width, const int frame_height,
                               cudaSurfaceObject_t raw_tex_surf_obj,
                               float iTime) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= frame_width || y >= frame_height) {
    return;
  }

  float2 uv{float(x) / frame_width - 0.5f, float(y) / frame_height - 0.3f};
  uv *= 5.f;

  float d = abs(fmodf(dot(uv, uv) - iTime * 0.5f, 1.0f) - 0.5f) + 0.3f;
  float a =
      abs(fmodf(atan2f(uv.x, uv.y) / (M_PI * 1.75f) * 3.f, 1.0f)) - 0.5f + 0.2f;
  float4 col{fabsf(uv.x), fabsf(uv.y), 0.5f + 0.5f * sinf(iTime), 1.0f};

  if (a < d) {
    float4 rev_col{col.z, col.y, col.x, col.w};
    col = rev_col * d;
  } else {
    col *= a;
  }
  col.w = 1.0f;

  // Write to surface
  uchar4 col_u4{uint8_t(col.x * 255), uint8_t(col.y * 255),
                uint8_t(col.z * 255), 255};
  surf2Dwrite(col_u4, raw_tex_surf_obj, x * sizeof(uchar4), y);
}

void ComputeImageVk::compute(VkExtent2D &size) {

  static auto tStart = std::chrono::high_resolution_clock::now();
  auto tEnd = std::chrono::high_resolution_clock::now();
  auto tDiff =
      std::chrono::duration<float, std::milli>(tEnd - tStart).count() / 1000.f;

  // Wait for GL to finish
  cudaExternalSemaphoreWaitParams extSemaphoreWaitParams = {};
  CheckCUDA(cudaWaitExternalSemaphoresAsync(&m_semaphores.cudaReady,
                                            &extSemaphoreWaitParams, 1));
  dim3 block(32, 4);
  dim3 grid((size.width + 32 - 1) / 32, (size.height + 4 - 1) / 4);
  compute_kernel<<<grid, block>>>(size.width, size.height, m_texSurfObj, tDiff);
  cudaCheckError();

  // Signal to GL that the texture is ready
  cudaExternalSemaphoreSignalParams extSemaphoreSignalParams = {};
  CheckCUDA(cudaSignalExternalSemaphoresAsync(&m_semaphores.cudaComplete,
                                              &extSemaphoreSignalParams,
                                              1));
}
