/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <chrono>
#include <iostream>

#include "gl_vkpp.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/shaders_vk.hpp"

#ifdef NDEBUG

#define CheckCUDA(call) call;
#define cudaCheckError() ;

#else

#define CheckCUDA(call)                                                                                                \
  {                                                                                                                    \
    const cudaError_t err = call;                                                                                      \
    if(err != cudaSuccess)                                                                                             \
    {                                                                                                                  \
      std::cerr << "CudaDebugCall() failed at: " << __FILE__ << ":" << __LINE__ << "; ";                               \
      std::cerr << "code: " << err << "; description: " << cudaGetErrorString(err) << std::endl;                       \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  }

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                                                                               \
  {                                                                                                                    \
    cudaError_t e = cudaGetLastError();                                                                                \
    if(e != cudaSuccess)                                                                                               \
    {                                                                                                                  \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));                                 \
      exit(0);                                                                                                         \
    }                                                                                                                  \
  }

#endif

inline VkExternalSemaphoreHandleTypeFlagBits getDefaultSemaphoreHandleType()
{
#ifdef _WIN64
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

inline VkExternalMemoryHandleTypeFlagBits getDefaultMemHandleType()
{
#ifdef _WIN64
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

#ifdef _WIN64
using HandleType_T = HANDLE;
#else
using HandleType_T = int;
#endif

inline HandleType_T getMemHandle(const VkDevice& device, const VkDeviceMemory& memory)
{
#ifdef _WIN64
  HandleType_T                  handle = 0;
  VkMemoryGetWin32HandleInfoKHR memInfo{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
  memInfo.memory     = memory;
  memInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  VkResult result    = vkGetMemoryWin32HandleKHR(device, &memInfo, &handle);
#else
  HandleType_T         handle = -1;
  VkMemoryGetFdInfoKHR memInfo{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
  memInfo.memory     = memory;
  memInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
  VkResult result    = vkGetMemoryFdKHR(device, &memInfo, &handle);
#endif
  if(result != VK_SUCCESS)
  {
    assert(result == VK_SUCCESS);
  }
  return handle;
}


class ComputeImageVk
{

public:
  ComputeImageVk() = default;
  void setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice)
  {
    m_device         = device;
    m_physicalDevice = physicalDevice;

    m_alloc.init(device, physicalDevice);
  }

  VkDevice              m_device;
  nvvkpp::Texture2DVkGL m_textureTarget;
  VkPhysicalDevice      m_physicalDevice;

  nvvk::ExportResourceAllocatorDedicated m_alloc;
  cudaArray_t                            m_cudaTextureArray;
  cudaExternalMemory_t                   m_cudaImageMemory;
  cudaMipmappedArray_t                   m_cudaImageMipmappedArray;
  cudaSurfaceObject_t                    m_texSurfObj;
  struct Semaphores
  {
    VkSemaphore             vkCudaToGl;
    VkSemaphore             vkGlToCuda;
    GLuint                  glReady;
    GLuint                  glComplete;
    cudaExternalSemaphore_t cudaReady;
    cudaExternalSemaphore_t cudaComplete;
  } m_semaphores;


  void destroy()
  {
    CheckCUDA(cudaDestroySurfaceObject(m_texSurfObj));
    m_textureTarget.destroy(m_alloc);
    vkDestroySemaphore(m_device, m_semaphores.vkCudaToGl, nullptr);
    vkDestroySemaphore(m_device, m_semaphores.vkGlToCuda, nullptr);
  }

  void prepare(int width, int height)
  {
    createSemaphores();
    m_textureTarget =
        prepareTextureTarget(VK_IMAGE_LAYOUT_GENERAL, {uint32_t(width), uint32_t(height), 1}, VK_FORMAT_R8G8B8A8_UNORM);
  }

  void createSemaphores()
  {
    glGenSemaphoresEXT(1, &m_semaphores.glReady);
    glGenSemaphoresEXT(1, &m_semaphores.glComplete);

    // Create semaphores
    const auto handleType = getDefaultSemaphoreHandleType();

    VkExportSemaphoreCreateInfo esci{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    esci.handleTypes = handleType;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &esci;
    vkCreateSemaphore(m_device, &sci, nullptr, &m_semaphores.vkCudaToGl);
    vkCreateSemaphore(m_device, &sci, nullptr, &m_semaphores.vkGlToCuda);

    // Import vk semaphores to GL and CUDA
#ifdef WIN32
    {
      HANDLE                           hglReady{NULL};
      HANDLE                           hglComplete{NULL};
      VkSemaphoreGetWin32HandleInfoKHR handleInfo{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
      handleInfo.handleType = handleType;
      handleInfo.semaphore  = m_semaphores.vkCudaToGl;
      vkGetSemaphoreWin32HandleKHR(m_device, &handleInfo, &hglReady);

      handleInfo.semaphore = m_semaphores.vkGlToCuda;
      vkGetSemaphoreWin32HandleKHR(m_device, &handleInfo, &hglComplete);

      glImportSemaphoreWin32HandleEXT(m_semaphores.glReady, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, hglReady);
      glImportSemaphoreWin32HandleEXT(m_semaphores.glComplete, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, hglComplete);
      cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};
      externalSemaphoreHandleDesc.type                            = cudaExternalSemaphoreHandleTypeOpaqueWin32;
      externalSemaphoreHandleDesc.flags                           = 0;
      externalSemaphoreHandleDesc.handle.win32.handle             = hglReady;
      CheckCUDA(cudaImportExternalSemaphore(&m_semaphores.cudaComplete, &externalSemaphoreHandleDesc));
      externalSemaphoreHandleDesc.handle.win32.handle = hglComplete;
      CheckCUDA(cudaImportExternalSemaphore(&m_semaphores.cudaReady, &externalSemaphoreHandleDesc));
    }
#else
    // it seems to be a bug that after using a handle with gl or cuda import "consumes" the handle and it had to be queried again from vk
    {
      HandleType_T                          fdReady = 0;
      HandleType_T                          fdComplete = 0;
      VkSemaphoreGetFdInfoKHR handleInfo{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
      handleInfo.handleType = handleType;
      handleInfo.semaphore  = m_semaphores.vkCudaToGl;
      vkGetSemaphoreFdKHR(m_device, &handleInfo, &fdReady);

      handleInfo.semaphore = m_semaphores.vkGlToCuda;
      vkGetSemaphoreFdKHR(m_device, &handleInfo, &fdComplete);
      cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {.type = cudaExternalSemaphoreHandleTypeOpaqueFd, .flags = 0};
      externalSemaphoreHandleDesc.handle.fd = fdReady;
      CheckCUDA(cudaImportExternalSemaphore(&m_semaphores.cudaComplete, &externalSemaphoreHandleDesc));
      externalSemaphoreHandleDesc.handle.fd = fdComplete;
      CheckCUDA(cudaImportExternalSemaphore(&m_semaphores.cudaReady, &externalSemaphoreHandleDesc));
    }
    {
      HandleType_T                          fdReady = 0;
      HandleType_T                          fdComplete = 0;
      VkSemaphoreGetFdInfoKHR handleInfo{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
      handleInfo.handleType = handleType;
      handleInfo.semaphore  = m_semaphores.vkCudaToGl;
      vkGetSemaphoreFdKHR(m_device, &handleInfo, &fdReady);

      handleInfo.semaphore = m_semaphores.vkGlToCuda;
      vkGetSemaphoreFdKHR(m_device, &handleInfo, &fdComplete);
      glImportSemaphoreFdEXT(m_semaphores.glReady, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fdReady);
      glImportSemaphoreFdEXT(m_semaphores.glComplete, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fdComplete);
    }
#endif
  }

  nvvkpp::Texture2DVkGL prepareTextureTarget(VkImageLayout targetLayout, const VkExtent3D& extent, VkFormat format)
  {
    VkExtent2D        imgSize         = VkExtent2D{extent.width, extent.height};
    VkImageUsageFlags usage           = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    VkImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, usage);

    VkExternalMemoryImageCreateInfo extMemInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    extMemInfo.handleTypes = getDefaultMemHandleType();
    imageCreateInfo.pNext  = &extMemInfo;

    nvvkpp::Texture2DVkGL texture;

    nvvk::Image           image  = m_alloc.createImage(imageCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);

    // Create the texture from the image and adding a default sampler
    VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    texture.texVk   = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);
    texture.imgSize = imgSize;

    auto                         mem_info             = m_alloc.getMemoryAllocator()->getMemoryInfo(image.memHandle);
    auto                         mem_handle           = getMemHandle(m_alloc.getDevice(), mem_info.memory);
    cudaExternalMemoryHandleDesc cudaExtMemHandleDesc = {};
#ifdef _WIN64
    cudaExtMemHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
    cudaExtMemHandleDesc.handle.win32.handle = mem_handle;
#else
    cudaExtMemHandleDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    cudaExtMemHandleDesc.handle.fd = mem_handle;
#endif
    cudaExtMemHandleDesc.size = mem_info.size;
    CheckCUDA(cudaImportExternalMemory(&m_cudaImageMemory, &cudaExtMemHandleDesc));

    cudaExternalMemoryMipmappedArrayDesc mipmappedArrayDesc = {};
    mipmappedArrayDesc.extent                               = make_cudaExtent(texture.imgSize.width, texture.imgSize.height, 0);  // depth must be zero for 2D textures and not one! Seems to result in different image alignments.
    mipmappedArrayDesc.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    mipmappedArrayDesc.numLevels  = 1;
    mipmappedArrayDesc.offset     = 0;
    mipmappedArrayDesc.flags      = cudaArraySurfaceLoadStore;


    CheckCUDA(cudaExternalMemoryGetMappedMipmappedArray(&m_cudaImageMipmappedArray, m_cudaImageMemory, &mipmappedArrayDesc));

    CheckCUDA(cudaGetMipmappedArrayLevel(&m_cudaTextureArray, m_cudaImageMipmappedArray, 0));
    cudaResourceDesc resDescr = {};
    resDescr.resType = cudaResourceTypeArray;
    resDescr.res.array.array = m_cudaTextureArray;

    // Create surface object
    CheckCUDA(cudaCreateSurfaceObject(&m_texSurfObj, &resDescr));
    return texture;
  }

  void compute(VkExtent2D& size);
};
