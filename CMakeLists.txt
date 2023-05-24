cmake_minimum_required(VERSION 3.5)

get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)

# CMake 3.18 added CMAKE_CUDA_ARCHITECTURES. This must be set before enabling
# CUDA. Otherwise it will throw a warning. On versions < 3.18, setting this has
# no effect.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES is not defined, so nvcc's default will be used.")
  set(CMAKE_CUDA_ARCHITECTURES OFF) # Non-empty; passes no architecture flags, so chooses default.
endif()
enable_language(CUDA)

Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

#####################################################################################
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
#
if(NOT BASE_DIRECTORY)

  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()
    
_add_project_definitions(${PROJNAME})

#####################################################################################
# additions from packages needed for this sample
# add refs  in LIBRARIES_OPTIMIZED
# add refs  in LIBRARIES_DEBUG
# add files in PACKAGE_SOURCE_FILES

set( EXENAME ${PROJNAME} )
_add_package_OpenGL()
_add_package_VulkanSDK()
_add_package_IMGUI()
_add_package_Cuda()


#####################################################################################
# process the rest of some cmake code that needs to be done *after* the packages add
_add_nvpro_core_lib()

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c *.cu)

#####################################################################################
# Executable
#
#if(WIN32 AND NOT GLUT_FOUND)
#  add_definitions(/wd4996) #remove printf warning
#  add_definitions(/wd4244) #remove double to float conversion warning
#  add_definitions(/wd4305) #remove double to float truncation warning
#else()
#  add_definitions(-fpermissive)
#endif()
add_executable(${EXENAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES} ${GLSL_SOURCES})

#_set_subsystem_console(${EXENAME})

#####################################################################################
# common source code needed for this sample
#
source_group(common FILES 
  ${COMMON_SOURCE_FILES}
  ${PACKAGE_SOURCE_FILES}
)
source_group("Source Files" FILES ${SOURCE_FILES})

if(UNIX)
  set(UNIXLINKLIBS dl pthread)
else()
  set(UNIXLINKLIBS)
endif()

#####################################################################################
# Linkage
#
target_link_libraries(${EXENAME} ${PLATFORM_LIBRARIES} nvpro_core)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${EXENAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${EXENAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#

_finalize_target( ${EXENAME} )

else(CMAKE_CUDA_COMPILER)
	Message(WARNING "Project ${PROJNAME} NOT built: CUDA not found. Please provide CMAKE_CUDA_COMPILER to the cmake invocation.")
endif(CMAKE_CUDA_COMPILER)
