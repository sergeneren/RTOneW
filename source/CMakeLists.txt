cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)

project(RTOneW LANGUAGES CXX CUDA)
set(CUDA_VERBOSE_BUILD ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)


# Set output directories

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/RTOneW/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/RTOneW/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/RTOneW/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/RTOneW/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/RTOneW/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/RTOneW/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/RTOneW/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/RTOneW/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/RTOneW/bin)


#######################################################



find_package(Qt5Widgets CONFIG REQUIRED)
find_package(Qt5Core CONFIG REQUIRED)
find_package(Qt5Gui CONFIG REQUIRED)

file( GLOB	CORE_HEADERS
		"Core/*.h")
file( GLOB CUDA_SOURCES 
	"*.cu"
	"*.cuh")
file( GLOB QT_SOURCES 
	"*.ui"
	)

set(	RTWO_SRCS
		main.cpp
		renderHub.cpp
		mainWindow.cpp
		customGV.cpp	
		mainWindow.ui
		render_kernels.cu
		customGV.h
		mainWindow.h
		${CORE_HEADERS}
)

source_group(Core FILES ${CORE_HEADERS})
source_group(Cuda FILES ${CUDA_SOURCES})
source_group(QT5 FILES ${QT_SOURCES})

add_executable(RTOneW ${RTWO_SRCS} )

target_link_libraries(RTOneW PRIVATE Qt5::Widgets Qt5::Core Qt5::Gui)
target_include_directories(RTOneW PRIVATE "${CMAKE_SOURCE_DIR}/Core")
