# Older versions of CMake are likely to work just fine but, since
# I don't know where to cut off I just use the version I'm using
cmake_minimum_required(VERSION "3.17")


project(Reconstruction)


# Tell compiler to use C++ 17 features which is needed because
# Clang version is often behind in the XCode installation
set(CMAKE_CXX_STANDARD 17)
set(CMKAE_BUILD_TYPE “Release”)




# # set OpenCV_DIR variable equal to the path to the cmake
# # files within the previously installed opencv program
# set(OpenCV_DIR /Users/adammcquistan/code/c++/opencv/install/lib/cmake/opencv4)






# configure the necessary common CMake environment variables
# needed to include and link the OpenCV program into this
# demo project, namely OpenCV_INCLUDE_DIRS and OpenCV_LIBS




# OpenCV
find_package ( OpenCV REQUIRED )
# tell the build to include the headers from OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} )
message("DEP: OPENCV [OK] ")

# Eigen
# tell the build to include the headers from Eigen
include_directories("/opt/homebrew/Cellar/eigen/3.4.0_1/include/")
message("DEP: Eigen [OK] ")

# Ceres
# tell the build to include the headers from Ceres
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})
message("DEP: CERES [OK] ")

# QT5
find_package(qt5 REQUIRED)
include_directories(${QT5_INCLUDE_DIRS})
message("DEP: QT5 [OK] ")


# gflags / Boost / PCL
# find other packages
find_package(gflags REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(vtk REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})
find_package(PCL 1.12 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
message("DEP: gflags Boost PCL [OK] ")

# find_package ( Eigen3 3.4 REQUIRED NO_MODULE )
# include_directories(/usr/local/include ./include)

# specify the executable target to be built
add_executable(3DR main.cpp data_io.hpp feature_process.hpp feature_process.cpp
				my_package.hpp viewer.hpp viewer.cpp frame.hpp frame.cpp
				estimate_motion.hpp estimate_motion.cpp)

# tell it to link the executable target against OpenCV
target_link_libraries(3DR ${OpenCV_LIBRARIES})
target_link_libraries(3DR gflags)
target_link_libraries(3DR ${PCL_LIBRARIES})
target_link_libraries(3DR ${CERES_LIBRARIES})
target_link_libraries(3DR ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})
#target_compile_options(3DR PUBLIC -Wall -Wextra -pedantic)







