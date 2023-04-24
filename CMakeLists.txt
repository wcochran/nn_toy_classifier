cmake_minimum_required(VERSION 3.0)
project( nn_toy_classifier )
set( CMAKE_CXX_STANDARD 17 )
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#include_directories(/opt/local/include/eigen3)
add_executable( nn_toy_classifier nn_toy_classifier.cpp )
target_link_libraries( nn_toy_classifier Eigen3::Eigen )

