#
# Copyright 2023 Bernd Pfrommer <bernd.pfrommer@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.5)
project(simple_image_recon_py)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

# to use FindPython tools (pybind11_add_module)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11)

# simple_image_recon_lib must have been copied in beforehand
add_subdirectory(simple_image_recon_lib)
# find_package(simple_image_recon_lib REQUIRED)

pybind11_add_module(_simple_image_recon SHARED src/simple_image_recon.cpp)
target_link_libraries(_simple_image_recon PUBLIC simple_image_recon_lib pybind11::pybind11)
target_include_directories(_simple_image_recon
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

# -----------------------------------------

install(TARGETS
_simple_image_recon
DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
)


