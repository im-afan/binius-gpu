# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 4.0.3)

# Reject any attempt to use a toolchain file. We must not use one because
# we could be downloading it here. If the CMAKE_TOOLCHAIN_FILE environment
# variable is set, the cache variable will have been initialized from it.
unset(CMAKE_TOOLCHAIN_FILE CACHE)
unset(ENV{CMAKE_TOOLCHAIN_FILE})

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(nlohmann_json-populate NONE)



include(ExternalProject)
ExternalProject_Add(nlohmann_json-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "FALSE" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "URL" "https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip" "URL_HASH" "SHA256=a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d" "PATCH_COMMAND" "/home/intern-04/cmake-4.0.3-linux-x86_64/bin/cmake" "-D" "CUDA_VERSION=12.6.68" "-D" "CXX_VERSION=12.3.0" "-D" "CXX_ID=GNU" "-P" "/home/intern-04/binius-gpu/third-party/nvbench/cmake/patches/json_unordered_map_ice.cmake"
                    SOURCE_DIR          "/home/intern-04/binius-gpu/build/_deps/nlohmann_json-src"
                    BINARY_DIR          "/home/intern-04/binius-gpu/build/_deps/nlohmann_json-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


