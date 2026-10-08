cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Compiler: auto-detect bisheng (Ascend C compiler) before project()
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(_BISHENG bisheng
        HINTS "$ENV{ASCEND_HOME_PATH}/bin"
              "/usr/local/Ascend/ascend-toolkit/latest/bin"
    )
    if(_BISHENG)
        set(CMAKE_CXX_COMPILER "${_BISHENG}")
    else()
        message(FATAL_ERROR
            "bisheng compiler not found. "
            "Set CMAKE_CXX_COMPILER or ensure ASCEND_HOME_PATH is set.")
    endif()
endif()

project(sgl_glm53_ascend_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Ascend C: compile .cpp sources as Ascend C for dav-2201
add_compile_options(-x asc --npu-arch=dav-2201)

# ---------------------------------------------------------------------------
# Detect PyTorch / torch_npu paths via Python
# ---------------------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "
import torch, torch.utils.cpp_extension, torch_npu, sysconfig, json, os
tnpu = os.path.dirname(torch_npu.__file__)
print(json.dumps({
    'torch_include': torch.utils.cpp_extension.include_paths(),
    'torch_lib': os.path.join(os.path.dirname(torch.__file__), 'lib'),
    'npu_include': os.path.join(tnpu, 'include'),
    'npu_lib': os.path.join(tnpu, 'lib'),
    'python_include': sysconfig.get_config_var('INCLUDEPY'),
    'python_lib': sysconfig.get_config_var('LIBDIR'),
    'cxx11_abi': int(torch._C._GLIBCXX_USE_CXX11_ABI),
}))
"
    OUTPUT_VARIABLE _TORCH_JSON
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _TORCH_RESULT
)
if(NOT _TORCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to detect PyTorch / torch_npu configuration. "
        "Ensure torch and torch_npu are installed in the active Python environment.")
endif()

string(JSON _CXX11_ABI  GET "${_TORCH_JSON}" "cxx11_abi")
string(JSON _TORCH_LIB  GET "${_TORCH_JSON}" "torch_lib")
string(JSON _NPU_INC    GET "${_TORCH_JSON}" "npu_include")
string(JSON _NPU_LIB    GET "${_TORCH_JSON}" "npu_lib")
string(JSON _PY_INC     GET "${_TORCH_JSON}" "python_include")
string(JSON _PY_LIB     GET "${_TORCH_JSON}" "python_lib")

string(JSON _N_TORCH_INC LENGTH "${_TORCH_JSON}" "torch_include")
set(_TORCH_INCS "")
math(EXPR _LAST_INC "${_N_TORCH_INC} - 1")
foreach(_i RANGE 0 ${_LAST_INC})
    string(JSON _INC GET "${_TORCH_JSON}" "torch_include" ${_i})
    list(APPEND _TORCH_INCS "${_INC}")
endforeach()

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_CXX11_ABI})

# ---------------------------------------------------------------------------
# Interface library: torch + torch_npu includes/libs
# ---------------------------------------------------------------------------
add_library(torch_deps INTERFACE)
target_include_directories(torch_deps INTERFACE
    ${_TORCH_INCS}
    "${_PY_INC}"
    "${_NPU_INC}"
    "${CMAKE_SOURCE_DIR}/csrc/include"
)
target_link_directories(torch_deps INTERFACE
    "${_PY_LIB}"
    "${_TORCH_LIB}"
    "${_NPU_LIB}"
)
target_link_libraries(torch_deps INTERFACE torch_npu torch c10)

# ---------------------------------------------------------------------------
# Kernel registration helper
# ---------------------------------------------------------------------------
# add_ascend_kernel(<name> <source>)
#   Creates an OBJECT library target called <name>_kernel.
function(add_ascend_kernel NAME SOURCE)
    set(_TARGET "${NAME}_kernel")
    add_library(${_TARGET} OBJECT "${SOURCE}")
    target_link_libraries(${_TARGET} PRIVATE torch_deps)

    set_property(GLOBAL APPEND PROPERTY _SGL_KERNEL_TARGETS "${_TARGET}")
    set_property(GLOBAL APPEND PROPERTY _SGL_KERNEL_OBJECTS "$<TARGET_OBJECTS:${_TARGET}>")
endfunction()

# ---------------------------------------------------------------------------
# SGL_OPS: selective kernel inclusion (empty = all)
# ---------------------------------------------------------------------------
set(SGL_OPS "" CACHE STRING
    "Semicolon-separated list of kernels to build (empty = all)")

# ---------------------------------------------------------------------------
# Kernel definitions — one per independently developed operator
# ---------------------------------------------------------------------------
set(_ALL_KERNEL_NAMES helloworld prepared_recurrent reduce)

set(_SRC_helloworld          csrc/helloworld/helloworld.cpp)
set(_SRC_prepared_recurrent  csrc/kda/prepared_recurrent.cpp)
set(_SRC_reduce              csrc/reduce/reduce_sum.cpp)

foreach(_NAME IN LISTS _ALL_KERNEL_NAMES)
    if(SGL_OPS AND NOT "${_NAME}" IN_LIST SGL_OPS)
        message(STATUS "Kernel skipped: ${_NAME}")
        continue()
    endif()
    add_ascend_kernel("${_NAME}" "${_SRC_${_NAME}}")
    message(STATUS "Kernel enabled: ${_NAME} -> ${_SRC_${_NAME}}")
endforeach()

# ---------------------------------------------------------------------------
# Aggregate target: build all enabled kernels
# ---------------------------------------------------------------------------
get_property(_KERNEL_TARGETS GLOBAL PROPERTY _SGL_KERNEL_TARGETS)
add_custom_target(kernels DEPENDS ${_KERNEL_TARGETS})

# ---------------------------------------------------------------------------
# Final shared library: custom_ops_lib
# ---------------------------------------------------------------------------
get_property(_KERNEL_OBJECTS GLOBAL PROPERTY _SGL_KERNEL_OBJECTS)

if(NOT _KERNEL_OBJECTS)
    message(FATAL_ERROR "No kernels enabled. Check the SGL_OPS setting.")
endif()

add_library(sgl_glm53_ascend_ops_lib SHARED ${_KERNEL_OBJECTS})
target_link_libraries(sgl_glm53_ascend_ops_lib PRIVATE torch_deps)
add_dependencies(sgl_glm53_ascend_ops_lib ${_KERNEL_TARGETS})

set_target_properties(sgl_glm53_ascend_ops_lib PROPERTIES
    PREFIX ""
    OUTPUT_NAME "sgl_glm53_ascend_ops_lib"
)
