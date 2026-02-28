cmake_minimum_required(VERSION 3.19)
project(test_reducemax)
set(CMAKE_CXX_STANDARD 11)
set(ASCEND_PATH /home/developer/Ascend/cann-8.5.0)
include_directories(
    ${ASCEND_PATH}/aarch64-linux/include
    ${ASCEND_PATH}/opp/vendors/customize/op_api/include
)
link_directories(
    ${ASCEND_PATH}/aarch64-linux/lib64
    ${ASCEND_PATH}/opp/vendors/customize/op_api/lib
)
add_executable(test_reducemax test_main.cpp)
target_link_libraries(test_reducemax cust_opapi ascendcl nnopbase acl_op_compiler)
