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

target_link_libraries(test_reducemax
    cust_opapi    # 自定义算子 API 库
    ascendcl      # ACL 核心库
    nnopbase
    acl_op_compiler
)

add_executable(test_perf test_perf.cpp)
target_link_libraries(test_perf
    cust_opapi ascendcl nnopbase acl_op_compiler)

add_executable(test_perf2 test_perf2.cpp)
target_link_libraries(test_perf2
    cust_opapi ascendcl nnopbase acl_op_compiler)

add_executable(test_perf3 test_perf3.cpp)
target_link_libraries(test_perf3
    cust_opapi ascendcl nnopbase acl_op_compiler)
