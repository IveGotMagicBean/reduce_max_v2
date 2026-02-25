#include <iostream>
#include <fstream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_reduce_max_custom.h"

using namespace std;

template<typename T>
bool ReadFile(const string& path, vector<T>& data, size_t count) {
    ifstream f(path, ios::binary);
    if (!f) { cout << "[ERROR] Cannot open: " << path << endl; return false; }
    data.resize(count);
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    return true;
}

template<typename T>
void WriteFile(const string& path, const T* data, size_t count) {
    ofstream f(path, ios::binary);
    f.write(reinterpret_cast<const char*>(data), count * sizeof(T));
}

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    cout << "[INFO] ACL initialized" << endl;

    // 读取输入（32 个 float16）
    const size_t N = 1024*1024;
    vector<uint16_t> hostInput(N);
    ReadFile("./input/input_x.bin", hostInput, N);

    // 申请设备内存
    void *devX, *devY, *devIdx;
    aclrtMalloc(&devX,   N * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devY,   16 * sizeof(uint16_t),     ACL_MEM_MALLOC_HUGE_FIRST);  // 输出是标量
    aclrtMalloc(&devIdx, 16 * sizeof(int32_t),      ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(devX, N * sizeof(uint16_t), hostInput.data(), N * sizeof(uint16_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
    cout << "[INFO] Input copied to device" << endl;

    // 构建 tensor（输入 [32]，输出 scalar [1]）
    int64_t inShape[1] = {1024*1024};
    int64_t outShape[1] = {1};
    int64_t reduceDim   = 0;
    int64_t isKeepDim   = 0;  // 输出为标量，不保留维度

    aclTensor* xTensor   = aclCreateTensor(inShape,  1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, inShape,  1, devX);
    aclTensor* yTensor   = aclCreateTensor(outShape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, outShape, 1, devY);
    aclTensor* idxTensor = aclCreateTensor(outShape, 1, ACL_INT32,   nullptr, 0, ACL_FORMAT_ND, outShape, 1, devIdx);

    // 获取 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnReduceMaxCustomGetWorkspaceSize(
        xTensor, reduceDim, isKeepDim, yTensor, idxTensor,
        &workspaceSize, &executor);

    if (ret != 0) { cout << "[ERROR] GetWorkspaceSize failed: " << ret << endl; return -1; }
    cout << "[INFO] Workspace size: " << workspaceSize << endl;

    void* workspace = nullptr;
    if (workspaceSize > 0)
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 执行算子
    ret = aclnnReduceMaxCustom(workspace, workspaceSize, executor, stream);
    if (ret != 0) { cout << "[ERROR] Execute failed: " << ret << endl; return -1; }

    aclrtSynchronizeStream(stream);
    cout << "[INFO] ✅ Operator executed!" << endl;

    // 拷贝结果回 host
    uint16_t hostY   = 0;
    int32_t  hostIdx = 0;
    aclrtMemcpy(&hostY,   sizeof(uint16_t), devY,   sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&hostIdx, sizeof(int32_t),  devIdx, sizeof(int32_t),  ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("./output/output_y.bin",   &hostY,   1);
    WriteFile("./output/output_idx.bin", &hostIdx, 1);

    cout << "[INFO] Output max value (raw uint16): " << hostY   << endl;
    cout << "[INFO] Output max index             : " << hostIdx << endl;

    // 释放资源
    aclDestroyTensor(xTensor);
    aclDestroyTensor(yTensor);
    aclDestroyTensor(idxTensor);
    aclrtFree(devX); aclrtFree(devY); aclrtFree(devIdx);
    if (workspace) aclrtFree(workspace);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
