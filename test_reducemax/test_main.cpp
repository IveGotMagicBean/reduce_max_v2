#include <iostream>
#include <fstream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_reduce_max_custom.h"

using namespace std;

template<typename T>
bool ReadFile(const string& filePath, vector<T>& data, size_t count) {
    ifstream file(filePath, ios::binary);
    if (!file) {
        cout << "[ERROR] Cannot open: " << filePath << endl;
        return false;
    }
    data.resize(count);
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    return true;
}

template<typename T>
bool WriteFile(const string& filePath, const T* data, size_t count) {
    ofstream file(filePath, ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    return true;
}

int main() {
    // 初始化ACL
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    cout << "[INFO] ACL initialized" << endl;
    
    // 读取输入数据
    const size_t dataSize = 32;
    vector<uint16_t> hostInput(dataSize);
    ReadFile("./input/input_x.bin", hostInput, dataSize);
    
    // 分配Device内存
    void *devInput, *devOutput, *devIdx;
    aclrtMalloc(&devInput, dataSize * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devOutput, sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devIdx, sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    
    // 拷贝输入到Device
    aclrtMemcpy(devInput, dataSize * sizeof(uint16_t), 
                hostInput.data(), dataSize * sizeof(uint16_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
    
    cout << "[INFO] Input copied to device" << endl;
    
    // 创建tensor
    int64_t inputShape[1] = {32};
    int64_t outputShape[1] = {1};
    
    aclTensor* inputTensor = aclCreateTensor(inputShape, 1, ACL_FLOAT16, 
                                             nullptr, 0, ACL_FORMAT_ND, 
                                             inputShape, 1, devInput);
    aclTensor* outputTensor = aclCreateTensor(outputShape, 1, ACL_FLOAT16,
                                              nullptr, 0, ACL_FORMAT_ND,
                                              outputShape, 1, devOutput);
    aclTensor* idxTensor = aclCreateTensor(outputShape, 1, ACL_INT32,
                                           nullptr, 0, ACL_FORMAT_ND,
                                           outputShape, 1, devIdx);
    
    // 调用算子
    int64_t reduceDim = 0;
    int64_t isKeepDim = 0;
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    auto ret = aclnnReduceMaxCustomGetWorkspaceSize(
        inputTensor, reduceDim, isKeepDim, outputTensor, idxTensor,
        &workspaceSize, &executor);
    
    if (ret != 0) {
        cout << "[ERROR] GetWorkspaceSize failed: " << ret << endl;
        return -1;
    }
    
    cout << "[INFO] Workspace: " << workspaceSize << " bytes" << endl;
    
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    
    ret = aclnnReduceMaxCustom(workspace, workspaceSize, executor, stream);
    
    if (ret != 0) {
        cout << "[ERROR] Execute failed: " << ret << endl;
        return -1;
    }
    
    aclrtSynchronizeStream(stream);
    cout << "[INFO] ✅ Operator executed successfully!" << endl;
    
    // 拷贝结果回Host
    uint16_t hostOutput;
    int32_t hostIdx;
    aclrtMemcpy(&hostOutput, sizeof(uint16_t), devOutput, sizeof(uint16_t),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&hostIdx, sizeof(int32_t), devIdx, sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);
    
    // 保存结果
    WriteFile("./output/output_y.bin", &hostOutput, 1);
    WriteFile("./output/output_idx.bin", &hostIdx, 1);
    
    cout << "[INFO] Output index: " << hostIdx << endl;
    
    // 清理
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    aclDestroyTensor(idxTensor);
    aclrtFree(devInput);
    aclrtFree(devOutput);
    aclrtFree(devIdx);
    if (workspace) aclrtFree(workspace);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    return 0;
}
