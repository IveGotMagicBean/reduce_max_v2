## Directory Structure Introduction
``` 
├── AclNNInvocation             // Invokes the ReduceSum operator using the aclnn method
│   ├── inc                     // Header file directory
│   │   ├── common.h            // Declares common method classes for reading binary files
│   │   ├── op_runner.h         // Operator description declaration file, including operator inputs/outputs, operator type, and input/output descriptions
│   │   └── operator_desc.h     // Operator runtime information declaration file, including the number of operator inputs/outputs, input/output sizes, etc.
│   ├── input                   // Directory for storing script-generated input data
│   ├── output                  // Directory for storing operator runtime output data and ground truth data
│   ├── scripts
│   │   ├── acl.json            // acl configuration file
│   │   ├── gen_data.py         // Script for generating input data and ground truth data
│   │   └── verify_result.py    // Ground truth comparison file
│   ├── src
│   │   ├── CMakeLists.txt      // Compilation rules file
│   │   ├── common.cpp          // Common functions, implementation file for reading binary files
│   │   ├── main.cpp            // Entry point for the single operator invocation application
│   │   ├── op_runner.cpp       // Main flow implementation file for single operator invocation
│   │   └── operator_desc.cpp   // Constructs input and output descriptions for the operator
│   └── run.sh                  // Execution command script
``` 

## Code Implementation Introduction
After developing and deploying the custom operator, you can verify the functionality of the single operator by invoking it. The code in src/main.cpp is the execution method for the single operator API. Single operator API execution is based on the C language API to execute the operator, without the need to provide a single operator description file for offline model conversion, directly calling the single operator API interface.

After the custom operator is compiled and deployed, the single operator API is automatically generated and can be directly called in the application. The operator API is generally defined as a "two-stage interface", such as:
   ```cpp    
   aclnnStatus aclnnReduceSumGetWorkspaceSize(const aclTensor *x, const aclTensor *axes, bool keepDims, bool ignoreNan, 
                                                        char *dtypeOptional, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
   aclnnStatus aclnnReduceSum(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream);
   ```
Where aclnnReduceSumGetWorkspaceSize is the first stage interface, mainly used to calculate how much workspace memory is needed during this API call. After obtaining the required workspace size for this API calculation, allocate Device-side memory according to the workspaceSize, and then call the second stage interface aclnnTriu to perform the calculation.
## Running the Example Operator
### 1. Compiling the Operator Project
Before running this example, please refer to [Compiling the Operator Project](../README.en.md#operatorcompile) to complete the preliminary preparation.

### 2. Running the aclnn Example

  - Enter the example directory

    ```bash
    cd ${git_clone_path}/samples/operator_contrib/ReduceSumSample/FrameworkLaunch/AclNNInvocation
    ```
  - Example execution    

    During the example execution, test data will be automatically generated, then the aclnn example will be compiled and run, and finally, the results will be verified. The specific process can be found in the run.sh script.

    ```bash
    bash run.sh
    ```

## Update Log
  | Date | Update Items |
|----|------|
| 2025/03/07 | Readme created |