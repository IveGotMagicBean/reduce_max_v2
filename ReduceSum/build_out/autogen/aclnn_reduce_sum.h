
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_REDUCE_SUM_H_
#define ACLNN_REDUCE_SUM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnReduceSumGetWorkspaceSize
 * parameters :
 * x : required
 * axes : required
 * keepDims : optional
 * ignoreNan : optional
 * dtypeOptional : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnReduceSumGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *axes,
    bool keepDims,
    bool ignoreNan,
    char *dtypeOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnReduceSum
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnReduceSum(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
