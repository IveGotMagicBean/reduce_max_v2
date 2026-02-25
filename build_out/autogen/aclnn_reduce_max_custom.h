
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_REDUCE_MAX_CUSTOM_H_
#define ACLNN_REDUCE_MAX_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnReduceMaxCustomGetWorkspaceSize
 * parameters :
 * x : required
 * reduceDim : required
 * isKeepDim : optional
 * yOut : required
 * idxOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnReduceMaxCustomGetWorkspaceSize(
    const aclTensor *x,
    int64_t reduceDim,
    int64_t isKeepDim,
    const aclTensor *yOut,
    const aclTensor *idxOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnReduceMaxCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnReduceMaxCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
