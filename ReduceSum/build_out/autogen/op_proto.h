#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ReduceSum)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(axes, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(keep_dims, Bool, false)
    .ATTR(ignore_nan, Bool, false)
    .ATTR(dtype, String, "float")
    .OP_END_FACTORY_REG(ReduceSum);

}

#endif
