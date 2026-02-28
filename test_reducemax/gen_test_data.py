import numpy as np
import os

def gen_data(shape, reduce_dim):
    np.random.seed(42)
    x = (np.random.randn(*shape) * 3).astype(np.float16)
    golden_val = np.max(x, axis=reduce_dim).astype(np.float16)
    golden_idx = np.argmax(x, axis=reduce_dim).astype(np.int32)

    tag = "x".join(map(str, shape)) + f"_dim{reduce_dim}"
    os.makedirs(f"test_reducemax/input_{tag}", exist_ok=True)
    os.makedirs(f"test_reducemax/output_{tag}", exist_ok=True)

    x.tofile(f"test_reducemax/input_{tag}/input_x.bin")
    golden_val.tofile(f"test_reducemax/output_{tag}/golden_y.bin")
    golden_idx.tofile(f"test_reducemax/output_{tag}/golden_idx.bin")

    print(f"[{tag}] input={x.shape} output={golden_val.shape} max={golden_val.max():.5f}")

gen_data((1048576,),        reduce_dim=0)
gen_data((512, 128, 512),   reduce_dim=1)
gen_data((512, 128, 512),   reduce_dim=2)
gen_data((1024, 512, 1024), reduce_dim=1)
gen_data((512, 4096),       reduce_dim=1)
gen_data((512, 4096),       reduce_dim=0)
print("Done!")
