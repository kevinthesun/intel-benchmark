import nnvm
import tvm
import mxnet as mx
import numpy as np
import time

from tvm.contrib import graph_runtime
from mxnet import gluon
from mxnet.gluon.model_zoo.vision import get_model

run_times = 1000

def end2end_benchmark(model, target, batch_size):
    print("Testing %s" % (model))
    num_classes = 1000
    image_shape = (3, 299, 299) if "inception" in model else (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)

    block = get_model(model, pretrained=True)
    net, params = nnvm.frontend.from_mxnet(block)
    
    tvm.autotvm.task.DispatchContext.current = tvm.autotvm.apply_graph_best("resnet50_v1_opt.log")
    ctx = tvm.cpu()
    opt_level = 3
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(net, target=target, shape={"data": data_shape}, params=params)

    module = graph_runtime.create(graph, lib, ctx)

    module.set_input(**params)

    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
    input_data = tvm.nd.array(data_array, ctx=ctx)
    mx_data = mx.nd.array(data_array)
    module.set_input('data', input_data)
    
    # Warmup
    for _ in range(100):
        module.run()

    s = time.time()
    for _ in range(run_times):
        module.run()
    tvm_time = time.time() - s
    print("TVM %s inference time for batch size of %d: %f" % (model, batch_size, tvm_time * 1000/run_times))
    
    tvm_out = module.get_output(0, out=tvm.nd.empty(out_shape))
    mx_out = block(mx_data)
    np.testing.assert_array_almost_equal(tvm_out.asnumpy(), mx_out.asnumpy(), decimal=3)


if __name__ == "__main__":
    model = "resnet50_v1"
    batch_size = 1
    target = "llvm -mcpu=skylake-avx512"
    tm= end2end_benchmark(model, target, batch_size)
