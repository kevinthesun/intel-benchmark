import mxnet as mx
import numpy as np
import time
import argparse

from collections import namedtuple

parser = argparse.ArgumentParser(description='Search convolution workload.')
parser.add_argument('--model', type=str, required=True,
                    help="Pretrained model from gluon model zoo.")

Batch = namedtuple('Batch', ['data'])
run_times = 100

def end2end_benchmark(model, target, batch_size):
    print("Testing %s" % (model))
    num_classes = 1000
    image_shape = (3, 224, 224) if "inception" not in model else (3, 299, 299)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)
    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
    mx_data = mx.nd.array(data_array)

    sym, arg_params, aux_params = mx.model.load_checkpoint("symbol/" + model, 0)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', data_shape)],
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    # Warmup
    for _ in range(100):
        mod.forward(Batch([mx_data]))
        for output in mod.get_outputs():
            output.wait_to_read()

    s = time.time()
    for _ in range(run_times):
        mod.forward(Batch([mx_data]))
        for output in mod.get_outputs():
            output.wait_to_read()
    mkl_time = time.time() - s
    print("MXNet MKLDNN Average inference time for %s with %d runs: %f ms" % (model, run_times, mkl_time/run_times * 1000))

if __name__ == "__main__":
    args = parser.parse_args()
    model = args.model
    batch_size = 1
    target = "llvm -mcpu=skylake-avx512"
    end2end_benchmark(model, target, batch_size)

