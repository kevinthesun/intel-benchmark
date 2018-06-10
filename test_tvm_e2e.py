import nnvm
import tvm
import mxnet as mx
import numpy as np
import time
import argparse
import json

from tvm.contrib import graph_runtime
from topi.nn.conv2d import _get_alter_layout_schedule, _get_schedule_NCHWc
from topi.x86.conv2d import AVXConvCommonFwd, AVXConv1x1Fwd

parser = argparse.ArgumentParser(description='Search convolution workload.')
parser.add_argument('--model', type=str, required=True,
                    help="Pretrained model from gluon model zoo.")

run_times = 100
global_idx = 0
_SCHEDULES_GLOBAL = []

def parse_sch(sch_str):
    if sch_str.startswith("AVXConvCommonFwd"):
        arg_list = sch_str[17:-1].split(',')
    elif sch_str.startswith("AVXConv1x1Fwd"):
        arg_list = sch_str[14:-1].split(',')
    else:
        raise RuntimeError("Schedule format not recognized for %s." % sch_str)
    arg_dict = {}
    for item in arg_list:
        item = item.strip()
        if '=' not in item:
            raise RuntimeError("Full keyword argument is required "
                               "to parse schedule from string.")
        arg_pair = item.split('=')
        if arg_pair[0] == "unroll_kw":
            arg_pair[1] = True if arg_pair[1] == "True" else False
        else:
            arg_pair[1] = int(arg_pair[1])
        arg_dict[arg_pair[0]] = arg_pair[1]
    if sch_str.startswith("AVXConvCommonFwd"):
        sch = AVXConvCommonFwd(**arg_dict)
    else:
        sch = AVXConv1x1Fwd(**arg_dict)
    return sch


def load_sch(sch_file):
    global global_idx, _SCHEDULES_GLOBAL
    global_idx = 0
    with open(sch_file, "r") as sf:
        _SCHEDULES_GLOBAL_str = json.load(sf)["schedules"]

    _SCHEDULES_GLOBAL = [parse_sch(sch) for sch in _SCHEDULES_GLOBAL_str]
    _SCH_DICT = {}

    @_get_alter_layout_schedule.register("cpu", override=True)
    def _get_alter_layout_schedule_skylake(wkl):
        global global_idx, _SCHEDULES_GLOBAL
        sch = _SCHEDULES_GLOBAL[global_idx]
        layout = "NCHW%dc" % sch.ic_bn
        out_layout = "NCHW%dc" % sch.oc_bn
        sch_key = "%s %s %s" % (wkl, layout, out_layout)
        _SCH_DICT[sch_key] = sch
        global_idx += 1
        return sch

    @_get_schedule_NCHWc.register("cpu", override=True)
    def _get_schedule_NCHWc_skylake(wkl, layout, out_layout):
        sch_key = "%s %s %s" % (wkl, layout, out_layout)
        sch = _SCH_DICT[sch_key]
        return sch


def end2end_benchmark(model, target, batch_size):
    print("Testing %s" % (model))
    num_classes = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)

    sym, arg_params, aux_params = mx.model.load_checkpoint("symbol/%s" % model, 0)
    net, params = nnvm.frontend.from_mxnet(sym, arg_params, aux_params)

    ctx = tvm.cpu()
    opt_level = 3
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(net, target, shape={"data": data_shape}, params=params)
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)

    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
    input_data = tvm.nd.array(data_array, ctx=ctx)
    module.set_input('data', input_data)

    # Warmup
    for _ in range(100):
        module.run()

    s = time.time()
    for _ in range(run_times):
        module.run()
    tvm_time = time.time() - s
    print("TVM average inference time for %s with %d runs: %f" % (model, run_times, tvm_time * 1000/run_times))


if __name__ == "__main__":
    args = parser.parse_args()
    model = args.model
    batch_size = 1
    target = "llvm -mcpu=skylake-avx512"
    sch_file = "schedule_file/%s_sch.json" % model 
    load_sch(sch_file)
    end2end_benchmark(model, target, batch_size)
