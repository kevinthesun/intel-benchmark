# TVM VS MXNet-MKLDNN Benchmark for CNN on Intel Skylake CPU

Benchmark environment: AWS c5.9xlarge instance with 18 physical cores.

Usage:
First install dependecies:
```bash
./install.sh
```
Run MXNet-MKLDNN benchmark:
```bash
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=num_of_cpu_cores python test_mkl_e2e.py --model model_name
```
Run TVM benchmark:
```bash
TVM_NUM_THREADS=num_of_cpu_cores python test_tvm_e2e.py --model model_name
```
Change num_of_cpu_cores to be the number of cpu cores you want to use for benchmark. model_name can come from the following names: resnet18_v1, resnet34_v1, resnet50_v1, resnet101_v1, resnet152_v1, vgg11_bn, vgg19_bn.
