# TVM VS MXNet-MKLDNN Benchmark for CNN on Intel Skylake CPU

Benchmark environment: AWS c5.9xlarge instance with 18 physical cores.

Usage:
First install tvm with llvm.

Download resnet50_v1 schedules: https://gist.github.com/kevinthesun/03ee47ebd35043a6737bdfb7f6a0dee0

Run TVM benchmark:
```bash
TVM_NUM_THREADS=num_of_cpu_cores python benchmark.py
```
Change num_of_cpu_cores to be the number of cpu cores you want to use for benchmark.
