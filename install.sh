#!/bin/bash
sudo apt-get install llvm-5.0
sudo pip install mxnet-mkl
git clone https://github.com/kevinthesun/tvm.git --recursive -b AlterLayoutImprove
cd tvm
cp make/config.mk .
echo LLVM_CONFIG=llvm-config-5.0 >> config.mk
make -j
cd python && sudo python setup.py install && cd ..
cd topi/python && sudo python setup.py install && cd ../..
cd nnvm && make -j
cd python && sudo python setup.py install && cd ../../..
wget https://github.com/kevinthesun/intel-benchmark/releases/download/0.1/mxnet_model.zip
unzip mxnet_model.zip
wget https://github.com/kevinthesun/intel-benchmark/releases/download/0.1/tvm_schedule.zip
unzip tvm_schedule.zip
