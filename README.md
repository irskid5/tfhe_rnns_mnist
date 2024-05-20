# TFHE RNNs over MNIST

Evaluation of the MNIST RNN, both regular and enlarged, over the encrypted MNIST test dataset using TFHE (or CGGI).

## Pre-installation Steps

1) sudo apt-get install libhdf5-serial-dev
2) Make sure you have cmake, build-essential, and NVIDIA Cuda Toolkit > 11.6 installed
3) Make sure you have GPUs installed in your system

Tip: Code to turn gpus on/off: sudo nvidia-smi drain -p 0000:67:00.0 -m 0

## Evaluation Instructions