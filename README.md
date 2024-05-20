# TFHE RNNs over MNIST

Evaluation of the MNIST RNN, both regular and enlarged, over the encrypted MNIST test dataset using TFHE (or CGGI).

## Pre-installation Steps

1) sudo apt-get install libhdf5-serial-dev
2) Make sure you have cmake, build-essential, and NVIDIA Cuda Toolkit > 11.6 installed
3) Make sure you have >=1 GPU installed in your system
4) Run "cargo build --release".

## Evaluation Instructions

The dataset and model files are already included. The dataset file should contain ternarized values and the model parameters file, binarized values.

There are different evaluation configurations that can be defined in "src/main.rs". Specifically, in the following code snippet:

```
print_rnn_banner!(mnist_rnn, <arg0>, <arg1>, <arg2>);
```

the arguments after "mnist_rnn", are the arguments, or configuration options, that can be passed in to the function "mnist_rnn" which runs the main evaluation of the MNIST RNN. There are three arguments and thus, configuration options, provided to the user, that are highlighted in the next section. The configuration options need to be changed in the code and built by the compiler in order to run.

To run the code, use the following bash command in the root directory:

`cargo run --release`

Or compile the project into a binary using the following command in the root directory:

`cargo build --release`

and run the binary `target/release/tfhe_rnns_mnist` from the root directory:

### Configuration Options

#### run_pt (arg0)

A boolean that indicates whether or not to run the evaluation in plaintext concurrently with the evaluation over encrypted data. If true, the plainext evaluation is run concurrently and error metrics are calculated between encrypted and plaintext activations (percent difference) and pre-activations (mean absolute error), per layer.

#### model_type (arg1)

A boolean. If true, runs the regular model. If false, runs the enlarged model.

#### config (arg2)

A reference to one of two possible static CGGI parameter sets, SET1 or SET2, which are equal to the CGGI parameter sets 1 and 2 described in the paper, respectively.