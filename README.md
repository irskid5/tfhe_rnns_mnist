# tfhe_rnns_mnist

TO INSTALL, DO THE FOLLOWING PREINSTALL STEPS:

1) sudo apt-get install libhdf5-serial-dev
2) Make sure you have cmake, build-essential, and NVIDIA Cuda Toolkit > 11.6 installed

If you don't do this, it complains about HDF5 missing files.

--------------------------------------------------------------------------------

Code to turn gpus on/off: sudo nvidia-smi drain -p 0000:67:00.0 -m 0