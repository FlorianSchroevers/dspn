# Deep Set Prediction Networks

![Overview of set optimisation on the example of CLEVR bounding box prediction](overview.png)

## DSPN

This is the unofficial adaptation of the implementation of the NeurIPS 2019 paper [Deep Set Prediction Networks][0].
They propose a new way of predicting sets with a neural network that doesn't suffer from discontinuity issues.
This is done by backpropagating through a set encoder to act as a set decoder.
You can take a look at the [poster for NeurIPS 2019][4] or the [poster for the NeurIPS 2019 workshop on Sets & Partitions][5]. This adaptation adds a new loss function, support for two new datasets, and the option to merge datasets.

To use the decoder, you only need [`dspn.py`][1].
You can see how it is used in [`model.py`][2] with `build_net` and the `Net` class.
For details on the exact steps to reproduce the experiments, check out the README in the `dspn` directory.
You can download pre-trained models and the predictions thereof from the [Resources][3] page.

## Setup

Some assembly is required to make sure everythin works as intented (presumes working with Anaconda). 

 * Install Cuda 10.2
 * Create and activate a conda environment
 * Install torch as described [Here][6]
 * For the next step, we need to make sure we have version <8.* of gcc and g++. [Here][7] is a guide to downgrade if necesarry.
 * Clone into the fancy branch of neuralnet-pytorch: `git clone -b fancy https://github.com/justanhduc/neuralnet-pytorch.git`
 * Change directory to `neuralnet-pytorch/neuralnet_pytorch/extensions/cuda/emd_c` and edit the file `emd_kernel.cu`
 * Edit the lines at the beginning:
 ```
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
 ```
 And change to:
 ```
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
 ```
 * Browse back to the neuralnet-pytorch base dir and run `python setup.py install`
 * Install other nesecarry packages: `pip install Pillow tensorboardX h5py matplotlib numpy pandas scipy tqdm`


# BibTeX entry

```
@inproceedings{zhang2019dspn,
    author        = {Yan Zhang and Jonathon Hare and Adam Pr\"ugel-Bennett},
    title         = {{Deep Set Prediction Networks}},
    booktitle     = {Advances in Neural Information Processing Systems 32},
    year          = {2019},
    eprint        = {1906.06565},
    url           = {https://arxiv.org/abs/1906.06565},
}
```


[0]: https://arxiv.org/abs/1906.06565
[1]: https://github.com/Cyanogenoid/dspn/blob/master/dspn.py
[2]: https://github.com/Cyanogenoid/dspn/blob/master/dspn/model.py
[3]: https://github.com/Cyanogenoid/dspn/releases/tag/resources
[4]: https://www.cyanogenoid.com/files/dspn-poster.pdf
[5]: https://www.cyanogenoid.com/files/dspn-workshop-poster.pdf
[6]: https://pytorch.org/get-started/locally/
[7]: https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa
