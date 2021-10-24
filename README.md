# TintiNet.jl

![](./logo_TintiNet.png "TintiNet.jl Logo")

TintiNet.jl is the package that implements the Topological Inference by Neural Transformer-Inceptor Network architecture, as developed and described by G. Bottino and L. Martínez in the manuscript _**Estimation of protein topological properties from single
sequences using a fast language model**_ (in preparation)

This repository is a work in progress and lacks some relevant functionality, but contains a working example for training and running inference on a TintiNet if the user has access to a CUDA-capable GPU with 5 GB or more.

___

## Setting up

In order to follow this tutorial, you will first need to ensure you have CUDA (major version 11 - we used 11.4) and CUDNN (major version 8 - we used 8.2) installed in your system. CUDA and CUDNN are proprietary software and can be downloaded from https://developer.nvidia.com/

You will also need a Julia language installation. We recommend version 1.6. To obtain Julia, visit https://julialang.org/downloads/ and follow the instructions.

### Installing packages

If you are new to Julia, there is a package manager included that will help you install TintiNet.jl. in the REPL, you can type `]` to bring up the package command-line. You may want to manually install the major dependencies of TintiNet.jl before adding the repository itself, and in a specific order, because there is a certain hierarchy to them.

  - Begin by installing the `CUDA` package by typing `$julia> ]add CUDA`. Ensure that `CUDA` installation completes without problems. The CUDA package is the main way of implementing and utilizing GPU code in Julia.
  - Proceed to install the `Flux` package by typing `$julia> ]add Flux`. It should install without problems now that you have CUDA installed. The Flux package is one of the main AI frameworks for Julia.
  - Proceed to install the `Transformers` package by typing `$julia> ]add Transformers`. It should install without problems now that you have CUDA and Flux installed. The Transformers package is a collection of Transformer-based language models implemented in Julia.
  - Finally, install the `TintiNet` package by typing `$julia> ]add https://github.com/Hugemiler/TintiNet.jl.git`. It should install any remaining dependencies.

___

## Running the Example

In this repository, we provide an example script that allows you to build a specific version of the IGBT TintiNet. The script will allow you to tinker with this specific architecture and train it on over 30k protein sequences, saving model and optimizer checkpoints.

### Tutorial

1. Clone this repository
2. Unzip the file `datasets.tar.gz` in the same folder.
3. (optional) Edit the header structures `hyperpar` and `config` in the `examples/train_TintiNet_IGBT_classifier.jl` file to alter the network size if necessary.
4. Uncomment the last line of `examples/train_TintiNet_IGBT_classifier.jl` and run the file using `$julia> include("train_TintiNet_IGBT_classifier.jl")` from the REPL.
5. The generated model files will be stored in the examples folder. Take note of the training log and select a checkpoint.
6. **[TODO]** edit the file `examples/inference_TintiNet_IGBT_classifier.jl` to point to the chosen checkpoint
7. **[TODO]** run the file `examples/inference_TintiNet_IGBT_classifier.jl` to use your brand-new trained network to run inference on the default test set.
