# TintiNet.jl

![](./logo_TintiNet.png "TintiNet.jl Logo")

TintiNet.jl is the package that implements the Topological Inference by Neural Transformer-Inceptor Network architecture, as developed and described by G. Bottino and L. Martínez in the manuscript _**Estimation of protein topological properties from single sequences using a fast language model**_ (in preparation)

This repository is a work in progress and lacks some relevant functionality, but contains a working example for training and running inference on a TintiNet if the user has access to a CUDA-capable GPU with 5 GB or more.

___

## Setting up

In order to follow this tutorial, you will first need to ensure you have CUDA (major version 11 - we used 11.4) and CUDNN (major version 8 - we used 8.2) installed in your system. CUDA and CUDNN are proprietary software and can be downloaded from https://developer.nvidia.com/

You will also need a Julia language installation. We recommend version 1.6. To obtain Julia, visit https://julialang.org/downloads/ and follow the instructions.

### Installing packages

If you are new to Julia, there is a package manager included that will help you install TintiNet.jl. in the REPL, you can type `]` to bring up the package command-line and simply install the `TintiNet` package by typing `$julia> ]add https://github.com/Hugemiler/TintiNet.jl.git`. It should install any dependencies, provided you have already set up CUDA and CUDNN adequately.

___

## Running the Example

In this repository, we provide an example script that allows you to build a specific version of the IGBT TintiNet. The script will allow you to tinker with this specific architecture and train it on over 30k protein sequences, saving model and optimizer checkpoints.

### Tutorial

1. Setting up:
- Clone this repository
- Unzip the file `datasets.tar.gz` in the same folder.
- (optional) Edit the header structures `hyperpar` and `config` in the `examples/train_TintiNet_IGBT_classifier.jl` file to alter the network size if necessary.

2. Predicting 1D properties on protein samples using the pretrained models
- Ensure Julia is in our `PATH`
- Navigate to the exmples directory and run the prediction script with `$> julia predict_TintiNet_IGBT.jl inputfile outputdir`. Argument `inputfile` receives a FASTA document with a list of proteins (refer to `example_samples.fasta` for structure), and argument `outputdir`receives the path to a director where the prediction outputs will be written. We provide an example inputfile comprised of the first 200 entries of CATH-S40. To use this inputfile, run command `$> julia predict_TintiNet_IGBT.jl example_samples.fasta example_outputs`

3. Training the Classifier network and using your model for inference
- Uncomment the last line of `examples/train_TintiNet_IGBT_classifier.jl` and run the file using `$julia> include("train_TintiNet_IGBT_classifier.jl")` from the REPL.
- The generated model files will be stored in the examples folder. Take note of the training log and select a checkpoint.
- To run prediction using your own trained model, edit line `@load "./example_classifier_model.bson" checkpoint_model` in `predict_TintiNet_IGBT.jl` to point to your checkpoint.

4. Training the Regressor network and using your model for inference
- Uncomment the last line of `examples/train_TintiNet_IGBT_regressor.jl` and run the file using `$julia> include("train_TintiNet_IGBT_regressor.jl")` from the REPL.
- The generated model files will be stored in the examples folder. Take note of the training log and select a checkpoint.
- To run prediction using your own trained model, edit line `@load "./example_regressor_model.bson" checkpoint_model` in `predict_TintiNet_IGBT.jl` to point to your checkpoint.
