# Generative Concatenative Image Captioner

## Abstract
We present an image captioning model using a Generative Concatenative Network (GCN). We accomplished this by taking an existing convolutional network (e.g. VGG16) and feeding it into a word level recurrent neural network (RNN). The network is trained and tested on the Microsoft Common Objects in COntext (MSCOCO) dataset. Image captioning is performed using a generative concatenative method where image representations are presented to the RNN at specific time steps by concatenating the representation with our word encoding. 

## Architecture
Our implementation is modeled after Vinyals\' et al. network presented in [(Vinyals et al. 2015)](https://arxiv.org/pdf/1411.4555.pdf), as well as the GCN presented in 
[(Lipton et al., 2015)](https://arxiv.org/pdf/1511.03683.pdf), which consists of a ConvNet encoding image activations and concatenating these with the input to an RNN. 
In this model, a ConvNet is used to map images to a real vector, and the RNN is used to generate English language sentences conditioned on this image representation. 
For this project we did not implement attention, so captions are generated using the image gestalt. A high level view of the architecture is given below. 

![architecture](assets/architecture.png | width=800)

The activations of the image are presented to the LSTM by concatenating them with the word representation and are parameterized by a set of weights. At each time step the next 
word presented to the network is the predicted word from the previous time step. Caption generation starts with a special START token and terminates when a special END token is output by the network.

## Example Output
The following are example captions generated for the given images with different softmax temperatures and different models tested.

![tennis](assets/tennis.png | width=800)

![cat](assets/cat.png | width=800)

## Setup
- Make sure you are using python3 with all of the usual packages (numpy, matplotlib, scipy, etc)
- Install tensorflow-gpu v >= 1.0.0
- Install Keras
- Install the MSCOCO utilities (https://github.com/cocodataset/cocoapi)

## Running experiments
-Download the MSCOCO training/validation images and captions to a directory and create a symlink to this directory `${REPO_ROOT}/coco`

-You need to run `python fwdprop_mscoco.py`. This will generate the image representations to the directory `${REPO_ROOT}/fwd_prop`

You can train the main model as follows:

```bash
$ python train.py
```

This creates a file "model.h5" that can be used to generate captions for new images.

You can run all of our experiments as follows:

```bash
$ python runExperiments.py
```

## Captioning images
You can caption an image using the following:

```bash
$ python writeCaptions.py model.h5 <temperature> <path_to_image>
```

Where \<temperature\> is the softmax temperature


# Web server
There is a (very light) web server which can be run to pass in and caption an image. To run it, install node.js,
navigate to the web-server directory, execute the following:

```bash
$ npm install busboy
$ node server.js
```

You can then navigate to http://localhost:8080 and use the interface to generate captions for a given image.
