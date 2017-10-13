# Setup
- Make sure you are using python3 with all of the usual packages (numpy, matplotlib, scipy, etc)
- Install tensorflow-gpu v >= 1.0.0
- Install Keras
- Install the MSCOCO utilities (https://github.com/cocodataset/cocoapi)

# Running experiments
-Download the MSCOCO training/validation images and captions to a directory and create a symlink to this directory `${REPO_ROOT}/coco`

-You need to run `python fwdprop_mscoco.py`. This will generate the image representations to the directory `${REPO_ROOT}/fwd_prop`

You can train the main model as follows:

```python
$ python train.py
```

This creates a file "model.h5" that can be used to generate captions for new images.

You can run all of our experiments as follows:

```python
$ python runExperiments.py
```

#Captioning images
You can caption an image using the following:

```python
$ python writeCaptions.py model.h5 <temperature> <path_to_image>
```

Where \<temperature\> is the softmax temperature


# Web server
There is a (very light) web server which can be run to pass in and caption an image. To run it, install node.js 
and execute the following:

```bash
$ node web-server/server.js
```

You can then navigate to http://localhost:8080 and use the interface to generate captions for a given image.