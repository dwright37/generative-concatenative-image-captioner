# Commands to gnerate captions
For the cat image, use:
`CUDA_VISIBLE_DEVICES="" python writeCaptions.py 1.0 baseline.h5 /home/rob/mscoco_forwardprop/validation/599.npy`

For the tennis image, use:
`CUDA_VISIBLE_DEVICES="" python writeCaptions.py 1.0 baseline.h5 /home/rob/mscoco_forwardprop/validation/564.npy`

Obviously you can replace the image with whatever activations you want and can change out the weights to different models in the argument. Might have to do some editing by hand for changing the model in the file. 
If the weights file has 'embedding' in the name, it will use the embedding model
