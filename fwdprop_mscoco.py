import keras
from keras.applications import VGG16
from keras.models import Model
from ReadCOCOUtil import *
from os import path, mkdir
import shutil

def getModel(nb_remove_layers=1):
    '''
    Function creates vgg16 model without final classification layer

    Arguments
    =================
    nb_remove_layers - number of layers to remove from vgg16

    Returns
    =================
    model - vgg16 model with 'nb_remove_layers' layers removed.
    '''
    # Read in vgg16 model
    vgg_model = VGG16( weights='imagenet', include_top=True )
    out_layer = vgg_model.layers[-(nb_remove_layers + 1)]
    vgg_out = out_layer.output #Last FC layer's output
    #Freeze all layers of vgg model
    for layer in vgg_model.layers:
        layer.trainable = False

    #Add a flatten layer if not fc layer to get correct dimensionality
    if 'Dense' not in type(out_layer).__name__:
        vgg_out = Flatten()(vgg_out)

    #Create new model that only goes up to penultimate layer of vgg16
    model = Model( input=vgg_model.input, output=vgg_out )

    return model

def fwdprop_save(cocoutil, train_val, output_dir):
    if train_val is 'train':
        nbImgs = cocoutil.nbTrain
    elif train_val is 'val':
        nbImgs = cocoutil.nbVal
    # Iterate through training
    n = 0
    batchsz = 1000
    print(train_val + ' Output Dir = ' + output_dir)
    while n < nbImgs:
        print(train_val + ':\t' + str(n) + '/' + str(nbImgs))
        # Read in data
        if train_val is 'train':
            data = cocoutil.getTrainBatchImgs(batch_size = batchsz)
        elif train_val is 'val':
            data = cocoutil.getValidationBatchImgs(batch_size = batchsz)
        # Forward propogate
        output = model.predict(data[0])
        # Save output
        for t, img in enumerate(output):
            filename = path.join(output_dir, str(data[2][t]))
            np.save(filename, img)
        n = n + batchsz


# Define directory where propogated images will be saved
output_dir = './fwd_prop'
if not path.isdir(output_dir): mkdir(output_dir)

# Replace output directories if not already created
train_dir = path.join(output_dir, 'train')
val_dir = path.join(output_dir, 'validation')
if path.isdir(train_dir): shutil.rmtree(train_dir)
if path.isdir(val_dir): shutil.rmtree(val_dir)
mkdir(train_dir)
mkdir(val_dir)

# Generate model
model = getModel(nb_remove_layers = 1)

# Create ReadCOCOUtil object
cocoutil = ReadCOCOUtil()

# Forward propogate Training and Validation
fwdprop_save(cocoutil, 'train', train_dir)
fwdprop_save(cocoutil, 'val', val_dir)
