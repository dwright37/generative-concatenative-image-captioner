import keras
from keras.applications import VGG16
from keras.models import Model


def getImageModel(nb_remove_layers=1):
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

