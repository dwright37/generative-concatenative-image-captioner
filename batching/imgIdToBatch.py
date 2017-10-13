import numpy as np
from keras.utils import np_utils
import ipdb

def captionToBatch(dictionary, useEmbedding, embeddingModel, embeddingLength, caption, activationBase, remind):
    imageLength = 4096
    i = 0
    caption = '<start> ' + caption
    activation = np.repeat(activationBase[np.newaxis,np.newaxis,:], len(caption.split()),axis=1)
    captionWords = [w.lower() for w in caption.replace('.', '').split()]
    caption_length = len(captionWords)
    # How many time to remind network of image
    if remind == 'Begining_Only':
        rem_ind = 0
    elif remind == 'Every':
        rem_ind = np.arange(0, caption_length, 1)
    elif remind == 'Every_Second':
        rem_ind = np.arange(0, caption_length, 2)
    elif remind == 'Every_Third':
        rem_ind = np.arange(0, caption_length, 3)
    # Make input batch
    captionWordIds = [dictionary[w] if dictionary.has_key(w) else len(dictionary) for w in captionWords]
    if useEmbedding:
        captionAsInput = embeddingModel.predict(captionWords)
        captionAsInput = captionAsInput.reshape((1, captionAsInput.shape[0], captionAsInput.shape[1]))
        # Add image and caption to input
        #print "caption input shape:", captionAsInput.shape
        #print "activation input shape:", activation.shape
        #captionAsInput = np.append(captionAsInput, activation)
        captionAsInput = np.pad(captionAsInput, ((0,0),(0,0),(0,imageLength)), 'constant')
        captionAsInput[rem_ind, 0, embeddingLength:] = activation[0][0]
        #captionsAsInput.append(captionAsInput)
        captionAsInput = captionAsInput.reshape((captionAsInput.shape[1], captionAsInput.shape[2]))
        #captionsAsInput.append(captionAsInput.reshape((captionAsInput.shape[1], captionAsInput.shape[2])))
    else:
        captionAsInput = np_utils.to_categorical(captionWordIds, len(dictionary)+1)
        captionAsInput = np.pad(captionAsInput, ((0,0),(0,imageLength)), 'constant')
        captionAsInput[rem_ind, len(dictionary)+1:] = activation[0][0]
        #captionsAsInput.append(captionAsInput)
        
            

    # Make output batch (teacher forcing)       
    captionWords = captionWords[1:]
    captionWords.append('.')
    captionWordIds = [dictionary[w] if dictionary.has_key(w) else len(dictionary) for w in captionWords]
    captionAsOutput = np_utils.to_categorical(captionWordIds, len(dictionary)+1)
    #captionsAsOutput.append(captionAsOutput)

    return captionAsInput, captionAsOutput

def imgIdToBatch(dictionary, 
useEmbedding, 
embeddingModel, 
embeddingLength, 
captions, 
imgId, 
imgPath,
remind):

    # Load activaion
    activationBase = np.load(imgPath)
    # Standardize activations to be standard normal
    activationBase = activationBase - np.mean(activationBase)
    activationBase = activationBase / np.std(activationBase)

    # Process captions
    captions = [c for c in captions]
    captionsAsInput, captionsAsOutput = [], []
    for caption in captions:
        x, y = captionToBatch(dictionary, useEmbedding, embeddingModel,
                embeddingLength, caption, activationBase, remind)
        captionsAsInput.append(x)
        captionsAsOutput.append(y)
    return captionsAsInput, captionsAsOutput
