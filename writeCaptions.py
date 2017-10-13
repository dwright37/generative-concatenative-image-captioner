import matplotlib
matplotlib.use('Agg')
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import sys
import ipdb
from os import path

from getImageModel import getImageModel
from COCOVocab import COCOVocab
from semantic_embedding import *
from rnn import getModel

def preprocessImage(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x

# Look at the output of softmax
def sample(pDistribution, nb_classes=256, temperature=1.0):
    pDistribution = pDistribution.astype('float64')
    # avoid taking log of 0
    pDistribution = np.add(pDistribution, 10 ** -20)
    pDistribution = np.log(pDistribution) / temperature
    pDistribution = np.exp(pDistribution)
    pDistribution = pDistribution / np.sum(pDistribution)
    #sampledCharIndex = nb_classes-1
    #while sampledCharIndex == nb_classes-1:
    sampledChar = np.random.multinomial(1, pDistribution.reshape((nb_classes,)), 1)[0]
    sampledCharIndex = np.argmax(sampledChar)
    return sampledCharIndex, pDistribution

if __name__ == '__main__':
    T = 1.0
    if len(sys.argv) > 1:
        T = float(sys.argv[1])
        weightsFile = sys.argv[2]
        imgPath = sys.argv[3]

    useEmbedding = 'embedding' in weightsFile
    embeddingLength = 512
    #vocabLength = 2000 # Take the top N words
    vocabLength = 9715 # Take the top N words
    imageLength = 4096
    captionLength = 15
    vocabDummy = COCOVocab(dictionarySize=9715)
    embeddingModel = EmbeddingModel(vocabDummy.dictionary, 'semanticEmbedding.model')
    cocoVocab = COCOVocab(dictionarySize=vocabLength,
        useEmbedding=useEmbedding,
        embeddingLength=embeddingLength,
        imageLength=imageLength, embeddingModel=embeddingModel)

    weights = None

    nbClasses = cocoVocab.nbClasses

    batch_size = 14 - captionLength
    batch_size = 1 if batch_size < 1 else batch_size

    input_dim = embeddingLength if useEmbedding else nbClasses
    input_dim += imageLength

    #model = getModel(input_dim=input_dim,
    #        nb_classes=nbClasses, optimizer='Nadam')
    model = getModel(input_dim=input_dim, hidden_layers=1,
            nb_classes=nbClasses, optimizer='Nadam', hidden_units=512, nCaption=1)
    model.load_weights(weightsFile)
    # Load activaion
    imgId = 2445 # 2445 baseball image, 457029 skateboard
    #imageSet = 'train'
    #imgPath = path.join('/home/rob/mscoco_forwardprop', imageSet, str(imgId) + '.npy' )
    if '.jpg' in imgPath or '.jpeg' in imgPath:
        imgModel = getImageModel()
        img = image.load_img(imgPath, target_size=(224,224))
        img = preprocessImage(img)
        activationBase = imgModel.predict(img)
    else:
        activationBase = np.load(imgPath)
    activationBase = activationBase - np.mean(activationBase)
    activationBase = activationBase / np.std(activationBase)
    for i in range(2):
        print("")
        print("")
        input = ["<start>"]
        predictedWord = input[-1]
        while input[-1] not in '.':
            initial = np.zeros((1, 20, input_dim))
            sys.stdout.write(predictedWord + ' ')
            X, y = cocoVocab.captionToBatch([" ".join(input)] if len(input) < 20 else [" ".join(input[-20:])])
            X[0,0,input_dim-imageLength:] = activationBase
            initial[0,20-len(input):] = X
            X = initial
            #print(X)
            pDistribution = model.predict(X)[0]
            #print(pDistribution.shape)
            predictedWordId, _ = sample(pDistribution[-1], nbClasses, temperature=T)
            predictedWord =  cocoVocab.idToWords[predictedWordId]
            #print(predictedWord)
            input.append(predictedWord)
