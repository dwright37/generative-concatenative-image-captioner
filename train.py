import matplotlib
matplotlib.use('Agg')

import copy as cp
import ipdb

from rnn import getModel
from COCOVocab import COCOVocab
from training.trainOnCaptionedImages import trainOnCaptionedImages as train
from semantic_embedding import EmbeddingModel
from plotting.plot_loss_acc import plot_loss_acc

def makeAndTrainRnn(model_meta, modelLabel):

    cocoVocab = COCOVocab(
        dictionarySize  = model_meta['vocabLength'],
        useEmbedding    = model_meta['useEmbedding'],
        embeddingLength = model_meta['embeddingLength'],
        imageLength     = model_meta['imageLength'], 
        embeddingModel  = model_meta['embeddingModel'])

    weights = None

    nbClasses = cocoVocab.nbClasses

    batchSize = 14 - model_meta['captionLength']
    batchSize = 1 if batchSize < 1 else batchSize

    input_dim = model_meta['embeddingLength'] if model_meta['useEmbedding'] else nbClasses
    input_dim += model_meta['imageLength']

    model = getModel(
        input_dim   =   input_dim,
        nb_classes  =   nbClasses,
        optimizer   =   'Adagrad',
        dropout     =   model_meta['dropout'],
        hidden_units      =   model_meta['hidden'],
        hidden_layers     =   model_meta['hidden_layers']
    )
    weights, loss_acc = train(
        cocoVocab,
        model,
        batchSize,
        nbClasses,
        weights = weights,
        modelLabel = modelLabel,
        remind = model_meta['remind'])

    # Add loss_acc to model model_meta
    model_meta['loss_acc'] = loss_acc
    return weights, model_meta


if __name__ == '__main__':

    # Create baseline model
    model = {
        'useEmbedding'      : True,
        'embeddingLength'   : 512,
        'vocabLength'       : 9715, # Take the top N words
        'imageLength'       : 4096,
        'captionLength'     : 15,
        'hidden'            : 512,
        'hidden_layers'     : 1,
        'embeddingModel'    : None,
        'dropout'           : 0.5,
        'remind'            : 'Begining_Only'
    }
    #Create a dummy COCOVocab just to get the dictionary
    vocabDummy = COCOVocab(dictionarySize=9715)
    model['embeddingModel'] = EmbeddingModel(vocabDummy.dictionary, 'semanticEmbedding.model')
    # Train model
    _, model = makeAndTrainRnn(model, 'baseline')

