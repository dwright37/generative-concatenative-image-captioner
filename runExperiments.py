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
    baseline_model = {
        'useEmbedding'      : False,
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
    # Train Basline model
    _, baseline_model = makeAndTrainRnn(baseline_model, 'baseline')
    plot_loss_acc([baseline_model], 'baseline')
#############################################
    # Train using embedding
    model_metas = [baseline_model] # Create list to store results of training
    # Create deep copy of old model
    new_model = cp.deepcopy(model_metas[0])
    new_model['useEmbedding'] = True
    #Create a dummy COCOVocab just to get the dictionary
    vocabDummy = COCOVocab(dictionarySize=9715)
    new_model['embeddingModel'] = EmbeddingModel(vocabDummy.dictionary, 'semanticEmbedding.model')
    # Train model
    _, model_temp = makeAndTrainRnn(new_model, 'embedding')
    model_metas.append(model_temp)
    # Plot results
    plot_loss_acc(model_metas, 'embeddings')

#############################################
    # Train on different number of dropout
    model_metas = [baseline_model] # Create list to store results of training
    dropouts = [0.4, 0.3]
    for dropout in dropouts:
        # Create deep copy of old model
        new_model = cp.deepcopy(model_metas[0])
        new_model['dropout'] = dropout
        # Train model
        _, model_temp = makeAndTrainRnn(new_model, 'dropout_' + str(dropout))
        model_metas.append(model_temp)
    # Plot results
    plot_loss_acc(model_metas, 'dropout')

#############################################
    # Different number of hidden units
    model_metas = [baseline_model] # Create list to store results of training
    hiddens = [256, 750]
    for hidden in hiddens:
        # Create deep copy of old model
        new_model = cp.deepcopy(model_metas[0])
        new_model['hidden'] = hidden
        # Train model
        _, model_temp = makeAndTrainRnn(new_model, 'hidden_' + str(hidden))
        model_metas.append(model_temp)
    # Plot results
    plot_loss_acc(model_metas, 'hidden')

#############################################
    # Train with 2 hidden layers
    model_metas = [baseline_model] # Create list to store results of training
    # Create deep copy of old model
    new_model = cp.deepcopy(model_metas[0])
    new_model['hidden_layers'] = 2
    # Train model
    _, model_temp = makeAndTrainRnn(new_model, 'hidden_layers_' + str(2))
    model_metas.append(model_temp)
    # Plot results
    plot_loss_acc(model_metas, 'hidden_layers')

#############################################
    # remind img every time, every third time, every second time
    model_metas = [baseline_model] # Create list to store results of training
    reminders = ['Every', 'Every_Second', 'Every_Third']
    for remind in reminders:
        # Create deep copy of old model
        new_model = cp.deepcopy(model_metas[0])
        new_model['remind'] = remind
        # Train model
        _, model_temp = makeAndTrainRnn(new_model, 'remind_' + remind)
        model_metas.append(model_temp)
    # Plot results
    plot_loss_acc(model_metas, 'remind')







#############################################
    # Train over different number of words
    # model_metas = [baseline_model] # Create list to store results of training
    # num_words = [5000,500]
    # for num_word in num_words:
    #     # Create deep copy of old model
    #     new_model = cp.deepcopy(model_metas[0])
    #     new_model['vocabLength'] = num_word
    #     # Train model
    #     _, model_temp = makeAndTrainRnn(new_model, 'vocab_' + str(num_word))
    #     model_metas.append(model_temp)
    # # Plot results
    # plot_loss_acc(model_metas, 'words')




