import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import json
import ipdb

def plot_loss_acc(model_metas, vary):
    """
    Function generates a plot for accuracy and loss as well as a .txt saving minimum and maximum loss 
    and accuracies. model_metas should be a list of dictionaries that contain the metadata and loss and 
    accuracy for a series of models. 

    Parameters
    ----------
    model_metas - <list> of dictionaries. Each element has a 'loss_acc' field wich contains the 
        validation loss and accuracy generated from trainOnCaptionedImages.py. Other fields give
        metadata of model (embedding, vocabLength ect.)
    vary        - <str> defining attribute that is varied across the models values = 'words', 
        'hidden', 'embeddings'

    """

    # Define folder that defines run
    if vary is not 'baseline':
        folder = os.path.join('./figures', 'vary_' + vary)
    else:
        folder = os.path.join('./figures', 'baseline')
    # Create folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Form textfilename
    txt_filename = os.path.join(folder, 'max_min_loss_acc_.txt')
    txt_file = open(txt_filename, 'w')
    # Clear current figures
    plt.figure(1).clf()
    plt.figure(2).clf()
    # Loop over models
    for model in model_metas:
        loss_acc = model['loss_acc']
        # Construct relevant label using vary paramater
        if vary == 'baseline':
            label = None
            json_nm = 'baseline'
        elif vary == 'dropout':
            label = str(model['dropout']) + 'dropout'
            json_nm = label
        elif vary == 'words':
            label = str(model['vocabLength']) + ' words'
            json_nm = label
        elif vary == 'hidden':
            label = str(model['hidden']) + ' hidden units'
            json_nm = label
        elif vary == 'hidden_layers':
            label = str(model['hidden_layers']) + ' hidden layers'
            json_nm = label
        elif vary == 'embeddings':
            if model['useEmbedding'] is True: 
                label = 'Embedded'
            elif model['useEmbedding'] is False: 
                label = 'No Embedding'
            json_nm = label
        elif vary == 'remind':
            label = model['remind']
            json_nm = label
        else:
            raise ValueError('Vary input to plot_loss_acc not recognized - can only accept "baseline", "words", "hidden" or "embeddings"')
        # Save loss_accuracy as .json
        with open(os.path.join(folder, json_nm + '.json'), 'w') as loss_file:    
            json.dump(loss_acc, loss_file)
        # Save minimum loss and max accuracy in .txt file
        txt_file.write('%s: \n\t Max Accuracy: %f \n\t Min Loss: %f\n\n' 
            % (label, max(loss_acc['acc']),  min(loss_acc['loss'])))

        # Plot loss
        fig = plt.figure(1)
        plt.plot(loss_acc['iter'], loss_acc['loss'], label = label)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc = 'upper right')
        
        # Plot accuracy
        fig = plt.figure(2)
        # Multiply by 100 to convert from decimal to percentage
        accuracy = np.multiply(loss_acc['acc'],100)
        plt.plot(loss_acc['iter'], accuracy, label = label)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc = 'lower right')
    
    txt_file.close()
    # Save loss figure
    fig = plt.figure(1)
    filename = os.path.join(folder, 'loss.pdf')
    fig.savefig(filename)

    # Save accuracy figure
    fig = plt.figure(2)
    filename = os.path.join(folder, 'accuracy.pdf')
    fig.savefig(filename)