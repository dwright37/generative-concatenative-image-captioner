import collections
import time
import numpy as np
from evalRnn import test, testCaptionedImages
from ReadCOCOUtil import ReadCOCOUtil
import gc
import ipdb

COCO = ReadCOCOUtil()
trainingImageIds = COCO.imgIdsTrain
validationImageIds = COCO.imgIdsVal

def validateModel(validationX, validationY, model, epoch, loss_acc):
    metric = model.evaluate(validationX, validationY, batch_size=10, verbose=1)
    loss_acc['acc'].append(metric[1])
    loss_acc['loss'].append(metric[0])
    loss_acc['iter'].append(epoch)
    return loss_acc


def trainOnCaptionedImages(cocoVocab, model, batchSize, nbClasses,
        weights=None, modelLabel='unlabelled', remind='Begining_Only'):
    validationX, validationY = cocoVocab.captionActvToBatch(validationImageIds[:500], 'validation', 20, remind)
    if weights is not None:
        model.set_weights(weights)
        #Save out our last best weights
        model.save_weights(modelLabel + '.h5')
    trainLen = .25*len(trainingImageIds)
    miniBatchSize = 500
    loss_acc = {'loss': [], 'acc':[], 'iter':[]}
    for i in range(10):
        print "Epoch " + str(i)
        loss_acc = validateModel(validationX, validationY, model, i, loss_acc)
        epoch = 0
        while (miniBatchSize*epoch + miniBatchSize) <= trainLen:
            print 'Percent complete: ' + str((float(miniBatchSize)*epoch)/trainLen * 100.0)
            X, Y = cocoVocab.captionActvToBatch(trainingImageIds[(miniBatchSize*epoch):(miniBatchSize*epoch + miniBatchSize)], 'train', 20, remind)
            model.fit(x=X, y=Y, batch_size=(10+50*i), nb_epoch=1, verbose=1)
            model.save_weights(modelLabel + '.h5')
            epoch += 1
            X = []
            Y = []
            gc.collect()
    weights_out = model.get_weights()
    print loss_acc
    return weights_out, loss_acc
