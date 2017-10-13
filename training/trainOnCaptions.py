import collections
import time
from evalRnn import test
from continueLearning import continueLearning

def trainOnCaptions(cocoVocab, model, batchSize, nbClasses,
        captionLength=5, weights=None):

    if weights is not None:
        model.set_weights(weights)
        #Save out our last best weights
        model.save_weights('weights.h5')

    captions = [c for c in cocoVocab.captions if len(c.split()) == captionLength]
    trainSize = int(round((8.0*len(captions))/10.0))

    if trainSize is 0:
        return model.get_weights

    train = [(c) for c in captions[:trainSize]]
    testSet =  [(c) for c in captions[trainSize:]]

    nbEpochs = trainSize
    testEvery = 100 #min(int(1001 / batchSize), trainSize)
    start = time.time()
    loc = 0
    losses = collections.deque([1000.0, 1000.1,1000.3], 4)
    loss_best = 1000.0
    weights_out = model.get_weights()
    #while continueLearning(losses) and i < nbEpochs:
    i = 0
    while i < nbEpochs:
        if i % testEvery is 0:
            (loss, acc) = test(cocoVocab, model, testSet,
                    captionLength, nbClasses, i)
            losses.appendleft(loss)
            if loss < loss_best:
                loss_best = loss
                #Save the best weights
                weights_out = model.get_weights()
        if loc*batchSize+batchSize >= len(train) - 1:
            batch = train[loc*batchSize:]
            loc = 0
        else:
            batch = train[loc*batchSize:loc*batchSize+batchSize]
            loc += 1

        X, y = cocoVocab.captionToBatch(batch)
        model.fit(X, y, verbose=0)
        i += 1

    model.save_weights('weights.h5')
    return weights_out
