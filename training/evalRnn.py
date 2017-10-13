import numpy as np
import ipdb

def testCaptionedImages(cocoVocab, model, testSet, nbClasses, epoch):
    testBatchSize = min(1000, len(testSet))
    if testBatchSize > 0:
        print "Epoch", epoch, "in"
        acc = []
        loss = []
        batchX = []
        batchY = []
        metrics = [0.0, 0.0]
        for q in testSet:
            X, y = cocoVocab.captionActvToBatch([q], 'train', 5)
            n = np.random.randint(X.shape[0])
            x = X[n]
            xi = np.reshape(x, (1, x.shape[0], x.shape[1]))
            yi = np.reshape(y[n], (1, y[n].shape[0], y[n].shape[1]))
            metric = eval(model, xi, yi, batch_size=1, input_dim=nbClasses)
            acc.append(metric[1])
            loss.append(metric[0])
            #metrics = reduce(lambda acc, x: (acc[0] + x[0], acc[1] + x[1]), metrics, (0,0))
            #metrics = (metrics[0]/ float(len(X)), metrics[1]/ float(len(X)))
        metrics[0] = sum(loss) / len(loss)
        metrics[1] = sum(acc) / len(acc)
        print "Accuracy: " + str(metrics[1])
        print "Loss: " + str(metrics[0])
        return metrics

def test(cocoVocab, model, testSet, captionLength, nbClasses, epoch):
    testBatchSize = min(1000, len(testSet))
    #testInputShape = (testBatchSize, captionLength, nbClasses)
    if testBatchSize > 0:
        print "Epoch", epoch, "in"
        testIn, testOut = cocoVocab.captionToBatch(testSet[:testBatchSize])
        return eval(model, testIn, testOut, batch_size=testBatchSize,
                input_dim=nbClasses)

def eval(model, inputs, outputs, batch_size=10, input_dim=30010):
    #acc_total = 0.0
    #loss_total = 0.0
    #for i in range(inputs.shape[0]):
    #    metrics = model.evaluate(inputs[i].reshape((1, inputs[i].shape[0], inputs[i].shape[1])),
    #            outputs[i].reshape((1, outputs[i].shape[0], outputs[i].shape[1])),
    #            batch_size=1, verbose=2)
    #    acc_total += metrics[1]
    #    loss_total += metrics[0]
    #    model.reset_states()
    #accuracy = acc_total / inputs.shape[0]
    #loss = loss_total / inputs.shape[0]
    #ipdb.set_trace()
    metrics = model.evaluate(inputs,
             outputs, batch_size=batch_size, verbose=0)
    accuracy = metrics[1]
    loss = metrics[0]
    return (loss, accuracy)
