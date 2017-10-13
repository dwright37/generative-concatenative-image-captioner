import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Embedding
from ReadCOCOUtil import ReadCOCOUtil
from collections import defaultdict
from keras.preprocessing import sequence
from os import path
import ipdb

from batching.imgIdToBatch import imgIdToBatch


# length controls how large our vocab is. it will take the <length> most
# frequent words + 2 for the '.' and start words
# length of 9715: gives all words used at least 5 times
def createDictionary(captions, length):
    words = {}

    for c in captions:
        for w in c.split():
            w = w.decode().lower().replace('.','')

            if w in words:
                words[w] += 1
            else:
                words[0] = 1

    counts = [(words[w], w) for w in words]
    counts.sort()
    counts.reverse()

    # View the number of times a word occurs in the caption dataset in
    # descending order
    #for c in counts:
        #print c

    # print len(counts)
    # 44535 raw and 37089 when you lowercase all the words
    # 30008 when you lowercase and remove periods from the words.

    wordsByFrequency = [c[1] for c in counts[:length]]
    wordIds = defaultdict(lambda: len(wordsByFrequency) + 2)
    idToWords = defaultdict(lambda: len(wordsByFrequency) + 2)
    for (i, w) in enumerate(wordsByFrequency):
        wordIds[w] = i
        idToWords[i] = w

    wordIds['.'] = len(wordsByFrequency)
    wordIds['<start>'] =  wordIds['.'] + 1
    idToWords[len(wordsByFrequency)] = '.'
    idToWords[ wordIds['.'] + 1 ] = '<start>'
    idToWords[ wordIds['.'] + 2 ] = '___'
    return wordIds, idToWords


class COCOVocab:
    # default dictionary size is entire vocabulary
    def __init__(self, dictionarySize=31008,
            useEmbedding=True, embeddingLength=256,
            imageLength=4096, embeddingModel=None):
        self.COCO = ReadCOCOUtil()
        annIds = self.COCO.cocoTrain.getAnnIds()
        self.captions = [x['caption'].encode('ascii') for x in self.COCO.cocoTrain.loadAnns(annIds)]
        self.dictionary, self.idToWords = createDictionary(self.captions, dictionarySize)
        self.nbClasses = dictionarySize + 3 # Add 3 for <start>, ., and unknown
        #For using word embeddings
        self.useEmbedding = useEmbedding
        if embeddingModel is None:
            self.embeddingModel = Sequential()
            self.embeddingModel.add(Embedding(dictionarySize, embeddingLength))
            self.embeddingModel.compile('rmsprop', 'mse')
        else:
            self.embeddingModel = embeddingModel

        self.embeddingLength = embeddingLength

        self.imageLength = imageLength

    def captionToBatch(self, captions):
        captionsAsInput = []
        captionsAsOutput = []
        for caption in captions:
            captionWords = [w.lower().replace('.', '') for w in caption.split()]
            # Make input batch
            captionWordIds = [self.dictionary[w] if self.dictionary.has_key(w) else len(self.dictionary) for w in captionWords]
            if self.useEmbedding:
                captionAsInput = self.embeddingModel.predict(np.asarray(np.asarray(captionWords)))
                #Zero pad for the image
                captionAsInput = np.pad(captionAsInput, ((0,0),(0,self.imageLength)), 'constant')
                captionsAsInput.append(captionAsInput.reshape((captionAsInput.shape[0], captionAsInput.shape[1])))
            else:
                captionAsInput = np_utils.to_categorical(captionWordIds, len(self.dictionary)+1)
                captionAsInput = np.pad(captionAsInput, ((0,0),(0,self.imageLength)), 'constant')
                captionsAsInput.append(captionAsInput)
            #captionsAsInput.append(np.asarray(captionWordIds))

            # Make output batch (teacher forcing)
            captionWords = captionWords[1:]
            captionWords.append('.')
            captionWordIds = [self.dictionary[w] if self.dictionary.has_key(w) else len(self.dictionary) for w in captionWords]
            captionAsOutput = np_utils.to_categorical(captionWordIds, len(self.dictionary)+1)
            captionsAsOutput.append(captionAsOutput)

        return np.asarray(captionsAsInput), np.asarray(captionsAsOutput)

    """
    Parameters
    ======================
    - imgIds <list>:    ids of images obtained from ReadCOCOUtil.imgsIdsTrain
        and ReadCOCOUtil.imgsIdsVal
    - imageSet <str>: 'train' or 'validation' defining whether imgIds are trainingset or
        validation set
    - captionLength<int>: Length of caption

    Output
    ======================
    - captionAsOutput <tuple>: same as captionToBatch()

    """
    def captionActvToBatch(self, imgIds, imageSet, captionLength, remind):
        """
            remind = 'Only_Begining', 'Every', 'Every_Second' or 'Every_Third'
        """
        captionsAsInput = []
        captionsAsOutput = []
        for imgId in imgIds:
            if imageSet == 'train' :
                annotationIds = self.COCO.cocoTrain.getAnnIds(imgId)
                captions =[x['caption'].encode('ascii') for x in self.COCO.cocoTrain.loadAnns(annotationIds)]
            else: 
                annotationIds = self.COCO.cocoVal.getAnnIds(imgId)
                captions =[x['caption'].encode('ascii') for x in self.COCO.cocoVal.loadAnns(annotationIds)]

            
            imgPath = path.join(self.COCO.activsDir, imageSet, str(imgId) + '.npy' )

            x, y = imgIdToBatch(self.dictionary, self.useEmbedding,
                    self.embeddingModel, self.embeddingLength,
                    captions, imgId, imgPath, remind)
            captionsAsInput += x
            captionsAsOutput += y

        captionsAsInput = sequence.pad_sequences(captionsAsInput, maxlen=captionLength, dtype='float32', padding='pre', truncating='post')
        captionsAsOutput = sequence.pad_sequences(captionsAsOutput, maxlen=captionLength, dtype='float32', padding='pre', truncating='post')
        return captionsAsInput, captionsAsOutput

