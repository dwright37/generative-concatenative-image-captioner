from gensim.models import Word2Vec
from ReadCOCOUtil import ReadCOCOUtil
from COCOVocab import COCOVocab
import numpy as np
import ipdb
import os

class TrainSentences(object):
    def __init__(self, dictionary):
        self.coco = ReadCOCOUtil()
        self.annIds = self.coco.cocoTrain.getAnnIds()
        self.dictionary = dictionary
        self.counter = 0

    def __iter__(self):
        captions = self.coco.cocoTrain.loadAnns(self.annIds)
        for caption in captions:
            caption = caption['caption'].encode('ascii').strip().lower()
            caption = caption.replace('.', '')
            #add start and end tokens
            caption = '<start> ' + caption + ' .'
            words = caption.split()
            for i,word in enumerate(words):
                if not self.dictionary.has_key(word):
                    words[i] = '___' 
            yield words

class EmbeddingModel:
    def __init__(self, dictionary, modelFileName):
        self.model = trainOrGetModel(dictionary, modelFileName)
        self.dictionary = dictionary

    def predict(self, words):
        embedded = []
        for word in words:
            if not self.dictionary.has_key(word):
                word = '___'
            embedded.append(self.model.wv[word])

        return np.asarray(embedded)

def trainOrGetModel(dictionary, model_fname='semanticEmbedding.model'):
    if os.path.exists(model_fname):
        return Word2Vec.load(model_fname)
    
    sentenceIter = TrainSentences(dictionary)
    model = Word2Vec(sentenceIter, size=512)
    model.save(model_fname)
    return model
