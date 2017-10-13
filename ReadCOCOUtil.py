from pycocotools.coco import COCO
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
# import ipdb
from os import path
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
activations_dim = 4096 # Dimensionality of activations once imgs have been fwd propgated through vgg16

class ReadCOCOUtil:

    def __init__(self, dataDir='./coco'):
        self.dataDir = dataDir
        self.cocoTrain = COCO(dataDir + '/annotations/captions_train2014.json')
        self.activsDir = './fwd_prop'
        self.imgIdsTrain = self.cocoTrain.getImgIds()
        np.random.shuffle(self.imgIdsTrain)
        self.nbTrain = len(self.imgIdsTrain)

        self.cocoVal = COCO(dataDir + '/annotations/captions_val2014.json')
        self.imgIdsVal = self.cocoVal.getImgIds()
        np.random.shuffle(self.imgIdsVal)
        self.nbVal = len(self.imgIdsVal)

        self.trainBatchLoc = 0
        self.valBatchLoc = 0

    def getTrainBatchImgs(self, batch_size=1):

        # handles wrap-around of training data
        if self.trainBatchLoc + batch_size >= len(self.imgIdsTrain):
            imgIds = self.imgIdsTrain[self.trainBatchLoc:]
            newEnd = (self.trainBatchLoc + batch_size) - len(self.imgIdsTrain)
            imgIds = imgIds + self.imgIdsTrain[:newEnd] # Wrapping
            self.trainBatchLoc = newEnd
        else:
            imgIds = self.imgIdsTrain[self.trainBatchLoc:self.trainBatchLoc + batch_size ]
            self.trainBatchLoc = self.trainBatchLoc + batch_size
        # Load imgs
        imgs = self.cocoTrain.loadImgs(imgIds)
        annIds = self.cocoTrain.getAnnIds(imgIds= [i['id'] for i in imgs])
        anns = self.cocoTrain.loadAnns(annIds)
        imgs = self._readAndPreprocessImages(imgs, 'train2014')
        # Make sure that imgs have correct dimensions (4)
        imgs = imgs.squeeze()
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis, :, :, :]

        return (imgs, anns, imgIds)

    def getValidationBatchImgs(self, batch_size=1):

        if self.valBatchLoc + batch_size >= len(self.imgIdsVal):
            imgIds = self.imgIdsVal[self.valBatchLoc:]
            newEnd = (self.valBatchLoc + batch_size) - len(self.imgIdsVal)
            imgIds = imgIds + self.imgIdsVal[:newEnd] # Wrapping
            self.valBatchLoc = newEnd
        else:
            imgIds = self.imgIdsVal[self.valBatchLoc:self.valBatchLoc + batch_size ]
            self.valBatchLoc = self.valBatchLoc + batch_size
        # Load imgs
        imgs = self.cocoVal.loadImgs(imgIds)
        annIds = self.cocoVal.getAnnIds(imgIds=[i['id'] for i in imgs]);
        anns = self.cocoVal.loadAnns(annIds)
        imgs = self._readAndPreprocessImages(imgs, 'val2014')
        # Make sure that imgs have correct dimensions (4)
        imgs = imgs.squeeze()
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis, :, :, :]

        return (imgs, anns, imgIds)

    def getTrainBatchActvs(self, batch_size=1):
        # Preallocate
        imgs = np.zeros((batch_size, activations_dim))
        # handles wrap-around of training data
        if self.trainBatchLoc + batch_size >= len(self.imgIdsTrain):
            imgIds = self.imgIdsTrain[self.trainBatchLoc:]
            newEnd = (self.trainBatchLoc + batch_size) - len(self.imgIdsTrain)
            imgIds = imgIds + self.imgIdsTrain[:newEnd] # Wrapping
            self.trainBatchLoc = newEnd
        else:
            imgIds = self.imgIdsTrain[self.trainBatchLoc:self.trainBatchLoc + batch_size ]
            self.trainBatchLoc = self.trainBatchLoc + batch_size
        # Open activations of imgs using ids
        for i, imgId in enumerate(imgIds):
            imgs[i,:] = np.load(path.join(self.activsDir , 'train', str(imgIds[i]) + '.npy'))
        annIds = self.cocoTrain.getAnnIds(imgIds= imgIds)
        anns = self.cocoTrain.loadAnns(annIds)
        # Make sure that imgs have correct dimensions (4)
        imgs = imgs.squeeze()
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis, :, :, :]

        return (imgs, anns, imgIds)

    def getValidationBatchActvs(self, batch_size=1):
        # Preallocate
        imgs = np.zeros((batch_size, activations_dim))
        # handles wrap-around of validation data
        if self.valBatchLoc + batch_size >= len(self.imgIdsVal):
            imgIds = self.imgIdsVal[self.valBatchLoc:]
            newEnd = (self.valBatchLoc + batch_size) - len(self.imgIdsVal)
            imgIds = imgIds + self.imgIdsVal[:newEnd] # Wrapping
            self.valBatchLoc = newEnd
        else:
            imgIds = self.imgIdsVal[self.valBatchLoc:self.valBatchLoc + batch_size ]
            self.valBatchLoc = self.valBatchLoc + batch_size
        # Open activations of imgs using ids
        for i, imgId in enumerate(imgIds):
            imgs[i,:] = np.load(path.join(self.activsDir , 'validation', str(imgIds[i]) + '.npy'))
        annIds = self.cocoVal.getAnnIds(imgIds= imgIds)
        anns = self.cocoVal.loadAnns(annIds)
        # Make sure that imgs have correct dimensions (4)
        imgs = imgs.squeeze()
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis, :, :, :]

        return (imgs, anns, imgIds)


    def _readAndPreprocessImages(self, imgs, train_valid):
        imagesData = []
        for img_coco in imgs:
            fileName = self.dataDir + "/" + train_valid + '/' + img_coco['file_name']
            # print '\t' + fileName
            img = image.load_img(fileName, target_size=(224,224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            imagesData.append(x)

        return np.asarray(imagesData)
