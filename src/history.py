import os
import sys

import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Reshape, Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, ExponentialDecay

from keras_yolov2.cosine_decay import WarmUpCosineDecayScheduler
from keras_yolov2.map_evaluation import MapEvaluation
from keras_yolov2.preprocessing import BatchGenerator
from keras_yolov2.utils import decode_netout, import_feature_extractor, import_dynamically
from keras_yolov2.yolo_loss import YoloLoss


with open('data/pickles/history/history_data/saved_weights/new_weights/MobileNet_caped300_data_aug_v0_ADAM_bestLoss.h5.p', 'rb') as input_file:
    e = pickle.load(input_file)
    print(e)