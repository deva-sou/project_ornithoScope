#! /usr/bin/env python3

import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import numpy as np
from tensorflow import keras

from keras_yolov2.frontend import YOLO
from keras_yolov2.preprocessing import parse_annotation_csv, BatchGenerator
from keras_yolov2.utils import enable_memory_growth, import_feature_extractor


argparser = argparse.ArgumentParser(
    description='Test Data Augmentation')

argparser.add_argument(
    '-c',
    '--conf',
    default='config/config_lab_mobilenetV1_test_data_aug.json',
    help='path to configuration file')


def _main_(args):
    config_path = args.conf
    enable_memory_growth()

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    split = False
    # parse annotations of the training set
    imgs, labels = parse_annotation_csv(config['data']['train_csv_file'],
                                                    config['model']['labels'],
                                                    config['data']['base_path'])
                                                    
    feature_extractor = import_feature_extractor(config['model']['backend'], (config['model']['input_size_h'], config['model']['input_size_w'], 3))
    print(feature_extractor.feature_extractor.summary())
    print(feature_extractor.get_output_shape())
    grid_h, grid_w = feature_extractor.get_output_shape()
                                           
                                         
    generator_config = {
            'IMAGE_H': config['model']['input_size_h'],
            'IMAGE_W': config['model']['input_size_w'],
            'IMAGE_C': 3,
            'GRID_H': grid_h,
            'GRID_W': grid_w,
            'BOX': len(config['model']['anchors'])/2,
            'LABELS': config['model']['labels'],
            'CLASS': len(config['model']['labels']),
            'ANCHORS': config['model']['anchors'],
            'BATCH_SIZE': config['train']['batch_size']
        }                                       
                                                    
    train_generator = BatchGenerator(   imgs, 
                                        generator_config, 
                                        norm=feature_extractor.normalize,
                                        policy_container = config['train']['augmentation'])
    for i in range(10):
        img, all_objs = train_generator.aug_image(imgs[i])
        fig, ax = plt.subplots()
        plt.imshow(img)
        for obj in all_objs:
            plt.gca().add_patch(Rectangle((obj['xmin'], obj['ymin']), obj['xmax']-obj['xmin'], obj['ymax']-obj['ymin'], edgecolor='red', facecolor='none', lw=4))
        plt.show()
	
    
    
if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)
