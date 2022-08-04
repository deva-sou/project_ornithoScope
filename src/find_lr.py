import argparse
import json
import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras_yolov2.frontend import YOLO
from keras_yolov2.preprocessing import BatchGenerator, parse_annotation_csv
from keras_yolov2.utils import enable_memory_growth
from keras_yolov2.learning_rate_finder import LRFinder
from keras_yolov2.yolo_loss import YoloLoss

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config/new_config/config_iNat_train_Adam_OCS.json',
    help='path to configuration file')


def _main_(args):
    config_path = args.conf
    enable_memory_growth()

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    train_imgs, train_labels = parse_annotation_csv(config['data']['train_csv_file'],
                                                    config['model']['labels'],
                                                    config['data']['base_path'])

    valid_path = config['data']['valid_csv_file']

    if os.path.exists(valid_path):
        print(f"\n \nParsing {valid_path.split('/')[-1]}")
        valid_imgs, seen_valid_labels = parse_annotation_csv(valid_path,
                                                        config['model']['labels'],
                                                        config['data']['base_path'])
        split = False
    else:
        split = True

    if split:
        train_valid_split = int(0.85 * len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]
        
    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = train_labels.keys()
        with open("labels.json", 'w') as outfile:
            json.dump({"labels": list(train_labels.keys())}, outfile)


    ###############################
    #   Construct the model 
    ###############################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=(config['model']['input_size_h'], config['model']['input_size_w']),
                labels=config['model']['labels'],
                anchors=config['model']['anchors'],
                gray_mode=config['model']['gray_mode'],
                freeze=config['train']['freeze'])

    ###############################
    #   Start the training process 
    ###############################

    config['train']['optimizer']['lr_scheduler']['name'] = 'None'

    lr_finder_callback = LRFinder(start_lr=1e-7, end_lr=10, max_steps=150, smoothing=0.9)

    yolo.train(train_imgs=train_imgs,
               valid_imgs=valid_imgs,
               train_times=1,
               nb_epochs=1,
               learning_rate=config['train']['learning_rate'],
               batch_size=config['train']['batch_size'],
               object_scale=config['train']['object_scale'],
               no_object_scale=config['train']['no_object_scale'],
               coord_scale=config['train']['coord_scale'],
               class_scale=config['train']['class_scale'],
               saved_weights_name=config['train']['saved_weights_name'],
               early_stop=config['train']['early_stop'],
               workers=config['train']['workers'],
               max_queue_size=config['train']['max_queue_size'],
               tb_logdir=config['train']['tensorboard_log_dir'],
               optimizer_config=config['train']['optimizer'],
               iou_threshold=config['valid']['iou_threshold'],
               score_threshold=config['valid']['score_threshold'],
               policy='none',
               saved_pickles_path=None,
               custom_callbacks=[lr_finder_callback],
               sampling=False)
    
    lr_finder_callback.plot()
    writepathpng = 'plot_lucien/lr_finder_%d.png'
    writepathcsv= 'plot_lucien/lr_finder_argmin.csv'

    id = 0
    while os.path.exists(writepathpng % id):
        
        id += 1
    plt.savefig(writepathpng % id) #à modififier selon les documents
    plt.show()
    
    '''with open('plot_lucien/lr_finder_argmin.csv', 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
           
            
            # write a row to the csv file
            print(lr_finder_callback)
            row=np.argmin(lr_finder_callback)
            writer.writerow(row )'''
        

if __name__ == '__main__':
    _args = argparser.parse_args()
    gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    with tf.device('/GPU:' + gpu_id):
        _main_(_args)