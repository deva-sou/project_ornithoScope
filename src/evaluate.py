#! /usr/bin/env python3
from keras_yolov2.preprocessing import parse_annotation_csv
from keras_yolov2.preprocessing import BatchGenerator
from keras_yolov2.utils import enable_memory_growth,print_results_metrics_per_classes
from keras_yolov2.frontend import YOLO
from keras_yolov2.map_evaluation import MapEvaluation
import argparse
import json
import os
import pickle
from datetime import datetime
import tensorflow as tf

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config/config_to_train/Adam_OCS_batchsize16_1e-5à1e-2.json', #à changer selon les config
    help='path to configuration file')

argparser.add_argument(
    '-i',
    '--iou',
    default=0.5,
    help='IOU threshold',
    type=float)

argparser.add_argument(
    '-w',
    '--weights',
    default='data/saved_weights/new_weights/Adam_OCS_batchsize16_1e-5à1e-2_v0_bestLoss.h5', #à changer selon les config
    help='path to pretrained weights')

argparser.add_argument(
  '-l',
  '--lite',
  default='',
  type=str,
  help='Path to tflite model')


def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    lite_path = args.lite
    
    enable_memory_growth()

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    if weights_path == '':
        weights_path = config['train']['pretrained_weights']

    ##########################
    #   Parse the annotations 
    ##########################
    without_valid_imgs = False

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation_csv(config['data']['train_csv_file'],
                                                    config['model']['labels'],
                                                    config['data']['base_path'])

       # remove samples without objects in the image
    for i in range(len(train_imgs)-1, 0, -1):
        if len(train_imgs[i]['object']) == 0:
            del train_imgs[i]

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        # print('Seen labels:\t', train_labels)
        # print('Given labels:\t', config['model']['labels'])
        # print('Overlap labels:\t', overlap_labels)           

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Evaluate on all seen labels.')
        config['model']['labels'] = train_labels.keys()
        with open("labels.json", 'w') as outfile:
            json.dump({"labels": list(train_labels.keys())}, outfile)
        
    ########################
    #   Construct the model 
    ########################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=(config['model']['input_size_h'], config['model']['input_size_w']),
                labels=config['model']['labels'],
                anchors=config['model']['anchors'],
                gray_mode=config['model']['gray_mode'])

    #########################################
    #   Load the pretrained weights (if any) 
    #########################################

    if weights_path != '':
        print("Loading pre-trained weights in", weights_path)
        yolo.load_weights(weights_path)
    elif os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])
    else:
        raise Exception("No pretrained weights found.")
    

    # Use tflite
    if lite_path != '':
        yolo.load_lite(lite_path)

    #########################
    #   Evaluate the network
    #########################

     # parse annotations of the validation set, if any.
    validation_paths = config['data']['test_csv_file']
    print(validation_paths)
    directory_name = f"{config['model']['backend']}_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
    print("Directory name for metrics: ", directory_name)
    parent_dir = config['data']['saved_pickles_path']
    path = os.path.join(parent_dir, directory_name)
    os.mkdir(path)
    for valid_path in validation_paths:
        if os.path.exists(valid_path):
            print(f"\n \nParsing {valid_path.split('/')[-1]}")
            valid_imgs, seen_valid_labels = parse_annotation_csv(valid_path,
                                                            config['model']['labels'],
                                                            config['data']['base_path'])
            #print("computing mAP for iou threshold = {}".format(args.iou))
            generator_config = {
                        'IMAGE_H': yolo._input_size[0],
                        'IMAGE_W': yolo._input_size[1],
                        'IMAGE_C': yolo._input_size[2],
                        'GRID_H': yolo._grid_h,
                        'GRID_W': yolo._grid_w,
                        'BOX': yolo._nb_box,
                        'LABELS': yolo.labels,
                        'CLASS': len(yolo.labels),
                        'ANCHORS': yolo._anchors,
                        'BATCH_SIZE': 4,
                        'TRUE_BOX_BUFFER': 10 # yolo._max_box_per_image,
                    } 
            valid_generator = BatchGenerator(valid_imgs, 
                                                generator_config,
                                                norm=yolo._feature_extractor.normalize,
                                                jitter=False)
            valid_eval = MapEvaluation(yolo, valid_generator,
                                    iou_threshold=args.iou,
                                    label_names=config['model']['labels'],
                                    model_name=config['model']['backend'])
            print('Number of valid images: ', len(valid_imgs))
            print('Computing metrics per classes...')
            predictions,class_metrics,class_res,p_global, r_global,f1_global = valid_eval.evaluate_map()
            print('Done.')
            #print('\nTask: ', valid_path)
            task_name = valid_path.split('/')[-1].split('.')[0]
            print("For ",task_name)
            print('VALIDATION LABELS: ', seen_valid_labels)
            print('Final results:')
            print_results_metrics_per_classes(class_res)
            print(f"Globals: P={p_global} R={r_global} F1={f1_global}\n")
            global_results = [p_global,r_global,f1_global]
            pickle.dump(predictions, open( f"{path}/prediction_TP_FP_FN_{config['model']['backend']}_{task_name}.p", "wb" ) )
            pickle.dump(class_metrics, open( f"{path}/TP_FP_FN_{config['model']['backend']}_{task_name}.p", "wb" ) )
            pickle.dump(class_res, open( f"{path}/P_R_F1_{config['model']['backend']}_{task_name}.p", "wb" ) )
            pickle.dump(global_results, open( f"{path}/P_R_F1_global_{config['model']['backend']}_{task_name}.p", "wb" ) )  

if __name__ == '__main__':
    _args = argparser.parse_args()
    gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    with tf.device('/GPU:' + gpu_id):
        _main_(_args)
