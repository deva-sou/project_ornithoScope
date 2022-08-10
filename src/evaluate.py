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

# tf.debugging.set_log_device_placement(True)

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config/config_lab_mobilenetV1.json',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='',
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
    count = 0
    while True:
        try:
            os.mkdir(path + f'_{count}')
            break
        except:
            count += 1
    path += f'_{count}'
    for test_path in validation_paths:
        if os.path.exists(test_path):
            print(f"\n \nParsing {test_path.split('/')[-1]}")
            test_imgs, seen_valid_labels = parse_annotation_csv(test_path,
                                                            config['model']['labels'],
                                                            config['data']['base_path'])
                                                            
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
                        'TRUE_BOX_BUFFER': 10
                    }
            
            test_generator = BatchGenerator(test_imgs, 
                                                generator_config,
                                                norm=yolo._feature_extractor.normalize,
                                                jitter=False,
                                                shuffle=False)
            test_eval = MapEvaluation(yolo, test_generator,
                                    iou_threshold=config['valid']['iou_threshold'],
                                    score_threshold=config['valid']['score_threshold'],
                                    label_names=config['model']['labels'],
                                    model_name=config['model']['backend'])

            print('Number of valid images: ', len(test_imgs))

            print('Computing metrics per classes...')
            (boxes_preds, bad_boxes_preds,
            class_predictions, class_metrics, class_res, class_p_global, class_r_global, class_f1_global,
            bbox_predictions, bbox_metrics, bbox_res, bbox_p_global, bbox_r_global, bbox_f1_global
            ) = test_eval.compute_P_R_F1()
            print('Done.')

            test_name = test_path.split('/')[-1].split('.')[0]
            print("For", test_name)
            print('VALIDATION LABELS: ', seen_valid_labels)
            print('Final results:')

            print('\nClass metrics:')
            class_mean_P, class_mean_R, class_mean_F1 = print_results_metrics_per_classes(class_res, seen_valid_labels)
            print(f"Globals class: P={class_p_global} R={class_r_global} F1={class_f1_global}")
            print(f"Means bbox: P={class_mean_P} R={class_mean_R} F1={class_mean_F1}")

            print('\nClass metrics:')
            bbox_mean_P, bbox_mean_R, bbox_mean_F1 = print_results_metrics_per_classes(bbox_res, seen_valid_labels)
            print(f"Globals bbox: P={bbox_p_global} R={bbox_r_global} F1={bbox_f1_global}")
            print(f"Means bbox: P={bbox_mean_P} R={bbox_mean_R} F1={bbox_mean_F1}")

            global_results = [class_p_global,class_r_global,class_f1_global]
            pickle.dump(class_predictions, open( f"{path}/prediction_TP_FP_FN_{config['model']['backend']}_{test_name}.p", "wb" ) )
            pickle.dump(class_metrics, open( f"{path}/TP_FP_FN_{config['model']['backend']}_{test_name}.p", "wb" ) )
            pickle.dump(class_res, open( f"{path}/P_R_F1_{config['model']['backend']}_{test_name}.p", "wb" ) )
            pickle.dump(global_results, open( f"{path}/P_R_F1_global_{config['model']['backend']}_{test_name}.p", "wb" ) )  
            pickle.dump(boxes_preds, open(f"{path}/boxes_{config['model']['backend']}_{test_name}.p", "wb"))
            pickle.dump(bad_boxes_preds, open(f"{path}/bad_boxes_{config['model']['backend']}_{test_name}.p", "wb"))

if __name__ == '__main__':
    _args = argparser.parse_args()
    gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    with tf.device('/GPU:' + gpu_id):
        _main_(_args)
