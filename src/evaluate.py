#! /usr/bin/env python3
from keras_yolov2.preprocessing import parse_annotation_xml, parse_annotation_csv
from keras_yolov2.preprocessing import BatchGenerator
from keras_yolov2.utils import enable_memory_growth
from keras_yolov2.frontend import YOLO
from keras_yolov2.map_evaluation import MapEvaluation
import argparse
import json
import os
import pickle

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
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
    default='',
    help='path to pretrained weights')


def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    
    enable_memory_growth()

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    if weights_path == '':
        weights_path = config['train']['pretrained_weights']

    ##########################
    #   Parse the annotations 
    ##########################
    without_valid_imgs = False
    # if config['parser_annotation_type'] == 'xml':
    #     # parse annotations of the training set
    #     train_imgs, train_labels = parse_annotation_xml(config['train']['train_annot_folder'], 
    #                                                     config['train']['train_image_folder'],
    #                                                     config['model']['labels'])

    #     # parse annotations of the validation set, if any.
    #     if os.path.exists(config['valid']['valid_annot_folder']):
    #         valid_imgs, valid_labels = parse_annotation_xml(config['valid']['valid_annot_folder'], 
    #                                                         config['valid']['valid_image_folder'],
    #                                                         config['model']['labels'])
    #     else:
    #         without_valid_imgs = True

    if config['parser_annotation_type'] == 'csv':
        # parse annotations of the training set
        train_imgs, train_labels = parse_annotation_csv(config['train']['train_csv_file'],
                                                        config['model']['labels'],
                                                        config['train']['train_csv_base_path'])

        # parse annotations of the validation set, if any.
        if os.path.exists(config['valid']['valid_csv_file']):
            print(f'\t csv found')
            valid_imgs, valid_labels = parse_annotation_csv(config['valid']['valid_csv_file'],
                                                            config['model']['labels'],
                                                            config['valid']['valid_csv_base_path'])
            print('\n \t \t len valid images ', len(valid_imgs),'\n')
        else:
            without_valid_imgs = True
    else:
        raise ValueError("'parser_annotations_type' must be 'xml' or 'csv' not {}.".format(config['parser_annotations_type']))

    # remove samples without objects in the image
    for i in range(len(train_imgs)-1, 0, -1):
        if len(train_imgs[i]['object']) == 0:
            del train_imgs[i]

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)           

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

    #########################
    #   Evaluate the network
    #########################

    print("calculing mAP for iou threshold = {}".format(args.iou))
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
    if not without_valid_imgs:
        valid_generator = BatchGenerator(valid_imgs, 
                                         generator_config,
                                         norm=yolo._feature_extractor.normalize,
                                         jitter=False)
        print('VALIDATION LABELS: ', config['model']['labels'])
        valid_eval = MapEvaluation(yolo, valid_generator,
                                   iou_threshold=args.iou,
                                   label_names=config['model']['labels'],
                                   model_name=config['model']['backend'])

        print('calculing metrics per classes')
        precisions,recalls,f1_scores,_map, average_precisions = valid_eval.evaluate_map()
        for label, average_precision in average_precisions.items():
            print(f"map {yolo.labels[label]}, {average_precision}")
        for label, precision in precisions.items():
            print(f"precision {yolo.labels[label]}, {precision}") 
        for label, recall in recalls.items():
            print(f"recall {yolo.labels[label]}, {recall}")
        for label, f1_score in f1_scores.items():
            print(f"f1 {yolo.labels[label]}, {f1_score}")
        pickle.dump(precisions, open( f"keras_yolov2/pickles/precisions_{config['model']['backend']}.p", "wb" ) )
        pickle.dump(recalls, open( f"keras_yolov2/pickles/recalls_{config['model']['backend']}.p", "wb" ) )
        pickle.dump(f1_scores, open( f"keras_yolov2/pickles/f1_scores_{config['model']['backend']}.p", "wb" ) )
        pickle.dump(average_precisions, open( f"keras_yolov2/pickles/average_precisions_{config['model']['backend']}.p", "wb" ) )
        
        print('validation dataset mAP: {}\n'.format(_map))


    else:
        train_generator = BatchGenerator(train_imgs, 
                                        generator_config, 
                                        norm=yolo._feature_extractor.normalize,
                                        jitter=False)  
        train_eval = MapEvaluation(yolo, train_generator,
                                iou_threshold=args.iou,
                                label_names=config['model']['labels'],
                                model_name=config['model']['backend'])
        print('calculing metrics per classes')
        precisions,recalls,f1_scores,_map, average_precisions = train_eval.evaluate_map()
        pickle.dump(precisions, open( f"keras_yolov2/pickles/{config['model']['backend']}_precisions.p", "wb" ) )
        pickle.dump(recalls, open( f"keras_yolov2/pickles/{config['model']['backend']}_recalls.p", "wb" ) )
        pickle.dump(f1_scores, open( f"keras_yolov2/pickles/{config['model']['backend']}_f1_scores.p", "wb" ) )
        pickle.dump(average_precisions, open( f"keras_yolo2/keras_yolov2/pickles/{config['model']['backend']}_average_precisions.p", "wb" ) )
    
    for label, average_precision in average_precisions.items():
        print(f"map {yolo.labels[label]}, {average_precision}")
    for label, precision in precisions.items():
        print(f"precision {yolo.labels[label]}, {precision}") 
    for label, recall in recalls.items():
        print(f"recall {yolo.labels[label]}, {recall}")
    for label, f1_score in f1_scores.items():
        print(f"f1 {yolo.labels[label]}, {f1_score}")
    print('mAP: {}'.format(_map))


if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)
