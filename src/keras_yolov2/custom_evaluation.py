#! /usr/bin/env python3
from keras_yolov2.preprocessing import parse_annotation_xml, parse_annotation_csv
from keras_yolov2.preprocessing import BatchGenerator
from keras_yolov2.utils import enable_memory_growth
from keras_yolov2.frontend import YOLO
import argparse
import json
import os
import pickle

import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2
from .utils import compute_overlap, compute_ap


class MapEvaluation(keras.callbacks.Callback):
    """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
    """

    def __init__(self, yolo, generator,
                 iou_threshold=0.5,
                 score_threshold=0.5,
                 save_path=None,
                 period=1,
                 save_best=False,
                 save_name=None,
                 tensorboard=None):

        super().__init__()
        self._yolo = yolo
        self._generator = generator
        self._iou_threshold = iou_threshold
        self._score_threshold = score_threshold
        self._save_path = save_path
        self._period = period
        self._save_best = save_best
        self._save_name = save_name
        self._tensorboard = tensorboard

        self.bestMap = 0

        if not isinstance(self._tensorboard, keras.callbacks.TensorBoard) and self._tensorboard is not None:
            raise ValueError("Tensorboard object must be a instance from keras.callbacks.TensorBoard")

    def on_epoch_end(self, epoch, logs={}):

        if epoch % self._period == 0 and self._period != 0:
            precision,recall,f1score,_map, average_precisions = self.evaluate_map()
            print('\n')
            for label, average_precision in average_precisions.items():
                print(self._yolo.labels[label], '{:.4f}'.format(average_precision))
            print('mAP: {:.4f}'.format(_map))

            if self._save_best and self._save_name is not None and _map > self.bestMap:
                print("mAP improved from {} to {}, saving model to {}.".format(self.bestMap, _map, self._save_name))
                self.bestMap = _map
                self.model.save(self._save_name)
            else:
                print("mAP did not improve from {}.".format(self.bestMap))

            if self._tensorboard is not None:
                with summary_ops_v2.always_record_summaries():
                    with self._tensorboard._val_writer.as_default():
                        name = "mAP"  # Remove 'val_' prefix.
                        summary_ops_v2.scalar('epoch_' + name, _map, step=epoch)

    def evaluate_map(self):
        precisions,recalls,f1_scores,average_precisions = self._calc_avg_precisions()
        _map = sum(average_precisions.values()) / len(average_precisions)

        return precisions,recalls,f1_scores,_map, average_precisions

    def _calc_avg_precisions(self):
        # gather all detections and annotations
        # all_detections = [[None for _ in range(self._generator.num_classes())]
        #                   for _ in range(self._generator.size())]
        # all_annotations = [[None for _ in range(self._generator.num_classes())]
        #                    for _ in range(self._generator.size())]
        all_detections = [[[] for _ in range(self._generator.num_classes())]
                           for _ in range(self._generator.size())]
        all_annotations = [[[] for _ in range(self._generator.num_classes())]
                           for _ in range(self._generator.size())]
        for i in range(self._generator.size()):
            raw_image, img_name = self._generator.load_image(i)
            raw_height, raw_width, _ = raw_image.shape  

            # make the boxes and the labels
            # if i % 50 == 0 : 
            #     print(f"prediction number {i} done")
            print(f"prediction number {i} done")
            pred_boxes = self._yolo.predict(raw_image,
                                            iou_threshold=self._iou_threshold,
                                            score_threshold=self._score_threshold)

            score = np.array([box.score for box in pred_boxes])
            if len(score) != 0:
                print('score ', score)
            pred_labels = np.array([box.get_label() for box in pred_boxes])
            if len(pred_labels) != 0:
                print('pred label ', pred_labels)
            if len(pred_boxes) > 0:
                print(pred_boxes)
                pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height, box.xmax * raw_width,
                                        box.ymax * raw_height, box.score] for box in pred_boxes])
                print('pred boxes ',pred_boxes)
            else:
                pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(self._generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = self._generator.load_annotation(i)
            
            if annotations.shape[1] > 0:
                # copy ground truth to all_annotations
                for label in range(self._generator.num_classes()):
                    all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
                
            print("all ann ", all_annotations[i])
        # print('all_detections ', all_detections)
        # print('all_annotations ', all_annotations)
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        precisions = {}
        recalls = {}
        f1_scores = {}

        for label in range(self._generator.num_classes()):
            print("Calculation on label: ", label)
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(self._generator.size()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += len(annotations)
                detected_annotations = []
                if len(detections) != 0: 
                    print(f"detections {detections} \n label {label}")
                    print(f"annotations {annotations}")


                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= self._iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
            f1_score = 2*precision * recall/(precision + recall)
            print(f"label {label}, precision {precision}, recall {recall}, f1_score {f1_score}")
            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision
            precisions[label] = precision
            recalls[label] = recall
            f1_scores[label] = f1_score

        print('computing done')
        
        return precisions,recalls,f1_scores,average_precisions

############################################
#                   MAIN
############################################

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config/config_lab_mobilenetV1.json',
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
    if config['parser_annotation_type'] == 'xml':
        # parse annotations of the training set
        train_imgs, train_labels = parse_annotation_xml(config['train']['train_annot_folder'], 
                                                        config['train']['train_image_folder'],
                                                        config['model']['labels'])

        # parse annotations of the validation set, if any.
        if os.path.exists(config['valid']['valid_annot_folder']):
            valid_imgs, valid_labels = parse_annotation_xml(config['valid']['valid_annot_folder'], 
                                                            config['valid']['valid_image_folder'],
                                                            config['model']['labels'])
        else:
            without_valid_imgs = True

    elif config['parser_annotation_type'] == 'csv':
        # parse annotations of the training set
        train_imgs, train_labels = parse_annotation_csv(config['train']['train_csv_file'],
                                                        config['model']['labels'],
                                                        config['train']['train_csv_base_path'])

        # parse annotations of the validation set, if any.
        if os.path.exists(config['valid']['valid_csv_file']):
            valid_imgs, valid_labels = parse_annotation_csv(config['valid']['valid_csv_file'],
                                                            config['model']['labels'],
                                                            config['valid']['valid_csv_base_path'])
            print('\t \t len valid images ', len(valid_imgs))
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

    print("computing mAP for iou threshold = {}".format(args.iou))
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
        valid_eval = MapEvaluation(yolo, valid_generator,
                                   iou_threshold=args.iou)

        print('computing metrics per classes')
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
                                iou_threshold=args.iou)
        print('computing metrics per classes')
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
