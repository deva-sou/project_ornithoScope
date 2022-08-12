import os
import shutil
import sys
import copy
import re
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from .backend import (  EfficientNetB0Feature, EfficientNetV2B0Feature, MobileNetV3LargeFeature,
                        MobileNetV3SmallFeature,
                        TinyYoloFeature,
                        FullYoloFeature,
                        MobileNetFeature,
                        MobileNetV2Feature,
                        SqueezeNetFeature,
                        Inception3Feature,
                        VGG16Feature,
                        ResNet50Feature,
                        BaseFeatureExtractor)


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1
        self.id = -1

    def get_label(self):
        #if self.label == -1: -> Bug sur le nombre de bbox prédites
        self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        #if self.score == -1: -> Bug sur le nombre de bbox prédites
        self.score = self.classes[self.get_label()]*self.c
        return self.score
    
    def copy(self):
        return BoundBox(self.xmin, self.ymin, self.xmax, self.ymax, self.c, self.classes.copy())

    def __repr__(self):
        """
        Helper method for printing the object's values
        :return:
        """
        return "<BoundBox({}, {}, {}, {}, {}, {}, {})>\n".format(
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.get_label(),
            self.get_score(),
            self.id
        )


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def draw_boxes(image, boxes, labels):
    image_h, image_w, _ = image.shape

    color_levels = [0, 255, 128, 64, 32]
    colors = []
    for r in color_levels:
        for g in color_levels:
            for b in color_levels:
                if r == g and r == b:  # prevent grayscale colors
                    continue
                colors.append((b, g, r))

    for box in boxes:
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)

        line_width_factor = int(min(image_h, image_w) * 0.005)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[box.get_label()], line_width_factor * 2)
        cv2.putText(image, "{} {:.3f}".format(labels[box.get_label()], box.get_score()),
                    (xmin, ymin + line_width_factor * 10), cv2.FONT_HERSHEY_PLAIN, 2e-3 * min(image_h, image_w),
                    (0, 255, 0), line_width_factor)
        if box.id >= 0:
            cv2.putText(image, "ID : {}".format(box.id),
                        (xmin, ymin + 2 * line_width_factor * 10), cv2.FONT_HERSHEY_PLAIN, 2e-3 * min(image_h, image_w),
                        (0, 255, 0), line_width_factor)

    return image


def decode_netout(netout, anchors, nb_class, obj_threshold=0.5, nms_threshold=0.3):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4] = _sigmoid(netout[..., 4])
    netout[..., 5:] = _softmax(netout[..., 5:])

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]
                confidence = netout[row, col, b, 4]

                if confidence >= obj_threshold:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]

                    x = (col + _sigmoid(x)) / grid_w  # center position, unit: image width
                    y = (row + _sigmoid(y)) / grid_h  # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w  # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h  # unit: image height

                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence, classes)
                    boxes.append(box)
    # print(boxes)
    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0   # We changed index_i to index_j
                        #boxes[index_j].score = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    # print(boxes)
    return boxes


def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / np.min(x) * t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def import_dynamically(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def import_feature_extractor(backend, input_size, freeze=False):
    if backend == 'Inception3':
        feature_extractor = Inception3Feature(input_size, freeze=freeze)
    elif backend == 'SqueezeNet':
        feature_extractor = SqueezeNetFeature(input_size, freeze=freeze)
    elif backend.startswith('EfficientNetB0'):
        feature_extractor = EfficientNetB0Feature(input_size, freeze=freeze)
    elif backend.startswith('EfficientNetV2B0'):
        feature_extractor = EfficientNetV2B0Feature(input_size, freeze=freeze)
    elif backend.startswith('MobileNetV3Small'):
        # Extract ALPHA
        alpha = re.search("alpha=([0-1]?\.[0-9]*)", backend)
        alpha = float(alpha.group(1)) if alpha != None else 1.0
        # Build MobileNetFeature
        feature_extractor = MobileNetV3SmallFeature(input_size, freeze=freeze, alpha=alpha)
    elif backend.startswith('MobileNetV3Large'):
        # Extract ALPHA
        alpha = re.search("alpha=([0-1]?\.[0-9]*)", backend)
        alpha = float(alpha.group(1)) if alpha != None else 1.0
        # Build MobileNetFeature
        feature_extractor = MobileNetV3LargeFeature(input_size, freeze=freeze, alpha=alpha)
    elif backend.startswith('MobileNetV2'):
        # Extract ALPHA
        alpha = re.search("alpha=([0-1]?\.[0-9]*)", backend)
        alpha = float(alpha.group(1)) if alpha != None else 1.0
        # Build MobileNetFeature
        feature_extractor = MobileNetV2Feature(input_size, freeze=freeze, alpha=alpha)
    elif backend.startswith('MobileNet'):
        # Extract ALPHA
        alpha = re.search("alpha=([0-1]?\.[0-9]*)", backend)
        alpha = float(alpha.group(1)) if alpha != None else 1.0
        # Build MobileNetFeature
        feature_extractor = MobileNetFeature(input_size, freeze=freeze, alpha=alpha)
    elif backend == 'Full Yolo':
        feature_extractor = FullYoloFeature(input_size, freeze=freeze)
    elif backend == 'Tiny Yolo':
        feature_extractor = TinyYoloFeature(input_size, freeze=freeze)
    elif backend == 'VGG16':
        feature_extractor = VGG16Feature(input_size, freeze=freeze)
    elif backend == 'ResNet50':
        feature_extractor = ResNet50Feature(input_size, freeze=freeze)
    elif os.path.dirname(backend) != "":
        base_path = os.path.dirname(backend)
        sys.path.append(base_path)
        custom_backend_name = os.path.basename(backend)
        custom_backend = import_dynamically(custom_backend_name)
        feature_extractor = custom_backend(input_size)
        if not issubclass(custom_backend, BaseFeatureExtractor):
            raise RuntimeError('You are trying to import a custom backend, your backend must be in inherited from '
                               ' "backend.BaseFeatureExtractor".')
        print('Using a custom backend called {}.'.format(custom_backend_name))
    else:
        raise RuntimeError('Architecture not supported! Only support Full Yolo, Tiny Yolo, MobileNet, SqueezeNet, VGG16'
                           ', ResNet50, or Inception3 at the moment!')
   
    return feature_extractor


# these funcition are from imutils, you can check this library here: https://github.com/jrosebr1/imutils
# just added this function to have less dependencies
def list_images(base_path, valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):
    # return the set of files that are valid
    return list_files(base_path, valid_exts, contains=contains)


def list_files(base_path, valid_exts="", contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(base_path):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(valid_exts):
                # construct the path to the image and yield it
                image_path = os.path.join(rootDir, filename)
                yield image_path


def enable_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def from_id_to_label_name(list_label, list_label_id):
    #list_label = ["MESCHA","VERDEUR",...]
    #list_label_id = [2,2,2]
    #print(list_label)
    #print('from', list_label_id)
    list_ret = []
    for id in list_label_id: 
        #print('id', id)
        #print('>',list_label[int(id)])
        list_ret.append(list_label[int(id)])
    #print('to', list_ret)
    return list_ret


def compute_bbox_TP_FP_FN(pred_boxes, true_boxes, list_of_classes):
    # Store TP, FP and FN labels
    predictions = {'TP': [], 'FP': [], 'FN': []}

    # Store nicely predicted box ids
    good_boxes = []

    # Loop on every predicted boxes
    for k, box_pred in enumerate(pred_boxes):
        pred_label = box_pred.get_label()

        # Get true box ids with the same label
        true_indexs = [i for i in range(len(true_boxes)) if true_boxes[i].get_label() == pred_label]

        # Sort them by IoU
        true_indexs = sorted(true_indexs, key=lambda i : bbox_iou(true_boxes[i], box_pred), reverse=True)

        # The predicted box does not correspond to any true box (using class only)
        if len(true_indexs) == 0:
            continue
        
        # If the IoU is correct, it is a TP
        if bbox_iou(true_boxes[true_indexs[0]], box_pred) > 0.5:
            predictions['TP'].append(list_of_classes[pred_label])
            true_boxes.pop(true_indexs[0])
            good_boxes.append(k)
            continue
    
    # Remaining predicted boxes are FP
    predictions['FP'] = [list_of_classes[box_pred.get_label()] for k, box_pred in enumerate(pred_boxes) if not k in good_boxes]

    # Remaining ture boxes are FN
    predictions['FN'] = [list_of_classes[box_true.get_label()] for box_true in true_boxes]

    return predictions


def compute_class_TP_FP_FN(dict_pred):
    true_labels = dict_pred['true_name']
    pred_labels = dict_pred['predictions_name']
    TP = []
    FP = copy.deepcopy(pred_labels)
    FN = copy.deepcopy(true_labels)
    for pl in pred_labels:
        if pl in true_labels:
            true_labels.remove(pl)
            try:
                FP.remove(pl)
            except ValueError:
                "not in the list"
            try:
                FN.remove(pl)
            except ValueError:
                "not in the list"
            try:
                TP.append(pl)
            except ValueError:
                "not in the list"
    dict_pred['TP'] = TP
    dict_pred['FN'] = FN 
    dict_pred['FP'] = FP

# fonction de calcul des précisions recall et score f1 de chaque classe
def get_precision_recall_from_prediction(list_of_results, list_of_classes):
    class_metrics = []
    for classes in list_of_classes:
        class_metrics.append({'Specie':classes,'TP':0, 'FP':0, 'FN':0})
    
    for i in range(len(list_of_results)):
        # pred_labels = list_of_results[i]['predictions_name']
        # print('pred', pred_labels)
        # true_labels = list_of_results[i]['true_name']
        # print('true', true_labels)
        TP = list_of_results[i]['TP']
        FP = list_of_results[i]['FP']
        FN = list_of_results[i]['FN']
        # print(f'TP {TP}, FN {FN}, FP {FP}')
        
        for lab in TP:
            class_metrics[list_of_classes.index(lab)]['TP'] += 1

        for lab in FN:
            class_metrics[list_of_classes.index(lab)]['FN'] += 1
            
        for lab in FP:
            class_metrics[list_of_classes.index(lab)]['FP'] += 1
    #print('loc', class_metrics)
    return class_metrics

def results_metrics_per_classes(class_metrics):
    class_res = []  
    i = 0
    for class_bird in class_metrics:
        specie = class_bird['Specie']
        TP = class_bird['TP']
        FP = class_bird['FP']
        FN = class_bird['FN']
        #print('TP-FP-FN')
        #print(f'{specie} {TP} {FP} {FN}')
        if (TP+FP) == 0: 
            P = 0
        else: 
            P = TP/(TP+FP)
        if (TP+FN) == 0: 
            R = 0
        else:
            R = TP/(TP+FN)
        if P == 0 and R == 0:
            F_score = 0
        else:
            F_score = 2*P*R/(P+R)
        #print('P-R-F_score')
        #print(f'{specie} {P} {R} {F_score}')
        class_res.append({'Specie': specie,'Precision': round(P,3), 'Rappel': round(R,3), 'F-score': round(F_score,3)})
        i += 1
    return class_res

# fonction qui print les metriques par classes p,r,f1
def print_results_metrics_per_classes(class_res, seen_valid):
    P_list = []
    R_list = []
    F1_list = []
    for res in class_res:
        if res['Specie'] in seen_valid:
            P = res['Precision']
            R = res['Rappel']
            F1 = res['F-score']
            P_list.append(P)
            R_list.append(R)
            F1_list.append(F1)
            print(f"Specie = {res['Specie']}, Precision = {P} - Rappel = {R} - F-score = {F1} ")
    return round(np.mean(P_list), 3), round(np.mean(R_list), 3), round(np.mean(F1_list), 3)

def get_p_r_f1_global(class_metrics):
    # class_metrics = {'TP': 2434, 'FP': 283, 'FN': 80}
    l = len(class_metrics)
    tp = fp = fn = p = r = f = 0.0
    for class_bird in class_metrics:
        if class_bird['TP'] !=0 and class_bird['FP'] !=0 and class_bird['FN'] !=0:
            tp += class_bird['TP']
            fp += class_bird['FP']
            fn += class_bird['FN']
        else:
            l-=1
    if (tp+fp) != 0.0:
        p = tp/(tp+fp)
    else:
        p = 0
    if (tp+fp) != 0.0:
        r = tp/(tp+fn)
    else:
        p = 0.0
    if (p+r) != 0.0:
        f = 2*((p*r)/(p+r))
    else:
        f = 0.0
    return round(p,3),round(r,3),round(f,3) 

def df_to_grouped_csv(path_input_test_csv, path_input_test_per_task_csv):
    dataset_test = pd.read_csv("/home/acarlier/code/OrnithoScope_keras/keras_yolo2/birds_data_csv/input_test.csv",
                           names=['task_name','xmin','xmax','ymin','ymax','label','h','w'])
    df = copy.deepcopy(dataset_test)
    df[['_task_name', 'file_name']] = df['task_name'].str.split('/', 1, expand=True)
    df_group = df.groupby(['_task_name'])
    for name, group in df_group:
        outfile = f"{name}.csv"
        #group['task_name'] = group['_task_name'] + '-' + group['file_name']
        group[['task_name','xmin','xmax','ymin','ymax','label','h','w']].to_csv(
            f"path_input_test_per_task_csv/{outfile}", 
            index=False,header=False)