#! /usr/bin/env python3
from keras_yolov2.utils import draw_boxes, enable_memory_growth
from keras_yolov2.frontend import YOLO
from keras_yolov2.utils import list_images
from tqdm import tqdm
import numpy as np
import argparse
import json
import cv2
import os
import csv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    '-r',
    '--real_time',
    default=False,
    type=bool,
    help='use a camera for real time prediction')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

argparser.add_argument(
    '-o',
    '--output',
    default='img',
    help='Output format, img (default) or csv')
    
    
def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    image_path = args.input
    use_camera = args.real_time
    output_format = args.output

    videos_format = [".mp4", "avi"]
    enable_memory_growth()

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    if weights_path == '':
        weights_path = config['train']['pretrained_weights"']

    ###################
    #   Make the model 
    ###################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=(config['model']['input_size_h'], config['model']['input_size_w']),
                labels=config['model']['labels'],
                anchors=config['model']['anchors'],
                gray_mode=config['model']['gray_mode'])

    #########################
    #   Load trained weights
    #########################

    yolo.load_weights(weights_path)

    ###########################
    #   Predict bounding boxes
    ###########################

    if use_camera:
        video_reader = cv2.VideoCapture(int(image_path))
        pbar = tqdm()
        while True:
            pbar.update(1)
            ret, frame = video_reader.read()
            if not ret:
                break
            boxes = yolo.predict(frame)
            frame = draw_boxes(frame, boxes, config['model']['labels'])
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:
                break
        pbar.close()
    elif os.path.splitext(image_path)[1] in videos_format:
        file, ext = os.path.splitext(image_path)
        video_out = '{}_detected.avi'.format(file)
        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(video_out)
        video_writer = cv2.VideoWriter(video_out,
                                       cv2.VideoWriter_fourcc(*'XVID'),
                                       50.0,
                                       (frame_w, frame_h))

        for _ in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            boxes = yolo.predict(image,
                                 iou_threshold=config['valid']['iou_threshold'],
                                 score_threshold=config['valid']['score_threshold'])

            image = draw_boxes(image, boxes, config['model']['labels'])
            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()
    else:
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            boxes = yolo.predict(image,
                                 iou_threshold=config['valid']['iou_threshold'],
                                 score_threshold=config['valid']['score_threshold'])
            image = draw_boxes(image, boxes, config['model']['labels'])

            print(len(boxes), 'boxes are found')
            cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
        else:
            if output_format == 'img':
                detected_images_path = os.path.join(image_path, "detected")
                if not os.path.exists(detected_images_path):
                    os.mkdir(detected_images_path)
            elif output_format == 'csv':
                header = ['File_path', 'xmin', 'ymin', 'xmax', 'ymax', 'presence']
                detected_csv = os.path.join(image_path, "detected.csv")
                # open csv file in the write mode
                f = open(detected_csv, 'w')
                writer = csv.writer(f)
                writer.writerow(header)

            images = list(list_images(image_path))
            for fname in tqdm(images):
                image = cv2.imread(fname)
                boxes = yolo.predict(image)

                if output_format == 'img':
                    image = draw_boxes(image, boxes, config['model']['labels'])
                    fname = os.path.basename(fname)
                    cv2.imwrite(os.path.join(image_path, "detected", fname), image)
                elif output_format == 'csv':
                    for box in boxes:
                        row = [fname, box.xmin, box.ymin, box.xmax, box.ymax, box.score]
                        writer.writerow(row)
                    
            if output_format == 'csv':
                f.close()




if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)
