import json
import cv2
import time
import datetime
import argparse
import os
import csv
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from keras_yolov2.frontend import YOLO
from keras_yolov2.utils import draw_boxes, list_images
from keras_yolov2.tracker import NMS, BoxTracker


argparser = argparse.ArgumentParser(
  description='Predict tflite model in real time or with movies / images')

argparser.add_argument(
  '-c',
  '--conf',
  default='config/config_lab_mobilenetV1.json',
  type=str,
  help='path to configuration file')

argparser.add_argument(
  '-w',
  '--weights',
  default='',
  type=str,
  help='path to pretrained weights')

argparser.add_argument(
  '-l',
  '--lite',
  default='',
  type=str,
  help='Path to tflite model')

argparser.add_argument(
  '-r',
  '--real_time',
  default=False,
  type=bool,
  help='use a camera for real time prediction')

argparser.add_argument(
  '-i',
  '--input',
  type=str,
  help='path to an image or an video (mp4 format)')

argparser.add_argument(
  '-o',
  '--output',
  default='img',
  type=str,
  help='Output format, img (default) or csv')


def _main_(args):
  config_path = args.conf
  weights_path = args.weights
  image_path = args.input
  use_camera = args.real_time
  lite_path = args.lite
  output_format = args.output

  videos_format = ['.mp4', 'avi']

  # Load config file
  with open(config_path) as config_buffer:
    config = json.load(config_buffer)

  # Set weights path
  if weights_path == '':
    weights_path = config['train']['pretrained_weights']

  # Create model
  yolo = YOLO(backend=config['model']['backend'],
              input_size=(config['model']['input_size_h'], config['model']['input_size_w']),
              labels=config['model']['labels'],
              anchors=config['model']['anchors'],
              gray_mode=config['model']['gray_mode'])

  # Load weights
  yolo.load_weights(weights_path)

  # Use tflite
  if lite_path != '':
    yolo.load_lite(lite_path)

  # Tracker
  BT = BoxTracker()

  ### Real Time
  if use_camera:

    # Set video capture
    video_reader = cv2.VideoCapture(int(image_path))

    # Variables to calculate FPS
    start_time = time.time()
    counter, fps = 0, 0
    fps_avg_frame_count = 10

    # Main loop
    running = True
    while running:
      counter += 1

      # Read video
      ret, frame = video_reader.read()
      if not ret:
        running = False
        continue

      # Predict
      boxes = yolo.predict(frame,
                            iou_threshold=config['valid']['iou_threshold'],
                            score_threshold=config['valid']['score_threshold'])

      # Decode and draw boxes
      boxes = NMS(boxes)
      boxes = BT.update(boxes).values()
      
      # Draw boxes
      frame = draw_boxes(frame, boxes, config['model']['labels'])

      # Show the date
      current_time = str(datetime.datetime.utcfromtimestamp(int(time.time())))
      cv2.putText(frame, current_time, (20, 20), cv2.FONT_HERSHEY_PLAIN,
          1, (0, 255, 0), 2)

      # Calculate the FPS
      if counter % fps_avg_frame_count == 0:
          end_time = time.time()
          fps = fps_avg_frame_count / (end_time - start_time)
          start_time = time.time()
      
      # Show the FPS
      fps_text = '{:02.1f} fps'.format(fps)
      cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_PLAIN,
          1, (0, 255, 0), 2)

      # Show frame
      cv2.imshow("frame", frame)

      # Quit
      key = cv2.waitKey(1)
      if key == ord("q") or key == 27:
        running = False
  
  ### Video
  elif os.path.splitext(image_path)[1] in videos_format:
    file, ext = os.path.splitext(image_path)
    video_out = '{}_detected.avi'.format(file)

    video_reader = cv2.VideoCapture(image_path)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    video_writer = cv2.VideoWriter(video_out,
                                    cv2.VideoWriter_fourcc(*'XVID'),
                                    15.0,
                                    (frame_w, frame_h))

    for _ in tqdm(range(nb_frames)):
      # Read video
      ret, frame = video_reader.read()
      if not ret:
        running = False
        continue

      # Predict
      boxes = yolo.predict(frame,
                            iou_threshold=config['valid']['iou_threshold'],
                            score_threshold=config['valid']['score_threshold'])

      # Decode and draw boxes
      boxes = NMS(boxes)
      boxes = BT.update(boxes).values()
      frame = draw_boxes(frame, boxes, config['model']['labels'])

      # Write video output
      video_writer.write(np.uint8(frame))

    video_reader.release()
    video_writer.release()
    
    # History infos
    res = {}
    for id, history in zip(BT.tracker_history.keys(), BT.tracker_history.values()):
      res[id] = {}
      for class_id, trust in history:
        class_label = config['model']['labels'][class_id]
        if class_label in res[id]:
          res[id][class_label] += trust
        else:
          res[id][class_label] = trust
    print(res)

  ### Image
  else:
    # One image
    if os.path.isfile(image_path):
      # Open image
      frame = cv2.imread(image_path)

      # Predict
      boxes = yolo.predict(frame)

      # Draw boxes
      frame = draw_boxes(frame, boxes, config['model']['labels'])

      # Write image output
      cv2.imwrite(image_path[:-4] + '_lite_detected' + image_path[-4:], frame)

    # Image folder
    else:
      if output_format == 'img':
        # Create output image file/folder
        detected_images_path = os.path.join(image_path, "detected")
        if not os.path.exists(detected_images_path):
          os.mkdir(detected_images_path)

      elif output_format.startswith('csv'):
        # Create output csv
        detected_csv = os.path.join(image_path, "detected.csv")
        f = open(detected_csv, 'w')
        writer = csv.writer(f)

      images = list(list_images(image_path))
      for fname in tqdm(images):
        frame = cv2.imread(fname)

        # Predict
        boxes = yolo.predict(frame,
                              iou_threshold=config['valid']['iou_threshold'],
                              score_threshold=config['valid']['score_threshold'])

        if output_format == 'img':
          image = draw_boxes(frame, boxes, config['model']['labels'])
          fname = os.path.basename(fname)
          cv2.imwrite(os.path.join(image_path, "detected", fname), image)
        elif output_format == 'csv':
          for box in boxes:
            row = [fname, box.xmin, box.ymin, box.xmax, box.ymax, box.score]
            writer.writerow(row)
        elif output_format == 'csv_input':
          image_h, image_w, _ = frame.shape
          for box in boxes:
            row = [
                fname,
                int(box.xmin * image_w),
                int(box.ymin * image_h),
                int(box.xmax * image_w),
                int(box.ymax * image_h),
                config['model']['labels'][box.get_label()],
                image_w,
                image_h
              ]
            writer.writerow(row)
              
      if output_format.startswith('csv'):
        f.close()
  

if __name__ == '__main__':
  _args = argparser.parse_args()
  gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')
  with tf.device('/GPU:' + gpu_id):
    _main_(_args)