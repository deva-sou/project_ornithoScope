import argparse
import cv2
import argparse
import json
from tqdm import tqdm

from keras_yolov2.frontend import YOLO
from keras_yolov2.tracker import NMS, BoxTracker
from keras_yolov2.utils import draw_boxes

argparser = argparse.ArgumentParser(
  description='Predict tflite model in real time or with movies / images')

argparser.add_argument(
  '-l',
  '--lite',
  default=0,
  type=int,
  help='Model number (0:TF ; 1:TFL32 ; 2:TFL16)')

args = argparser.parse_args()

# Paths
config_path = 'config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json'
weights_path = 'data/saved_weights/data_aug_policies/MobileNet_caped300_data_aug_v0_bestLoss.h5'
lite_path32 = 'data/saved_lite/mobilenetV1_labels_caped300_data_augv0.tflite'
lite_path16 = 'data/saved_lite/mobilenetV1_labels_caped300_data_augv0_float16.tflite'
image_path = 'data/imgs/img_test/stress-test.jpg'


# Load config file
with open(config_path) as config_buffer:
    config = json.load(config_buffer)

# Create model
yolo = YOLO(backend=config['model']['backend'],
            input_size=(config['model']['input_size_h'], config['model']['input_size_w']),
            labels=config['model']['labels'],
            anchors=config['model']['anchors'],
            gray_mode=config['model']['gray_mode'])

# Load weights
yolo.load_weights(weights_path)

# Use tflite
if args.lite != 0:
    yolo.load_lite(lite_path32 if args.lite == 1 else lite_path16)

# Read image
frame = cv2.imread(image_path)

# Box tracker
BT = BoxTracker()

for _ in tqdm(range(1000)):
    # Predict
    boxes = yolo.predict(frame,
                        iou_threshold=config['valid']['iou_threshold'],
                        score_threshold=config['valid']['score_threshold'])

    # Decode and draw boxes
    boxes = NMS(boxes)
    boxes = BT.update(boxes).values()
    frame = draw_boxes(frame, boxes, config['model']['labels'])

    # Write frame
    cv2.imwrite('data/imgs/img_test/stess-test_detected.jpg', frame)