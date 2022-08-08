import os
import pickle
import json
import cv2

from keras_yolov2.utils import draw_boxes

# Path to evaluation hisotry
pickle_path = "data/pickles/MobileNet_2022-08-08-14:16:22_0/bad_boxes_MobileNet_input_test.p"

# Open pickle
with open(pickle_path, 'rb') as fp:
    img_boxes = pickle.load(fp)

# Path to config filed use to evaluate
config_path = "config/pre_config/ADAM_OCS_v2_full_sampling_iNat_long.json"

# Open config file as a dict
with open(config_path) as config_buffer:
    config = json.load(config_buffer)

# Make sure the output path exists
if not os.path.exists(config["data"]["base_path"] + '/badpreds/'):
    os.makedirs(config["data"]["base_path"] + '/badpreds/')

# Draw predicted boxes and save
for img in img_boxes:
    img_path = config["data"]["base_path"] + '/' + img
    frame = cv2.imread(img_path)
    frame = draw_boxes(frame, img_boxes[img], config['model']['labels'])
    cv2.imwrite(config["data"]["base_path"] + '/badpreds/' + str.replace(img, '/', '_'), frame)
