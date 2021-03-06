import pickle
import json
import cv2

import numpy as np

from keras_yolov2.utils import draw_boxes

pickle_path = "data/pickles/MobileNet_2022-08-01-15:36:48/boxes_MobileNet_task_20210705-07_balacet.p"

with open(pickle_path, 'rb') as fp:
    img_boxes = pickle.load(fp)

config_path = "config/config_to_train/ADAM_lr4.json"

with open(config_path) as config_buffer:
    config = json.load(config_buffer)

video_writer = cv2.VideoWriter("video_out.avi",
                                    cv2.VideoWriter_fourcc(*'XVID'),
                                    1.0,
                                    (800, 600))

for img in img_boxes:
    img_path = config["data"]["base_path"] + '/' + img
    frame = cv2.imread(img_path)
    frame = draw_boxes(frame, img_boxes[img], config['model']['labels'])
    video_writer.write(np.uint8(cv2.resize(frame, (800, 600))))

    # cv2.imshow('Prediction', cv2.resize(frame, (800, 600)))
    # key = cv2.waitKey(0)
    # if key == 27 or key == ord('q'):
    #     break
video_writer.release()