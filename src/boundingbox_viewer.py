from keras_yolov2.preprocessing import parse_annotation_xml, parse_annotation_csv
from keras_yolov2.utils import import_feature_extractor
import argparse
import json
import cv2

argparser = argparse.ArgumentParser()

argparser.add_argument(
    '-c',
    '--conf',
    default='config/config.json',
    help='Path to configuration file')


def main(args):
    config_path = args.conf

    # Load config file as a dict
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # Load train images
    train_imgs, _ = parse_annotation_csv(config['data']['train_csv_file'],
                                        config['model']['labels'],
                                        config['data']['base_path'])
    
    print('Total image count:', len(train_imgs))

    # Main loop
    id = 0
    running = True
    while running:
        # Extract image path and boxes
        train_img = train_imgs[id]
        image = cv2.imread(train_img['filename'])
        bboxs = train_img['object']

        # Draw the boxes
        for bbox in bboxs:
            image = cv2.rectangle(image, (int(bbox['xmin']), int(bbox['ymin'])), (int(bbox['xmax']), int(bbox['ymax'])), (0, 255, 0), 5)

        # Plot the image
        cv2.imshow('Boundingbox viewer', cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3)))

        # Wait for key to continue
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:
            running = False
        elif key == 82: # up arrow
            id += 1
        elif key == 84: # down arrow
            id -= 1
        elif key == 81 or key == 83: # left/right arrow
            # Print image infos
            print(id, train_img['filename'])


if __name__ == "__main__":
    _args = argparser.parse_args()
    main(_args)
