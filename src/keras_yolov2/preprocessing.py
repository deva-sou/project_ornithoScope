import copy
from multiprocessing.connection import wait
import os
import xml.etree.ElementTree as et

import cv2
from cv2 import resize
import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from perlin_noise import PerlinNoise

from .utils import BoundBox, bbox_iou, draw_boxes
from bbaug.policies import policies
from bbaug.augmentations import augmentations
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def parse_annotation_xml(ann_dir, img_dir, labels=[]):
    # This parser is used on VOC dataset
    all_imgs = []
    seen_labels = {}
    ann_files = os.listdir(ann_dir)
    for ann in tqdm(sorted(ann_files)):
        img = {'object': []}

        tree = et.parse(os.path.join(ann_dir, ann))

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
                                

                    if 'attributes' in attr.tag:
                        for attribute in list(attr):
                            a = list(attribute)
                            if a[0].text == 'species':
                                obj['name'] = a[1].text

                                if obj['name'] in seen_labels:
                                    seen_labels[obj['name']] += 1
                                else:
                                    seen_labels[obj['name']] = 1

                                if len(labels) > 0 and obj['name'] not in labels:
                                    break
                                else:
                                    img['object'] += [obj]
                                
        all_imgs += [img]

    return all_imgs, seen_labels


def parse_annotation_csv(csv_file, labels=[], base_path=""):
    # This is a generic parser that uses CSV files
    # File_path,xmin,ymin,xmax,ymax,class

    all_imgs = []
    seen_labels = {}

    all_imgs_indices = {}
    count_indice = 0
    with open(csv_file, "r") as annotations:
        annotations = annotations.read().split("\n")
        for i, line in enumerate(tqdm(annotations)):
            if line == "":
                continue
            try:
                fname, xmin, ymin, xmax, ymax, obj_name, width, height = line.strip().split(",")
                fname = os.path.join(base_path, fname)

                img = dict()
                img['object'] = []
                img['filename'] = fname
                img['width'] = width
                img['height'] = height

                # If the object has no name, this means that this image is a background image
                if obj_name == "":
                    all_imgs_indices[fname] = count_indice
                    all_imgs.append(img)
                    count_indice += 1
                    continue

                obj = dict()
                obj['xmin'] = float(xmin)
                obj['xmax'] = float(xmax)
                obj['ymin'] = float(ymin)
                obj['ymax'] = float(ymax)
                obj['name'] = obj_name

                if len(labels) > 0 and obj_name not in labels:
                    continue
                else:
                    img['object'].append(obj)

                if fname not in all_imgs_indices:
                    all_imgs_indices[fname] = count_indice
                    all_imgs.append(img)
                    count_indice += 1
                else:
                    all_imgs[all_imgs_indices[fname]]['object'].append(obj)

                if obj_name not in seen_labels:
                    seen_labels[obj_name] = 1
                else:
                    seen_labels[obj_name] += 1

            except:
                print("Exception occured at line {} from {}".format(i + 1, csv_file))
                raise
    return all_imgs, seen_labels


def resize_bbox(bbox, initial_size, final_size):
    new_bbox = bbox.copy()
    new_bbox['xmin'] *= final_size[0] / initial_size[0]
    new_bbox['xmax'] *= final_size[0] / initial_size[0]
    new_bbox['ymin'] *= final_size[1] / initial_size[1]
    new_bbox['ymax'] *= final_size[1] / initial_size[1]
    return new_bbox


class CustomPolicy(policies.PolicyContainer):
    """
    Custom augmentation policy.
    """

    def __init__(self, images, config):
        self._config = config

        # List all augmentations
        name_to_augmentation = augmentations.NAME_TO_AUGMENTATION.copy()
        name_to_augmentation.update({
                'PerlinShadows': self.shadows_augmentation,
            })
        super().__init__(None, name_to_augmentation=name_to_augmentation)

        # Extract all image paths and all annotations
        self.all_path = []
        self.all_bboxs = []
        for image in images:
            path = image['filename']
            bboxs = image['object']

            self.all_path.append(path)
            self.all_bboxs.append(bboxs)

        # Create perlin noise mask
        noise = PerlinNoise(octaves=80, seed=np.random.randint(1e8))
        mask_w, mask_h = 100, 100
        print('Creating shadow mask...', end='\r')
        self.shadow = np.array([[noise([i / mask_h, j / mask_w]) for j in range(mask_w)] for i in range(mask_h)])
        print('                       ', end='\r')
    
    def select_random_policy(self):
        return [
                policies.POLICY_TUPLE('PerlinShadows', 0.3, 8),
                # policies.POLICY_TUPLE('Brightness', 0.2, 1),
                # policies.POLICY_TUPLE('Cutout', 0.2, 6),
                # policies.POLICY_TUPLE('Cutout_BBox', 1.0, 2),
                # policies.POLICY_TUPLE('Color', 0.2, 1),
                # policies.POLICY_TUPLE('Fliplr_BBox', 0.2, 3),
                # policies.POLICY_TUPLE('Rotate', 0.2, 3),
                # policies.POLICY_TUPLE('Solarize', 0.2, 1),
                # policies.POLICY_TUPLE('Translate_X', 0.2, 3),
                # policies.POLICY_TUPLE('Translate_X_BBox', 0.2, 3),
                # policies.POLICY_TUPLE('Translate_Y', 0.2, 3),
                # policies.POLICY_TUPLE('Translate_Y_BBox', 0.2, 3),                
            ]

    
    def shadows_augmentation(self, magnitude: int):
        """
        Create callable augmentation.
        """
        def aug(image, bounding_boxes):
            return self.PerlinShadows(image, amplitude=10 * magnitude, offset=0), bounding_boxes
        return aug


    def PerlinShadows(self, image, amplitude=80, offset=0):
        """
        Add perlin noise brightness mask.
        """
        h, w = image.shape[:2]

        # Select perlin noise mask area
        mask_w, mask_h = w // 20, h // 20
        full_mask_w, full_mask_h = self.shadow.shape
        x_pos, y_pos = np.random.randint(full_mask_w - mask_w), np.random.randint(full_mask_h - mask_h)
        shadow = self.shadow[x_pos:x_pos + mask_w, y_pos:y_pos + mask_h]

        # Set mask values between 0 and 255
        shadow = shadow - np.min(shadow, (0, 1))
        shadow = shadow / np.max(shadow, (0, 1))
        shadow = CustomPolicy.cosine_contraste_augmentation(shadow) * 255.0
        
        # Resize mask to image size
        shadow = shadow.astype('uint8')
        shadow = cv2.resize(shadow, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        shadow = shadow.astype('float') / 127.0 - 1.0

        # Convert RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = v.astype('float')

        # Recast mask values
        shadow = amplitude * shadow + offset

        # Apply shadow mask on brightness
        v += shadow
        v[v > 255.0] = 255.0
        v[v < 0.0] = 0.0

        # Convert back HSV to RGB
        v = v.astype('uint8')
        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        return image

    def cosine_contraste_augmentation(x: np.ndarray):
        """
        x, array of float between 0.0 and 1.0
        return array of float between 0.0 and 1.0 closer to limits.
        """
        return (-np.cos(np.pi * x) + 1) / 2


def create_mosaic(imgs, all_bbs, labels, output_size, scale_range, filter_scale=0.0):
    output_image = np.zeros(output_size, dtype=np.uint8)
    scale_x = scale_range[0] + np.random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + np.random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    new_bboxs = []
    for i, (img, bboxs) in enumerate(zip(imgs, all_bbs)):
        
        if i == 0:  # top-left
            initial_size = img.shape[-2::-1]
            final_size = (divid_point_x, divid_point_y)
            img = cv2.resize(img, final_size)
            output_image[:divid_point_y, :divid_point_x, :] = img
            for bbox in bboxs:
                bbox = resize_bbox(bbox, initial_size, final_size)
                new_bboxs.append(bbox)

        elif i == 1:  # top-right
            initial_size = img.shape[-2::-1]
            final_size = (output_size[1] - divid_point_x, divid_point_y)
            img = cv2.resize(img, final_size)
            output_image[:divid_point_y, divid_point_x:, :] = img
            for bbox in bboxs:
                bbox = resize_bbox(bbox, initial_size, final_size)
                bbox['xmin'] += divid_point_x
                bbox['xmax'] += divid_point_x
                new_bboxs.append(bbox)

        elif i == 2:  # bottom-left
            initial_size = img.shape[-2::-1]
            final_size = (divid_point_x, output_size[0] - divid_point_y)
            img = cv2.resize(img, final_size)
            output_image[divid_point_y:, :divid_point_x, :] = img
            for bbox in bboxs:
                bbox = resize_bbox(bbox, initial_size, final_size)
                bbox['ymin'] += divid_point_y
                bbox['ymax'] += divid_point_y
                new_bboxs.append(bbox)

        else:  # bottom-right
            initial_size = img.shape[-2::-1]
            final_size = (output_size[1] - divid_point_x, output_size[0] - divid_point_y)
            img = cv2.resize(img, final_size)
            output_image[divid_point_y:, divid_point_x:, :] = img
            for bbox in bboxs:
                bbox = resize_bbox(bbox, initial_size, final_size)
                bbox['xmin'] += divid_point_x
                bbox['ymin'] += divid_point_y
                bbox['xmax'] += divid_point_x
                bbox['ymax'] += divid_point_y
                new_bboxs.append(bbox)

    # if 0 < filter_scale:
    #     new_bboxs = [anno for anno in new_bboxs if
    #                 filter_scale < (anno[3] - anno[1]) and filter_scale < (anno[4] - anno[2])]
    
    return output_image, new_bboxs


class BatchGenerator(Sequence):
    def __init__(self, images, config, shuffle=True, sampling=False, jitter=True, norm=None, policy_container='none'):

        self._raw_images = images

        self._config = config

        self._shuffle = shuffle
        self._sampling = sampling
        self._jitter = jitter
        self._norm = norm
        self._policy_container = policy_container

        self._anchors = [BoundBox(0, 0, config['ANCHORS'][2 * i], config['ANCHORS'][2 * i + 1])
                         for i in range(int(len(config['ANCHORS']) // 2))]

        self._policy_chosen = self.get_policy_container()
        
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(float(len(self._images)) / self._config['IMG_PER_BATCH']))

    def size(self):
        return len(self._images)

    def load_annotation(self, i):
        annots = []

        for obj in self._images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self._config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        if self._config['IMAGE_C'] == 1:
            image = cv2.imread(self._images[i]['filename'], cv2.IMREAD_GRAYSCALE)
            image = image[..., np.newaxis]
        elif self._config['IMAGE_C'] == 3:
            image = cv2.imread(self._images[i]['filename'])
        else:
            raise ValueError("Invalid number of image channels.")
        return image, '/'.join(self._images[i]['filename'].split('/')[-2:])

    def get_policy_container(self):
        data_aug_policies = {
            'v0' : policies.PolicyContainer(policies.policies_v0()),
            'v1' : policies.PolicyContainer(policies.policies_v1()),
            'v2' : policies.PolicyContainer(policies.policies_v2()),
            'v3' : policies.PolicyContainer(policies.policies_v3())
        }

        policy_chosen = self._policy_container.lower()
        if policy_chosen in data_aug_policies:
            return data_aug_policies.get(policy_chosen)
       
        elif policy_chosen == 'custom':
            return CustomPolicy(self._images, self._config)

        elif policy_chosen == 'none':
            self._jitter = False
            return None

        else: 
            print("Wrong policy for data augmentation")
            print('Choose beetween:\n')
            print(list(data_aug_policies.keys()))
            exit(1)

    def __getitem__(self, idx):
        # Set lower an upper id for this batch
        l_bound = idx * self._config['IMG_PER_BATCH']
        r_bound = (idx + 1) * self._config['IMG_PER_BATCH']

        # Fix upper bound grate than number of image
        if r_bound > len(self._images):
            r_bound = len(self._images)
            l_bound = r_bound - self._config['IMG_PER_BATCH']

        # Initialize batch's input and output
        x_batch = np.zeros((self._config['BATCH_SIZE'], self._config['IMAGE_H'], self._config['IMAGE_W'],
                            self._config['IMAGE_C']))
        y_batch = np.zeros((self._config['BATCH_SIZE'], self._config['GRID_H'], self._config['GRID_W'], self._config['BOX'],
                            4 + 1 + len(self._config['LABELS'])))


        anchors_populated_map = np.zeros((self._config['BATCH_SIZE'], self._config['GRID_H'], self._config['GRID_W'],
                                          self._config['BOX']))

        for instance_count in range(self._config['BATCH_SIZE']):
            if self._config['MOSAIC'] == 'none':
                # Augment input image and bounding boxes' attributes
                img, all_bbs = self.aug_image(l_bound + instance_count)
            else:
                imgs = []
                all_bbs = []
                for i in range(4):
                    # Augment input image and bounding boxes' attributes
                    img, bbs = self.aug_image(l_bound + 4 * instance_count + i)
                    imgs.append(img)
                    all_bbs.append(bbs)

                # Merge images to create mosaic
                img, all_bbs = create_mosaic(
                        imgs=imgs,
                        all_bbs=all_bbs,
                        labels=self._config['LABELS'],
                        output_size=(self._config['IMAGE_W'], self._config['IMAGE_H'], 3),
                        scale_range=(0.3, 0.7),
                        filter_scale=0.0
                    )
                

            for bb in all_bbs:
                # Check if it is a valid boudning box
                if bb['xmax'] <= bb['xmin'] or bb['ymax'] <= bb['ymin'] or not bb['name'] in self._config['LABELS']:
                    continue

                
                scale_w = float(self._config['IMAGE_W']) / self._config['GRID_W']
                scale_h = float(self._config['IMAGE_H']) / self._config['GRID_H']

                # get which grid cell it is from
                obj_center_x = (bb['xmin'] + bb['xmax']) / 2
                obj_center_x = obj_center_x / scale_w
                obj_center_y = (bb['ymin'] + bb['ymax']) / 2
                obj_center_y = obj_center_y / scale_h

                obj_grid_x = int(np.floor(obj_center_x))
                obj_grid_y = int(np.floor(obj_center_y))

                if obj_grid_x < self._config['GRID_W'] and obj_grid_y < self._config['GRID_H']:
                    obj_indx = self._config['LABELS'].index(bb['name'])

                    obj_w = (bb['xmax'] - bb['xmin']) / scale_w
                    obj_h = (bb['ymax'] - bb['ymin']) / scale_h

                    box = [obj_center_x, obj_center_y, obj_w, obj_h]

                    # find the anchor that best predicts this box
                    best_anchor_idx = -1
                    max_iou = -1

                    shifted_box = BoundBox(0, 0, obj_w, obj_h)

                    for i in range(len(self._anchors)):
                        anchor = self._anchors[i]
                        iou = bbox_iou(shifted_box, anchor)

                        if max_iou < iou:
                            best_anchor_idx = i
                            max_iou = iou

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    self._change_obj_position(y_batch, anchors_populated_map,
                                                [instance_count, obj_grid_y, obj_grid_x, best_anchor_idx, obj_indx],
                                                box, max_iou)

            # assign input image to x_batch
            if self._norm is not None:
                x_batch[instance_count] = self._norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for bb in all_bbs:
                    if bb['xmax'] > bb['xmin'] and bb['ymax'] > bb['ymin']:
                        cv2.rectangle(img[..., ::-1], (bb['xmin'], bb['ymin']), (bb['xmax'], bb['ymax']),
                                      (255, 0, 0), 3)
                        cv2.putText(img[..., ::-1], bb['name'], (bb['xmin'] + 2, bb['ymin'] + 12), 0,
                                    1.2e-3 * img.shape[0], (0, 255, 0), 2)

                x_batch[instance_count] = img

        return x_batch, y_batch

    def _change_obj_position(self, y_batch, anchors_map, idx, box, iou):

        bkp_box = y_batch[idx[0], idx[1], idx[2], idx[3], 0:4].copy()
        anchors_map[idx[0], idx[1], idx[2], idx[3]] = iou
        y_batch[idx[0], idx[1], idx[2], idx[3], 0:4] = box
        y_batch[idx[0], idx[1], idx[2], idx[3], 4] = 1.
        y_batch[idx[0], idx[1], idx[2], idx[3], 5:] = 0  # clear old values
        y_batch[idx[0], idx[1], idx[2], idx[3], 4 + 1 + idx[4]] = 1

        shifted_box = BoundBox(0, 0, bkp_box[2], bkp_box[3])

        for i in range(len(self._anchors)):
            anchor = self._anchors[i]
            iou = bbox_iou(shifted_box, anchor)
            if iou > anchors_map[idx[0], idx[1], idx[2], i]:
                self._change_obj_position(y_batch, anchors_map, [idx[0], idx[1], idx[2], i, idx[4]], bkp_box, iou)
                break

    def on_epoch_end(self):
        # Shuffle raw images
        if self._shuffle:
            np.random.shuffle(self._raw_images)
        
        ## Use raw images as image set
        if not self._sampling:
            self._images = self._raw_images
            return

        ## Create image set using sampling
        self._images = []

        cap = 200

        # Initialize counter
        counter = {label: 0 for label in self._config['LABELS']}

        # Group images per species
        image_per_specie = {label: [] for label in self._config['LABELS']}
        for image in self._raw_images:
            for box in image['object']:
                image_per_specie[box['name']].append(image)
        
        # Shuffle a bit
        for image_list in image_per_specie.values():
            np.random.shuffle(image_list)
        
        # Loop to complete each species from the rarest
        counter_min_key = min(counter, key=counter.get)
        counter_min = counter[counter_min_key]
        while counter_min < cap:
            # Take the first picture and replace it in the queue
            header_image = image_per_specie[counter_min_key].pop(0)
            image_per_specie[counter_min_key].append(header_image)

            # Add current image to the image set
            self._images.append(header_image)

            # Increment counters
            for box in header_image['object']:
                counter[box['name']] += 1

            # Update rarest specie
            counter_min_key = min(counter, key=counter.get)
            counter_min = counter[counter_min_key]
        
        print(counter)
            


    def aug_image(self, idx):
        train_instance = self._images[idx]
        image_name = train_instance['filename']
        if self._config['IMAGE_C'] == 1:
            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        elif self._config['IMAGE_C'] == 3:
            image = cv2.imread(image_name)
        else:
            raise ValueError("Invalid number of image channels.")

        if image is None:
            raise Exception('Cannot find : ' + image_name)

        h, w = image.shape[:2]
        all_objs = copy.deepcopy(train_instance['object'])

        # Apply augmentation
        if self._jitter:
            bbs = []
            labels_bbs = []

            # Convert bouding boxes for the PolicyConatiner
            for obj in all_objs:
                xmin = obj['xmin']
                ymin = obj['ymin']
                xmax = obj['xmax']
                ymax = obj['ymax']

                bbs.append([xmin, ymin, xmax, ymax])
                labels_bbs.append(self._config['LABELS'].index(obj['name']))

            random_policy = self._policy_chosen.select_random_policy()
            image, bbs = self._policy_chosen.apply_augmentation(random_policy, image, bbs, labels_bbs)
            
            # Recreate bounding boxes
            all_objs = []
            for bb in bbs:
                obj = {}
                obj['xmin'] = bb[1]
                obj['xmax'] = bb[3]
                obj['ymin'] = bb[2]
                obj['ymax'] = bb[4]
                obj['name'] = self._config['LABELS'][bb[0]]
                all_objs.append(obj)


        # Resize the image to standard size
        image = cv2.resize(image, (self._config['IMAGE_W'], self._config['IMAGE_H']))
        if self._config['IMAGE_C'] == 1:
            image = image[..., np.newaxis]
        image = image[..., ::-1]  # make it RGB (it is important for normalization of some backends)

        # Fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                obj[attr] = int(obj[attr] * float(self._config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self._config['IMAGE_W']), 0)

            for attr in ['ymin', 'ymax']:
                obj[attr] = int(obj[attr] * float(self._config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self._config['IMAGE_H']), 0)

        return image, all_objs
