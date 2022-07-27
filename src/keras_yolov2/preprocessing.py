import copy
from email import policy
import os
from time import time
import xml.etree.ElementTree as et

import cv2
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import sigmoid
from tqdm import tqdm
from perlin_noise import PerlinNoise

from .utils import BoundBox, bbox_iou
from bbaug.policies import policies


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


class PersonalPolicy():
    """
    Personal augmentation policy.
    """

    def __init__(self):
        # Create perlin noise mask
        noise = PerlinNoise(octaves=80, seed=np.random.randint(1e8))
        mask_w, mask_h = 1000, 1000
        print('Creating shadow mask...')
        self.shadow = np.array([[noise([i / mask_h, j / mask_w]) for j in range(mask_w)] for i in range(mask_h)])
        print('', end='\r')

    def shadows_augmentation(self, image, amplitude=80, offset=0):
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
        shadow = PersonalPolicy.cosine_contraste_augmentation(shadow) * 255.0
        
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


    def modify_brightness(image: np.ndarray, value: int = 30):
        """
        Add brightness value to an image
        """

        # Convert RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Add brightness
        if value > 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        # Remove brightness
        else:
            v[v < -value] = 0
            v[v >= -value] += value

        # Convert back HSV to RGB
        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        return image


class BatchGenerator(Sequence):
    def __init__(self, images, config, shuffle=True, jitter=True, norm=None, policy_container='none'):

        self._images = images
        self._config = config

        self._shuffle = shuffle
        self._jitter = jitter
        self._norm = norm
        self._policy_container = policy_container

        self._anchors = [BoundBox(0, 0, config['ANCHORS'][2 * i], config['ANCHORS'][2 * i + 1])
                         for i in range(int(len(config['ANCHORS']) // 2))]

        self._policy_chosen = self.get_policy_container()
        
        if shuffle:
            np.random.shuffle(self._images)
    
    def __len__(self):
        return int(np.ceil(float(len(self._images)) / self._config['BATCH_SIZE']))

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
        
        elif policy_chosen == 'personal':
            return PersonalPolicy()

        elif policy_chosen == 'none':
            self._jitter = False
            return None

        else: 
            print("Wrong policy for data augmentation")
            print('Choose beetween:\n')
            print(list(data_aug_policies.keys()))
            exit(1)
    
    def num_classes(self):
        return len(self._config['LABELS'])

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

    def __getitem__(self, idx):
        # Set lower an upper id for this batch
        l_bound = idx * self._config['BATCH_SIZE']
        r_bound = (idx + 1) * self._config['BATCH_SIZE']

        # Fix upper bound grate than number of image
        if r_bound > len(self._images):
            r_bound = len(self._images)
            l_bound = r_bound - self._config['BATCH_SIZE']

        # Initialize batch's input and output
        x_batch = np.zeros((r_bound - l_bound, self._config['IMAGE_H'], self._config['IMAGE_W'],
                            self._config['IMAGE_C']))
        y_batch = np.zeros((r_bound - l_bound, self._config['GRID_H'], self._config['GRID_W'], self._config['BOX'],
                            4 + 1 + len(self._config['LABELS'])))


        anchors_populated_map = np.zeros((r_bound - l_bound, self._config['GRID_H'], self._config['GRID_W'],
                                          self._config['BOX']))

        for instance_count, train_instance in enumerate(self._images[l_bound:r_bound]):
            # Augment input image and bounding boxes' attributes
            img, all_bbs = self.aug_image(train_instance)

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
                    # TODO: check f this part below is working correctly
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
        if self._shuffle:
            np.random.shuffle(self._images)

    def aug_image(self, train_instance):
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

            # cv2.imshow('Before augmentation', cv2.resize(image, (w // 3, h // 3)))

            # Use Google Brain Team augmentation
            if isinstance(self._policy_chosen, policies.PolicyContainer):
                random_policy = self._policy_chosen.select_random_policy()
                image, bbs = self._policy_chosen.apply_augmentation(random_policy, image, bbs, labels_bbs)
            
            # Use personal augmentation
            elif isinstance(self._policy_chosen, PersonalPolicy):
                image = self._policy_chosen.shadows_augmentation(image,
                                                                amplitude=np.random.randint(50, 100),
                                                                offset=np.random.randint(-10, 10))
                # Add labels in first position of bounding boxes
                for box, label in zip(bbs, labels_bbs):
                    box.insert(0, label)

            # cv2.imshow('After augmentation', cv2.resize(image, (w // 3, h // 3)))
            # cv2.waitKey(0)
            
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
    
