import copy
from email import policy
import os
import xml.etree.ElementTree as et

import cv2
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from .utils import BoundBox, bbox_iou
from bbaug.policies import policies

import random
import cv2
import os
import glob
import numpy as np
from PIL import Image



def parse_annotation_xml(ann_dir, img_dir, labels=[]):
    # This parser is utilized on VOC dataset
    all_imgs = []
    seen_labels = {}
    print(ann_dir)
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
                                

        #if len(img['object']) > 0:
        all_imgs += [img]

    return all_imgs, seen_labels


def parse_annotation_csv(csv_file, labels=[], base_path=""):
    # This is a generic parser that uses CSV files
    # File_path,xmin,ymin,xmax,ymax,class

    #print("parsing {} csv file can took a while, wait please.".format(csv_file))
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
                fname, xmin, ymin, xmax, ymax, obj_name, width, height = line.strip().split(",") # ajouter ici height and width
                #print('fname',fname,'obj name', obj_name)
                fname = os.path.join(base_path, fname)

                #image = cv2.imread(fname) # supprimer ça pour éviter la lecture de l'image
                #height, width, _ = image.shape # ça aussi

                img = dict()
                img['object'] = []
                img['filename'] = fname
                img['width'] = width
                img['height'] = height

                if obj_name == "":  # if the object has no name, this means that this image is a background image
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

        # self.policy_container = policies.PolicyContainer(policies.policies_v3())
        self._policy_chosen = self.get_policy_container()
        #print(self._jitter)
        if shuffle:
            np.random.shuffle(self._images)

    
    
    def __len__(self):
        return int(np.ceil(float(len(self._images)) / self._config['BATCH_SIZE']))


    #On doit écrire un script qui fait soit notre politique d'augmentation de données soit celle de Deva mais pas les deux en même temps

    #policy_chosen=policy_container=policy(dans le train_generator du frontend)=config['train']['augmentation'] (dans le train) 

    def get_policy_container(self): 
        data_aug_policies = {
            'v0':policies.PolicyContainer(policies.policies_v0()),
            'v1':policies.PolicyContainer(policies.policies_v1()),
            'v2':policies.PolicyContainer(policies.policies_v2()),
            'v3':policies.PolicyContainer(policies.policies_v3())
            }
        policy_chosen = self._policy_container.lower()
        print('\npolicy_chosen: ',policy_chosen)
        if policy_chosen in data_aug_policies:
            self._jitter='True'
            return data_aug_policies.get(policy_chosen) #.get permet d'obtenir une valeur d'un dictionnaire
        elif policy_chosen=='mosaic':
            self._jitter='mosaic'
            return None
        elif policy_chosen == 'none':
            self._jitter = 'False'
            return None
        else : 
            print("Wrong policy for data augmentation")
            print('Choose beetween:\n')
            print(list(data_aug_policies.keys()))
            exit(0)

    


    
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
        l_bound = idx * self._config['BATCH_SIZE']
        r_bound = (idx + 1) * self._config['BATCH_SIZE']

        if r_bound > len(self._images):
            r_bound = len(self._images)
            l_bound = r_bound - self._config['BATCH_SIZE']

        instance_count = 0
        x_batch = np.zeros((r_bound - l_bound, self._config['IMAGE_H'], self._config['IMAGE_W'],
                            self._config['IMAGE_C']))  # input images

        y_batch = np.zeros((r_bound - l_bound, self._config['GRID_H'], self._config['GRID_W'], self._config['BOX'],
                            4 + 1 + len(self._config['LABELS'])))  # desired network output

        anchors_populated_map = np.zeros((r_bound - l_bound, self._config['GRID_H'], self._config['GRID_W'],
                                          self._config['BOX']))


        for train_instance in self._images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance)

            # if len(all_objs) == 0:
            #    print("eeee")

            for obj in all_objs:
                # check if it is a valid annotion
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self._config['LABELS']:
                    scale_w = float(self._config['IMAGE_W']) / self._config['GRID_W']
                    scale_h = float(self._config['IMAGE_H']) / self._config['GRID_H']
                    # get which grid cell it is from
                    obj_center_x = (obj['xmin'] + obj['xmax']) / 2
                    obj_center_x = obj_center_x / scale_w
                    obj_center_y = (obj['ymin'] + obj['ymax']) / 2
                    obj_center_y = obj_center_y / scale_h

                    obj_grid_x = int(np.floor(obj_center_x))
                    obj_grid_y = int(np.floor(obj_center_y))

                    if obj_grid_x < self._config['GRID_W'] and obj_grid_y < self._config['GRID_H']:
                        obj_indx = self._config['LABELS'].index(obj['name'])

                        obj_w = (obj['xmax'] - obj['xmin']) / scale_w
                        obj_h = (obj['ymax'] - obj['ymin']) / scale_h

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
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[..., ::-1], (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']),
                                      (255, 0, 0), 3)
                        cv2.putText(img[..., ::-1], obj['name'], (obj['xmin'] + 2, obj['ymin'] + 12), 0,
                                    1.2e-3 * img.shape[0], (0, 255, 0), 2)

                x_batch[instance_count] = img
            # increase instance counter in current batch
            instance_count += 1

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

    def update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale=0.):
                output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
                scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
                scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
                divid_point_x = int(scale_x * output_size[1])
                divid_point_y = int(scale_y * output_size[0])

                new_anno = []

                for i, idx in enumerate(idxs):
        
                    path = all_img_list[idx]
                    img_annos = all_annos[idx]

                    img = cv2.imread(path)
                    
                    if i == 0: # top-left  
                        img = cv2.resize(img, (divid_point_x, divid_point_y))
                        output_img[:divid_point_y, :divid_point_x, :] = img
                        for bbox in img_annos:
                            xmin = bbox[1] * scale_x
                            ymin = bbox[2] * scale_y  
                            xmax = bbox[3] * scale_x
                            ymax = bbox[4] * scale_y
                            new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
                    elif i == 1: # top-right
                        img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y))
                        output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
                        for bbox in img_annos:
                            xmin = scale_x + bbox[1] * (1 - scale_x)
                            ymin = bbox[2] * scale_y
                            xmax = scale_x + bbox[3] * (1 - scale_x)
                            ymax = bbox[4] * scale_y
                            new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
                    elif i == 2: # bottom-left
                        img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))
                        output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
                        for bbox in img_annos:
                            xmin = bbox[1] * scale_x
                            ymin = scale_y + bbox[2] * (1 - scale_y)
                            xmax = bbox[3] * scale_x
                            ymax = scale_y + bbox[4] * (1 - scale_y)
                            new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
                    else: # bottom-right
                        img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
                        output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
                        for bbox in img_annos:
                            xmin = scale_x + bbox[1] * (1 - scale_x)
                            ymin = scale_y + bbox[2] * (1 - scale_y)
                            xmax = scale_x + bbox[3] * (1 - scale_x)
                            ymax = scale_y + bbox[4] * (1 - scale_y)
                            new_anno.append([bbox[0], xmin, ymin, xmax, ymax])  
                    
                if 0 < filter_scale:
                    new_anno = [anno for anno in new_anno if
                                filter_scale < (anno[3] - anno[1]) and filter_scale < (anno[4] - anno[2])]

                return output_img, new_anno

    def get_dataset(anno_dir, img_dir):
        # class_id = category_name.index('person')
        img_paths = []
        annos = []

        # for anno_file in glob.glob(os.path.join(anno_dir, '*.txt')):
        for anno_file in glob.glob(os.path.join(anno_dir, '*.xml')):
            # anno_id = anno_file.split('/')[-1].split('.')[0]
            anno_id = anno_file.split('/')[-1].split('x')[0]
            
            with open(anno_file, 'r') as f:
            # num_of_objs = int(f.readline())

                img_path = os.path.join(img_dir, f'{anno_id}jpg')
            
                img = cv2.imread(img_path)

                img_height, img_width, _ = img.shape
            
                del img

                boxes = []
                bnd_box = parseXmlFiles(anno_file)
                print(bnd_box)
                for bnd_id, box in enumerate(bnd_box):

                    categories_id = box[0]
                    xmin = max(int(box[1]), 0) / img_width
                    ymin = max(int(box[2]), 0) / img_height
                    xmax = min(int(box[3]), img_width) / img_width
                    ymax = min(int(box[4]), img_height) / img_height
                    boxes.append([categories_id, xmin, ymin, xmax, ymax])
                    print(boxes)
                if not boxes:
                    continue
                
            img_paths.append(img_path)
            annos.append(boxes)
            print("annos: All coordinates after scaling the original image ：",annos)
            print(img_paths)
        return img_paths, annos

    def aug_image(self, train_instance):
        jitter = self._jitter
        #print('self jitter', jitter)
        image_name = train_instance['filename']
        if self._config['IMAGE_C'] == 1:
            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        elif self._config['IMAGE_C'] == 3:
            image = cv2.imread(image_name)
        else:
            raise ValueError("Invalid number of image channels.")

        if image is None:
            print('Cannot find ', image_name)

        h = image.shape[0]
        w = image.shape[1]
        all_objs = copy.deepcopy(train_instance['object'])
        #print(jitter)
        if jitter=='True': #si une policy aug est appliquée
            #print('jitter true')
            bbs = []
            labels_bbs = []
            for i, obj in enumerate(all_objs):
                xmin = obj['xmin']
                ymin = obj['ymin']
                xmax = obj['xmax']
                ymax = obj['ymax']
                # use label field to later match it with final boxes
                bbs.append([xmin, ymin, xmax, ymax])
                labels_bbs.append(self._config['LABELS'].index(obj['name']))
            # REPLACE WITH AUGMENTATION FROM GOOGLE BRAIN TEAM !
            # select a random policy from the policy set

            

            random_policy = self._policy_chosen.select_random_policy() 
            image, bbs = self._policy_chosen.apply_augmentation(random_policy, image, bbs, labels_bbs)


            all_objs = []
            for bb in bbs:
                obj = {}
                obj['xmin'] = bb[1]
                obj['xmax'] = bb[3]
                obj['ymin'] = bb[2]
                obj['ymax'] = bb[4]
                obj['name'] = self._config['LABELS'][bb[0]]
                all_objs.append(obj)

        
        
        if jitter=='mosaic': #si on utilise l'augmentation de données mosaic
            
            OUTPUT_SIZE = (1024, 1024) # Height, Width
            SCALE_RANGE = (0.3, 0.7)
            FILTER_TINY_SCALE = 1 / 50 # if height or width lower than this scale, drop it.
            #voc Data set in format ,anno_dir It's a label xml file ,img_dir It's corresponding to jpg picture 
            ANNO_DIR = 'data/inputs/input_all.csv'
            IMG_DIR ='data/inputs/raw_data/cleaned_labels/input_train_caped300_cleaned.csv'
            

            
            img_paths, annos = get_dataset(ANNO_DIR, IMG_DIR)

            idxs = random.sample(range(len(annos)), 4)# from annos Take... Randomly from the list length 4 Number 

            new_image, new_annos = update_image_and_anno(img_paths, annos, idxs, OUTPUT_SIZE, SCALE_RANGE, filter_scale=FILTER_TINY_SCALE)
            # Update to get new graph and corresponding data anno
            cv2.imwrite('./img/wind_output.jpg', new_image)
            print("coucou")
            #annos yes 
            for anno in new_annos:
                start_point = (int(anno[1] * OUTPUT_SIZE[1]), int(anno[2] * OUTPUT_SIZE[0]))# Top left corner 
                end_point = (int(anno[3] * OUTPUT_SIZE[1]), int(anno[4] * OUTPUT_SIZE[0]))# Lower right corner 
                cv2.rectangle(new_image, start_point, end_point, (0, 255, 0), 1, cv2.LINE_AA)# Once per cycle, a rectangle is formed in the composite drawing 
                
            cv2.imwrite('data/imgs/img_mosaic/image.jpg', new_image) #doit créer un nouveau fichier avec les nouvelles données?

            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
            new_image = Image.fromarray(new_image.astype(np.uint8))
            new_image.show()  #finir par ce code et déf les fonctions avant
            # cv2.imwrite('./img/wind_output111.jpg', new_image)

            


            #if __name__ == '__main__': #si c'est la prog principal alors exécute main, n'est pas écuter car on appelle trian et dans train préprocessing est appelé en cascade donc le nom n'est pas main car n'est pas le programme principal
           