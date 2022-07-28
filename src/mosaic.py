import random
import cv2
import os
import glob
import numpy as np
from PIL import Image
#from lxml import etree
#from ipdb import set_trace
    OUTPUT_SIZE = (1024, 1024) # Height, Width
    SCALE_RANGE = (0.5, 0.5)
    FILTER_TINY_SCALE = 1 / 50 # if height or width lower than this scale, drop it.
#voc Data set in format ,anno_dir It's a label xml file ,img_dir It's corresponding to jpg picture 
    ANNO_DIR = 'data/inputs/input_all.csv'
    IMG_DIR ='data/inputs/raw_data/cleaned_labels/input_train_caped300_cleaned.csv'
# category_name = ['background', 'person']


    def main():
    img_paths, annos = get_dataset(ANNO_DIR, IMG_DIR)
    # set_trace()
    idxs = random.sample(range(len(annos)), 4)# from annos Take... Randomly from the list length 4 Number 
    # set_trace()
    new_image, new_annos = update_image_and_anno(img_paths, annos, idxs, OUTPUT_SIZE, SCALE_RANGE, filter_scale=FILTER_TINY_SCALE)
    # Update to get new graph and corresponding data anno
    cv2.imwrite('./img/wind_output.jpg', new_image)
    #annos yes 
    for anno in new_annos:
        start_point = (int(anno[1] * OUTPUT_SIZE[1]), int(anno[2] * OUTPUT_SIZE[0]))# Top left corner 
        end_point = (int(anno[3] * OUTPUT_SIZE[1]), int(anno[4] * OUTPUT_SIZE[0]))# Lower right corner 
        cv2.rectangle(new_image, start_point, end_point, (0, 255, 0), 1, cv2.LINE_AA)# Once per cycle, a rectangle is formed in the composite drawing 
        cv2.imwrite('./img/wind_output_box.jpg', new_image)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        new_image = Image.fromarray(new_image.astype(np.uint8))
    # new_image.show()
    # cv2.imwrite('./img/wind_output111.jpg', new_image)
    def update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale=0.):
        output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
        scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
        scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
        divid_point_x = int(scale_x * output_size[1])
        divid_point_y = int(scale_y * output_size[0])
        new_anno = []
        for i, idx in enumerate(idxs):
        #set_trace()
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
        return output_img, new_anno

    def get_dataset(anno_dir, img_dir):
    # class_id = category_name.index('person')
        img_paths = []
        annos = []
    # for anno_file in glob.glob(os.path.join(anno_dir, '*.txt')):
        for anno_file in glob.glob(os.path.join(anno_dir, '*.xml')):
    # anno_id = anno_file.split('/')[-1].split('.')[0]
            anno_id = anno_file.split('/')[-1].split('x')[0]
    # set_trace()
    # with open(anno_file, 'r') as f:
    # num_of_objs = int(f.readline())
    # set_trace()
            img_path = os.path.join(img_dir, f'{anno_id}jpg')
            print(img_path)
            img = cv2.imread(img_path)
    # set_trace() 
            img_height, img_width, _ = img.shape
            print(img.shape)
            del img
            boxes = []
            bnd_box = parseXmlFiles(anno_file)
            print(bnd_box)
            for bnd_id, box in enumerate(bnd_box):
    # set_trace()
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


#def parseXmlFiles(anno_dir):
    #tree = etree.parse(anno_dir)
    #root = tree.getroot()
    #objectes = root.findall('.//object')
    #bnd_box = []
    #for object in objectes:
        #name = object.find("name").text
        #bndbox = object.find("bndbox")
        #xmin = float(bndbox.find("xmin").text)
        #xmax = float(bndbox.find("xmax").text)
        #ymin = float(bndbox.find("ymin").text)
        #ymax = float(bndbox.find("ymax").text)
# bnd_box.append([name, xmin, xmax, ymin, ymax])
        #bnd_box.append([name, xmin, ymin, xmax, ymax])
# print(len(bnd_box),bnd_box)
    #return bnd_box

#if __name__ == '__main__':
    #main()
