import os, os.path
import pandas as pd
from xml.etree import ElementTree as ET
from keras_yolov2.preprocessing import parse_annotation_xml, parse_annotation_csv
from keras_yolov2.utils import import_feature_extractor
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

def get_info_from_one_xml(xml_path, path_raw_data, path_ornithoTasks):
    list_object = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Filename Extraction and local path creation
    filename = root.find("./filename").text
    local_img_path = f"{path_raw_data}{filename.split('/', 1)[1]}"
    # Width and height extraction
    width = int(root.find("./size/width").text)
    height = int(root.find("./size/height").text)
    task_name_xml = filename.split('/')[1]
    # print('task_name', task_name_xml)

    for object in root.findall("./object"):
        # Extract specie name
        for attributes in object.findall("./attributes/"):
            if attributes.find('name').text == 'species':
                specie = attributes.find('value').text
                # Data creation
                data = {}
                data['split_value'] = pd.read_csv(path_ornithoTasks)
                data['file_path'] = local_img_path
                data['label'] = specie
                # Extract bndbx
                bndbox = object.find('./bndbox')
                x_min = round(float(bndbox.find("xmin").text) / width, 4)
                y_min = round(float(bndbox.find("ymin").text) / height, 4)
                x_max = round(float(bndbox.find("xmax").text) / width, 4)
                y_max = round(float(bndbox.find("ymax").text) / height, 4)
                data['x_min'] = x_min
                data['y_min'] = y_min
                data["empty_1"] = ""
                data["empty_2"] = ""
                data['x_max'] = x_max
                data['y_max'] = y_max
                data["empty_3"] = ""
                list_object.append(data)
    #   print(list_object)
    return list_object


def get_info_from_all_xml(tasks_dir, list_object, path_annotation, path_raw_data, path_ornithoTasks):
    for i in range(len(tasks_dir)):
        # For each task
        task_name = tasks_dir[i]
        complete_path = path_annotation + task_name + '/Annotations/bird/' + task_name + '/'
        # Get all images of a task
        image_paths = os.listdir(complete_path)
        for img in range(len(image_paths)):
            xml_path = complete_path + image_paths[img]
            get_info_from_one_xml(xml_path, path_raw_data, path_ornithoTasks)
            # print(xml_path, 'done')
    return list_object


def import_tasks_as_df(path_ornithoTasks):
    df_task = pd.read_csv(path_ornithoTasks)
    df_task = df_task.rename(columns={"Task name": "task_name"})
    return df_task

def add_validation_split_value(df):
    return df


def df_to_csv(df, path_and_name_csv):
    return df.to_csv(path_and_name_csv, encoding='utf-8', index=False, header=False)


def csv_to_df(csv_path):
    return pd.DataFrame.to_csv(csv_path)


def create_df_for_input(list_of_object, unwanted_labels_list):
    count = 0
    list_df = []
    for object in list_of_object:
        count += 1
        content = pd.DataFrame(object, index=[count])
        list_df.append(content)
    df_input = pd.concat(list_df)
    for label in unwanted_labels_list:
        df_input = df_input[(df_input.label != label)]
    return df_input


def create_input_as_df(task_dir, list_object, path_annotation, path_raw_data, path_ornithoTasks):
    print('Import data ...')
    get_info_from_all_xml(task_dir, list_object, path_annotation, path_raw_data, path_ornithoTasks)
    print('Done!')
    print('Create dataframe ...')
    df = create_df_for_input(list_object)
    print('Done!')
    return df


def create_input_as_csv(tasks_dir, list_object, path_input_csv, path_annotation, path_raw_data, path_ornithoTasks):
    df = create_input_as_df(tasks_dir, list_object, path_annotation, path_raw_data, path_ornithoTasks)
    df_input = add_validation_split_value(df)
    print('Export to csv ...')
    df_to_csv(df, path_input_csv)
    print('Done!')
    return df


def main(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())


    if config['parser_annotation_type'] == 'csv':
        # parse config.json
        path_annotation = config['input_generator']['path_annotation']
        path_raw_data = config['input_generator']['path_raw_data']
        unwanted_labels_list = config['model']['unwanted_labels'] 
        path_input_train_csv = config['train']['train_csv_file']
        path_input_test_csv = config['valid']['valid_csv_file']
   
    path_input_csv = 'keras_yolo2/birds_data_csv/input.csv'
    
    path_ornithoTasks = './birds_data_csv/Ornithotasks - CVAT_task.csv'
    print('path_ann',path_annotation)
    tasks_dir = os.listdir(path_annotation)
    
    columns_input_data = ["split_value", 
                          "file_path",
                          "x_min", "y_min",
                          "x_max", "y_max", 
                          "label"
                          ]
    df_input = pd.DataFrame(columns=columns_input_data)
    df = create_input_as_df(tasks_dir, path_annotation, path_raw_data, path_ornithoTasks)
    df_tasks = import_tasks_as_df(path_ornithoTasks)

if __name__ == '__main__':
    #_args = argparser.parse_args()
    #main(_args)

    import csv

    infile_name = '/home/acarlier/code/OrnithoScope_keras/keras_yolo2/birds_data_csv/input copy.csv'

    with open(infile_name, newline='') as infile:
        csv_writers = {}
        files = []
        reader = csv.DictReader(infile)

        for row in reader:
            if (key := f"{row['split']}") not in csv_writers:
                # Create the csv file and a corresponding DictWriter.
                outfile_name = f'{key}.csv'
                fileout = open(outfile_name, 'w', newline='')
                files.append(fileout)  # To have it closed later.
                writer = csv.DictWriter(fileout, fieldnames=reader.fieldnames)
                writer.writeheader()
                csv_writers[key] = writer

            # Write the line to corresponding csv writer.
            csv_writers[key].writerow(row)

        # Close all CSV output files.
        for f in files:
            f.close()