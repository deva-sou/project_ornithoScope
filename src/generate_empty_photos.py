import cv2

#fichiers csv
path_to_images = '/home/acarlier/code/project_ornithoScope/src/data/inputs/input_all.csv'
path_to_all_images = '/home/acarlier/code/project_ornithoScope/src/data/inputs/input_true_all.csv'
path_to_empty_images = '/home/acarlier/code/project_ornithoScope/src/data/inputs/input_true_empty.csv'
path_to_empty_images_train = '/home/acarlier/code/project_ornithoScope/src/data/inputs/input_true_empty_train.csv'
path_to_empty_images_test = '/home/acarlier/code/project_ornithoScope/src/data/inputs/input_true_empty_test.csv'

#extraction des images avec oiseaux

with open(path_to_all_images, 'r') as file_buffer_full:
    paths_full = []
    for line in file_buffer_full.readlines():
        line = line[:-1] if line[-1] == '\n' else line
        paths_full.append(line)

with open(path_to_images, 'r') as file_buffer:
    paths = []
    for line in file_buffer.readlines():
        line = line[:-1] if line[-1] == '\n' else line
        paths.append(line.split(',')[0])
       

for path in paths:
    if path in paths_full:
        paths_full.remove(path)




task_for_test = ['task_2021-03-01_10',
'task_20210611_Lab',
'task_20211204_Orlu',
'task_2021-03-01_09',
'task_20210612_1_Lab',
'task_20210526_UPS',
'task_20210705-07_balacet',
'task_21-01-2021']
#on écrit les images vide dans un train d'image empty

with open(path_to_empty_images_train, 'w') as final_file_test:  
    for path in paths_full:
        new_path = '/home/acarlier/code/data_ornithoscope/p0133_bird_data/raw_data/' + path
        img = cv2.imread(new_path)
        try:
            width, height, _ = img.shape #gives the dimensiosn of the images
            if path.split('/')[0] not in task_for_test:
                final_file_test.write(f'{path},,,,,,{width},{height}\n')
        except:
            pass

#on écrit les images vide dans un test d'image empty
 

with open(path_to_empty_images_test, 'w') as final_file_test:  
    for path in paths_full:
        new_path = '/home/acarlier/code/data_ornithoscope/p0133_bird_data/raw_data/' + path
        img = cv2.imread(new_path)
        try: #pratique quand on utilise imread car beaucoup d'erreurs les lecture des images
            width, height, _ = img.shape #gives the dimensiosn of the images
            if path.split('/')[0] in task_for_test:
                final_file_test.write(f'{path},,,,,,{width},{height}\n')
        except:
            pass








