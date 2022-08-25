#fichiers csv
path_to_images = '/home/acarlier/code/project_ornithoScope/src/data/inputs/input_all.csv'
path_to_all_images = '/home/acarlier/code/project_ornithoScope/src/data/inputs/input_true_all.csv'
path_to_empty_images = '/home/acarlier/code/project_ornithoScope/src/data/inputs/input_true_empty.csv'
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
        #paths.append(line)
        #we want to keep the first part of the file to .jpg
        paths.append(line.split(',')[0])
       

# print(paths_full[:5])
# print(paths[:5])

for path in paths_full:
    if path in paths:
        path_full = paths_full.pop(paths_full.index(path))

with open(path_to_empty_images, 'w') as final_file:
    for path in paths_full:
        final_file.write(path + '\n')