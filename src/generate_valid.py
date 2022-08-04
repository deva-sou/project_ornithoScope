import csv
import numpy as np

def remove_duplicates(input_list):
    dic = dict.fromkeys(input_list)
    output_list = list(dic.keys())
    return output_list

# Parameters
input_path = 'data/inputs/input_all.csv'
separator = ','
cap = 20
max_ratio = 0.1

# Create ouput file name
decomposed_path = input_path.split('.')
decomposed_path[-2] += '_validset'
output_path = '.'.join(decomposed_path)

print('\nOutput file: %s\n' % output_path)

# Open input file and extract boxes values
with open(input_path, 'r') as file_buffer:
    boxes = []
    for line in file_buffer.readlines():
        line = line[:-1] if line[-1] == '\n' else line
        boxes.append(line.split(separator))

# Shuffle a bit
np.random.shuffle(boxes)

# Count every class occuracy
initial_counter = {}
for box in boxes:
    species = box[5]
    if species in initial_counter:
        initial_counter[species] += 1
    else:
        initial_counter[species] = 1

# Initialize some usefull lists
available_species = list(initial_counter.keys())
max_image = {species: min(cap, round(max_ratio * initial_counter[species])) for species in available_species}
all_images = [box[0] for box in boxes]
available_images = remove_duplicates(all_images)
final_counter = {species: 0 for species in available_species}

# Create output file
output_file = open(output_path, 'w')

# Main loop
while len(available_species) > 0:
    # Select an image
    for box in boxes:
        if box[5] in available_species and box[0] in available_images:
            chosen_image = box[0]
            break
    
    # Remove the chosen image from available images
    available_images.remove(chosen_image)
    
    # List all boxes on the chosen image
    chosen_boxes = []
    for box in boxes:
        if box[0] == chosen_image:
            chosen_boxes.append(box)
    
    # Count every species on the chosen image
    chosen_counter = {}
    for box in chosen_boxes:
        species = box[5]
        if species in chosen_counter:
            chosen_counter[species] += 1
        else:
            chosen_counter[species] = 1
    
    # The image is valid only if all species don't break counters
    valid = True
    for species in chosen_counter:
        if chosen_counter[species] + final_counter[species] > max_image[species]:
            valid = False
            break
    
    # If the image is not valid, take an other one
    if not valid:
        continue

    # Increment final counter
    for species in chosen_counter:
        final_counter[species] += chosen_counter[species]
    
    # Write image and its boxes in the output file
    for box in chosen_boxes:
        output_file.write(','.join(box) + '\n')
    
    # Remove species that are full
    for species in chosen_counter:
        if final_counter[species] >= max_image[species]:
            available_species.remove(species)

    











































# #print(counter)

# file = open('data/inputs/valid_file.csv', 'w')
# writer=csv.writer(file)
# counter = {}
# L = ["MESCHA", "SITTOR", "MESBLE", "MESNON", "PINARB", "ACCMOU", "ROUGOR", "VEREUR", "TOUTUR", "RONGEUR", "ECUROU", "PIEBAV", "MESNOI", "MESHUP"]

# while(len(L)) > 0:
#     for box in boxes:
#         species = []
#         image = box[0]
#         objet = []
#         for boite in boxes:
#             if boite[0] == image:
#                 species.append(boite[5])
#                 objet.append(boite)
        
#         for especes in species:
            
#             if especes in counter:
#                 counter[especes] += 1
#             else:
#                 counter[especes] = 1
            
#             if counter[especes] <= min(20, 0.1 * initial_counter[especes]):
#                 for element in objet:
#                     writer.writerow(element)
#             if counter[especes] == min(20, 0.1 * initial_counter[especes]):
#                 L.remove(especes)