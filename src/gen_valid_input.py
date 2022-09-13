import numpy as np


# Parameters
input_path = '/home/acarlier/code/project_ornithoScope/src/data/inputs/input_train_iNat.csv'
separator = ','
cap = 20
max_ratio = 0.1
label = ['MESCHA', 'SITTOR', 'MESBLE', 'MESNON', 'PINARB', 'ACCMOU', 'ROUGOR', 'VEREUR', 'TOUTUR', 'ECUROU', 'PIEBAV', 'MULGRI', 'MESNOI', 'MESHUP', 'MOIDOM', 'ECUROU', 'PINNOR']

# Create ouput file name
decomposed_path = input_path.split('.')
output_train_path = '.'.join(
        decomposed_path[:-2] +
        [decomposed_path[-2] + '_trainset_better'] +
        decomposed_path[-1:]
    )
output_valid_path = '.'.join(
        decomposed_path[:-2] +
        [decomposed_path[-2] + '_validset_better'] +
        decomposed_path[-1:]
    )

print('\nTrain output file: %s' % output_train_path)
print('Validation output file: %s\n' % output_valid_path)

# Open input file and extract boxes values
with open(input_path, 'r') as file_buffer:
    boxes = []
    for line in file_buffer.readlines():
        line = line[:-1] if line[-1] == '\n' else line
        boxes.append(line.split(separator)) #boxes est une liste de listes qui contient les lignes des fichiers csv sous formes de listes splitée avec "," comme séparateur

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

# Initialise usefull dicts
max_counter = {species: min(cap, round(max_ratio * initial_counter[species])) for species in initial_counter}
min_counter_task = {}
classes_mineures = ['ACCMOU', 'ROUGOR', 'TOUTUR', 'MULGRI', 'MESNOI', 'MESHUP', 'MOIDOM', 'PINNOR']
for species in label:
    if species in classes_mineures:
        min_counter_task[species] = 10 #on veut garder une bonne partie d'image iNat pour les espèces mineures pour ne pas dérégler le train
    else:
        min_counter_task[species] = 18

final_counter = {species: 0 for species in initial_counter}
final_counter_task = {species: 0 for species in label}

# Create output files
output_train = open(output_train_path, 'w')
output_valid = open(output_valid_path, 'w')

# Main loop
while len(boxes) > 0:
    # Get current box and image
    current_box = boxes[0] #première ligne du csv
    current_image = current_box[0]

    # Get current image boxes en prenant bien deux lignes si deux oiseaux sur une même image
    current_boxes = []
    for box in boxes:
        if box[0] == current_image:
            current_boxes.append(box)
    
    # Remove current image boxes from the global boxes list
    for box in current_boxes:
        boxes.remove(box)
    
    # Count current image boxes per species (on actualise le compteur avec ce qu'on vient de rajouter)
    current_counter = {}

    for box in current_boxes:
        species = box[5]
        if species in current_counter:
            current_counter[species] += 1
        else:
            current_counter[species] = 1

    #Count image from tasks (on met à jour le nombre d'image qu'on a rajouté dans le validset et qui proviennent des tasks)

    current_counter_task = {}
    for species in label:
        current_counter_task[species] = 0

    for box in current_boxes:
        img_name = box[0]
        species = box[5]
        if img_name.startswith('task'):
            if img_name in current_counter_task:
                current_counter_task[species] += 1
            else:
                current_counter_task[species] = 1
    
  
     # `valid` is `True` if the image does not break max count limit
    valid = True
    act = True
    for species in current_counter: 
        if current_counter_task[species] + final_counter_task[species] > min_counter_task[species]:
        # if current_counter[species] + final_counter[species] > max_counter[species] and current_counter_task[species] + final_counter_task[species] > min_counter_task[species]:
            valid = False
            break
    for species in current_counter:
        #if current_counter_task[species] + final_counter_task[species] > min_counter_task[species]:
        if current_counter[species] + final_counter[species] > max_counter[species]:
            act  = False
            break
    
    
    
    if valid and [box[0].startswith('task') for box in current_boxes].count(True) in [1,2,3,4]: #si les images de la box proviennent des tasks et qu'il y a 1,2,3 ou 4 oiseaux dans la box
        # We will write these boxes in the validation file 
        # Increment counters
        output_file = output_valid
        for species in current_counter:
            final_counter[species] += current_counter[species]
            print(species)
            print(final_counter_task)
            final_counter_task[species] += current_counter_task[species] 
    elif act and [box[0].startswith('iNat') for box in current_boxes].count(True) in [1,2,3,4]:
        output_file = output_valid
        for species in current_counter:
            final_counter[species] += current_counter[species]
            final_counter_task[species] += current_counter_task[species] 

    else:
        # We will write in the train file
        print(final_counter_task)
        output_file = output_train
    
    # Write boxes in the selected output file
    for box in current_boxes:
        output_file.write(separator.join(box) + '\n')


