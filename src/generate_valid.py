import numpy as np

# Parameters
input_path = 'data/inputs/input_train.csv'
separator = ','
cap = 20
max_ratio = 0.1

# Create ouput file name
decomposed_path = input_path.split('.')
output_train_path = '.'.join(
        decomposed_path[:-2] +
        [decomposed_path[-2] + '_trainset'] +
        decomposed_path[-1:]
    )
output_valid_path = '.'.join(
        decomposed_path[:-2] +
        [decomposed_path[-2] + '_validset'] +
        decomposed_path[-1:]
    )

print('\nTrain output file: %s' % output_train_path)
print('Validation output file: %s\n' % output_valid_path)

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

# Initialise usefull dicts
max_counter = {species: min(cap, round(max_ratio * initial_counter[species])) for species in initial_counter}
final_counter = {species: 0 for species in initial_counter}

# Create output files
output_train = open(output_train_path, 'w')
output_valid = open(output_valid_path, 'w')

# Main loop
while len(boxes) > 0:
    # Get current box and image
    current_box = boxes[0]
    current_image = current_box[0]

    # Get current image boxes
    current_boxes = []
    for box in boxes:
        if box[0] == current_image:
            current_boxes.append(box)
    
    # Remove current image boxes from the global boxes list
    for box in current_boxes:
        boxes.remove(box)
    
    # Count current image boxes per species
    current_counter = {}
    for box in current_boxes:
        species = box[5]
        if species in current_counter:
            current_counter[species] += 1
        else:
            current_counter[species] = 1
    
    # `valid` is `True` if the image does not break max count limit
    valid = True
    for species in current_counter:
        if current_counter[species] + final_counter[species] > max_counter[species]:
            valid = False
            break
    
    if valid:
        # We will write these boxes in the validation file 
        output_file = output_valid

        # Increment counters
        for species in current_counter:
            final_counter[species] += current_counter[species]
    else:
        # We will write in the train file
        output_file = output_train
    
    # Write boxes in the selected output file
    for box in current_boxes:
        output_file.write(separator.join(box) + '\n')