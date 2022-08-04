import numpy as np

# Parameters
input_path = 'data/inputs/input_train_trainset.csv'
separator = ','
cap = 300

# Create ouput file name
decomposed_path = input_path.split('.')
decomposed_path[-2] += '_cap%d' % cap
output_path = '.'.join(decomposed_path)

print('\nOutput file: %s\n' % output_path)

# Open input file and extract boxes values
with open(input_path, 'r') as file_buffer:
    boxes = []
    for line in file_buffer.readlines():
        line = line[:-1] if line[-1] == '\n' else line
        boxes.append(line.split(separator))

# Count every class occuracy
initial_counter = {}
for box in boxes:
    species = box[5]
    if species in initial_counter:
        initial_counter[species] += 1
    else:
        initial_counter[species] = 1

counter = initial_counter.copy()

# Initialize final counter
keeped_counter = {}
for species in counter:
    keeped_counter[species] = 0

# List underrepresented species
underrepresented = [species for species in counter if counter[species] < cap]

# List images with underrepresented species
images_with_underepresented = []
for box in boxes:
    if box[5] in underrepresented:
        images_with_underepresented.append(box[0])

# Keep boxes on a picture with underrepresented species
keeped_lines = []
for i, box in enumerate(boxes):
    if box[0] in images_with_underepresented:
        keeped_lines.append(i)
        keeped_counter[box[5]] += 1

# Remove underrepresented species from species counter
for species in underrepresented:
    del counter[species]

# Loop on every remaining species
while len(counter) > 0:
    # Select the species with the smallest occuracy
    smallest_species = list(counter.keys())[0]
    for species in counter:
        if counter[species] < counter[smallest_species]:
            smallest_species = species
    
    # Loop to reach cap limit for the current species
    while keeped_counter[smallest_species] < cap:
        # Select an unused line with the current species
        line = np.random.randint(len(boxes))
        while line in keeped_lines or boxes[line][5] != smallest_species:
            line += 1
            line %= len(boxes)
        
        # Add every boxes that rae on the same picture of the selected line
        image = boxes[line][0]
        for i, box in enumerate(boxes):
            if box[0] == image:
                keeped_lines.append(i)
                keeped_counter[box[5]] += 1
    
    # Delete current species
    del counter[smallest_species]

# Transform keeped boxes into writable lines
lines = []
for line in keeped_lines:
    line_to_write = separator.join(boxes[line]) + '\n'
    lines.append(line_to_write)

# Write file lines
file = open(output_path, 'w')
file.writelines(lines)

# Print species count changes
for species in initial_counter:
    print('%s : %d -> %d' % (species, initial_counter[species], keeped_counter[species]))