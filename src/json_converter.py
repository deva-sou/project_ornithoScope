import json
import os


# Base path
base_path = '/media/Pictures'

# Folders with species label
folders = {
    '145303': 'TARAUL',
    '9801':   'GROBEC',
    '9398':   'CHAELE',
    '3017':   'PIGBIS',
    '14850':  'ETOEUR',
    '17871':  'PICEIP',
    '792985': 'PICMAR',
    '8088':   'GEACHE',
    '13851':  'MOIFRI',
    '204496': 'CORNOI',
    '7278':   'ORILON',
    '18911':  'PERCOL',
    '10069':  'PINNOR'
}


# Loop on folders
for folder in folders:
    specie = folders[folder]

    # Dict with file name as key
    output_dict = {}

    # Store file in list
    with open(os.path.join(base_path, folder + '.csv'), 'r') as file_buffer:
        lines = [line for line in file_buffer.readlines()]
    
    # Loop on every line (correspond to a unique box)
    for line in lines:
        # Extract and convert values
        image_path, xmin, ymin, xmax, ymax, _, _, width, height = line.split(',')
        xmin, ymin, xmax, ymax, width, height = int(xmin), int(ymin), int(xmax), int(ymax), int(width), int(height)
        
        if image_path in output_dict:
            # Add a new box
            output_dict[image_path]['boxes'].append({
                                                "xmin": xmin,
                                                "ymin": ymin,
                                                "xmax": xmax,
                                                "ymax": ymax,
                                                "specie": specie
                                            })
        else:
            # Create a new image in the dict
            output_dict[image_path] = {
                "File_path": image_path, 
                "visited": 0, 
                "width": width, 
                "height": height, 
                "boxes": [{
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "specie": specie
                    }]
            }
    
    # Convert the dict into a list
    output_list = list(output_dict.values())

    # Sort the list by the biggest boxes
    output_list = sorted(output_list,
            key=lambda img : max([
                    (box['xmax'] - box['xmin']) * (box['ymax'] - box['xmin'])
                    for box in img['boxes']
                ]),
            reverse=True)
    
    # Write the json file
    with open(os.path.join(base_path, folder + '.json')) as file_buffer:
        json.dump(output_list[:1000], file_buffer, 'w', indent=4)
