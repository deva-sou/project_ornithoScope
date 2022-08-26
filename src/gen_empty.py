import cv2
import os


task_input_path = 'data/inputs/input_all_iNat.csv'
all_image_path = 'data/inputs/all_paths.txt'


with open(task_input_path, 'r') as file_buffer:
    task_image_paths = []
    for line in file_buffer.readlines():
        image_path = line.split(',')[0]
        if len(image_path) > 1:
            task_image_paths.append(image_path)

# print(task_image_paths[:20])

with open(all_image_path, 'r') as file_buffer:
    all_image_paths = []
    for line in file_buffer.readlines():
        if len(line) > 1:
            all_image_paths.append(line[:-1])

# print(all_image_paths[:20])

for path in task_image_paths:
    if path in all_image_paths:
        all_image_paths.remove(path)

# print(all_image_paths)

test_tasks = [input_test[:-4] for input_test in os.listdir('/home/dams1309/project_ornithoScope/src/data/inputs/input_test_per_tasks')]

# print(test_tasks)

train_image_paths = []
test_image_paths = []

for path in all_image_paths:
    test_image = False
    for test_task in test_tasks:
        if test_task in path:
            test_image_paths.append(path)
            test_image = True
            break
    
    if not test_image:
        train_image_paths.append(path)

# print(all_image_paths)

# done = False
# i = 0
# while not done:
#     path = '/home/dams1309/raw_data/' + all_image_paths[i % len(all_image_paths)]
#     img = cv2.imread(path)
#     cv2.imshow('No bird image', cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
#     key = cv2.waitKey(0)
#     if key == ord('q'):
#         done = True
#     elif key == ord('n'):
#         i += 1
#     elif key == ord('p'):
#         i -= 1

with open('data/inputs/train_empty.csv', 'w') as file_buffer:
    for path in train_image_paths:
        img = cv2.imread('/home/dams1309/raw_data/' + path)
        try:
            width, height, _ = img.shape
            file_buffer.write(f'{path},,,,,,{width},{height}\n')
        except: pass

with open('data/inputs/test_empty.csv', 'w') as file_buffer:
    for path in test_image_paths:
        img = cv2.imread('/home/dams1309/raw_data/' + path)
        try:
            width, height, _ = img.shape
            file_buffer.write(f'{path},,,,,,{width},{height}\n')
        except: pass