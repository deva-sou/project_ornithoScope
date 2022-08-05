# Train config
python3 train.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json

# Evaluate config
python3 evaluate.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json 

# Real time
python3 predict.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json -r True -i 0

# Image.s/Video prediction
python3 predict.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json -i <path to file/directory>

# Multi training
sh multi_train.sh <path to file that list config files>
