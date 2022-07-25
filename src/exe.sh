# Select GPU
export CUDA_VISIBLE_DEVICES=<id>

# Train config
python3 train.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json

# Evaluate config
python3 evaluate.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json 

# Real time with USB webcam
python3 predict.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json -r True -i 0

# Image/Video/Images folder prediction
python3 predict.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json -i <path to file/directory>

#tracer courbes loss et val_loss après un entrainement

python3 history.py -p data/pickles/history/history_data/saved_weights/new_file_RMS_CDR_bestLoss.h5.p

# Multi training
sh multi_train.sh <path to file that list config files>

