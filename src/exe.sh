#TRAIN
#lignes de commandes à copier pour exécuter des predict, train... (copier: controle shift V)
#CUDA_VISIBLE_DEVICES=1, python3 train.py -c config/get_bad_images/config_lab_mobilenetV1_train_all_test_all.json
#######EVALUATE
#CUDA_VISIBLE_DEVICES=1, python3 evaluate.py -c config/get_bad_images/config_lab_mobilenetV1_train_all_test_all.json
#######PREDICT
#CUDA_VISIBLE_DEVICES=1, python3 predict.py -c config/get_bad_images/config_lab_mobilenetV1_train_all_test_all.json -w data/saved_weights/MobileNet_train_all_test_all_bestLoss.h5 -i /home/acarlier/code/data_ornithoscope/p0133_bird_data/raw_data/_bad_imgs
#CAP300V2_CLEANED_DATA
#CUDA_VISIBLE_DEVICES=1, python3 train.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv2_cleaned_data.json
#CUDA_VISIBLE_DEVICES=1, python3 evaluate.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv2_cleaned_data.json  

python3 predict.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json -w data/saved_weights/MobileNet_caped300_bestLoss.h5 -i /home/acarlier/code/data_ssd/ssd_maxime/PhotoFeeder/Annotated/balacet_2021_07_05-07/2021-07-05-11-03-05.jpg -o img # -o veut dire output et le format

# Train config
python3 train.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json

# Evaluate config
python3 evaluate.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json 

# Real time with USB webcam
python3 predict.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json -r True -i 0

# Image/Video/Images folder prediction
python3 predict.py -c config/data_aug_policies/config_lab_mobilenetV1_labels_caped300_data_augv0.json -i <path to file/directory>

# Multi training
sh multi_train.sh <path to file that list config files>
