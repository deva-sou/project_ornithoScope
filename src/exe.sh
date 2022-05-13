#TRAIN
#CUDA_VISIBLE_DEVICES=1, python3 train.py -c config/get_bad_images/config_lab_mobilenetV1_train_all_test_all.json

#EVALUATE
#CUDA_VISIBLE_DEVICES=1, python3 evaluate.py -c config/get_bad_images/config_lab_mobilenetV1_train_all_test_all.json

#PREDICT
CUDA_VISIBLE_DEVICES=1, python3 predict.py -c config/get_bad_images/config_lab_mobilenetV1_train_all_test_all.json -w data/saved_weights/MobileNet_train_all_test_all_bestLoss.h5 -i /home/acarlier/code/data_ornithoscope/p0133_bird_data/raw_data/_bad_imgs
