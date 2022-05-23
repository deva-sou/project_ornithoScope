import tensorflow as tf

# Convert the model
model_path = '/home/acarlier/code/project_ornithoScope/src/data/saved_weights/data_aug_policies/MobileNet_caped300_data_aug_v2_bestLoss.h5'
new_model = '/home/acarlier/code/project_ornithoScope/src/data/saved_weights/tflite/MobileNet_caped300_data_aug_v2_bestLoss.tflite'
model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(new_model, "wb").write(tflite_model)