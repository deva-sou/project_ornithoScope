import os
import sys

import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Reshape, Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, ExponentialDecay

from .cosine_decay import WarmUpCosineDecayScheduler
from .map_evaluation import MapEvaluation
from .preprocessing import BatchGenerator
from .utils import decode_netout, import_feature_extractor, import_dynamically
from .yolo_loss import YoloLoss


class YOLO(object):
    def __init__(self, backend, input_size, labels, anchors, gray_mode=False, freeze=False):

        self._input_size = input_size
        self._gray_mode = gray_mode
        self.labels = list(labels)
        self._nb_class = len(self.labels)
        self._nb_box = len(anchors) // 2
        self._anchors = anchors
        self._freeze = freeze

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        if self._gray_mode:
            self._input_size = (self._input_size[0], self._input_size[1], 1)
            input_image = Input(shape=self._input_size)
        else:
            self._input_size = (self._input_size[0], self._input_size[1], 3)
            input_image = Input(shape=self._input_size)


        self._feature_extractor = import_feature_extractor(backend, self._input_size, self._freeze)
               
        
        print(self._feature_extractor.feature_extractor.summary())
        print(self._feature_extractor.get_output_shape())
        self._grid_h, self._grid_w = self._feature_extractor.get_output_shape()
        features = self._feature_extractor.extract(input_image)

        # make the object detection layer
        output = Conv2D(self._nb_box * (4 + 1 + self._nb_class),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='Detection_layer',
                        kernel_initializer='lecun_normal')(features)
        output = Reshape((self._grid_h, self._grid_w, self._nb_box, 4 + 1 + self._nb_class), name="YOLO_output")(output)

        self._model = Model(input_image, output)
        print(self._model.summary())

        # initialize the weights of the detection layer
        layer = self._model.get_layer("Detection_layer")
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape) / (self._grid_h * self._grid_w)
        new_bias = np.random.normal(size=weights[1].shape) / (self._grid_h * self._grid_w)

        layer.set_weights([new_kernel, new_bias])

        # print a summary of the whole model
        self._model.summary()

        # declare class variables
        self._batch_size = None
        self._object_scale = None
        self._no_object_scale = None
        self._coord_scale = None
        self._class_scale = None
        self._debug = None
        self._warmup_batches = None
        self._interpreter = None
        self._tflite = False

    def load_weights(self, weight_path):
        self._model.load_weights(weight_path)
    
    def load_lite(self, lite_path):
        self._tflite = True
        self._interpreter = tf.lite.Interpreter(model_path=lite_path)
        self._interpreter.allocate_tensors()

    def train(self, train_imgs,  # the list of images to train the model
              valid_imgs,  # the list of images used to validate the model
              train_times,  # the number of time to repeat the training set, often used for small datasets
              nb_epochs,  # number of epoches
              learning_rate,  # the learning rate
              batch_size,  # the size of the batch
              object_scale,
              no_object_scale,
              coord_scale,
              class_scale,
              policy,
              saved_pickles_path,
              optimizer_config,
              saved_weights_name='best_weights.h5',
              workers=3,
              max_queue_size=8,
              early_stop=True,
              custom_callback=[],
              tb_logdir="./",
              iou_threshold=0.5,
              score_threshold=0.5,
              cosine_decay=False
              ):

        self._batch_size = batch_size

        self._object_scale = object_scale
        self._no_object_scale = no_object_scale
        self._coord_scale = coord_scale
        self._class_scale = class_scale

        self._debug = 0
        self._saved_pickles_path = saved_pickles_path
        #######################################
        # Make train and validation generators, nos ensembles de train et de validation
        #######################################

        generator_config = {
            'IMAGE_H': self._input_size[0],
            'IMAGE_W': self._input_size[1],
            'IMAGE_C': self._input_size[2],
            'GRID_H': self._grid_h,
            'GRID_W': self._grid_w,
            'BOX': self._nb_box,
            'LABELS': self.labels,
            'CLASS': len(self.labels),
            'ANCHORS': self._anchors,
            'BATCH_SIZE': self._batch_size,
        }
        
        #train_imgs: the list of img to train the model, donc format jpg
        #BatchGenerator: défini dans preprocessing

        train_generator = BatchGenerator(train_imgs,
                                         generator_config,
                                         norm=self._feature_extractor.normalize,
                                         policy_container = policy)
        valid_generator = BatchGenerator(valid_imgs,
                                         generator_config,
                                         norm=self._feature_extractor.normalize,
                                         jitter=False)

        ############################################
        # Compile the model
        ############################################

        optimizer = YOLO.create_optimizer(optimizer_config, learning_rate)

        loss_yolo = YoloLoss(self._anchors, (self._grid_w, self._grid_h), self._batch_size,
                             lambda_coord=coord_scale, lambda_noobj=no_object_scale, lambda_obj=object_scale,
                             lambda_class=class_scale)
        self._model.compile(loss=loss_yolo, optimizer=optimizer)

        ############################################
        # Make a few callbacks (gère l'évolution du lr en fonction du temps et selon nos exigences)
        ############################################

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                        patience = 7, min_lr = 0.00001, verbose = 1) #Reduce learning rate when a metric has stopped improving. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.

        early_stop_cb = EarlyStopping(monitor='val_loss',
                                      min_delta=0.001,
                                      patience=15,
                                      mode='min',
                                      verbose=1)#Stop training when a monitored metric has stopped improving. Where an absolute change of less than min_delta, will count as no improvement. 

        tensorboard_cb = TensorBoard(log_dir=tb_logdir,
                                     histogram_freq=0,
                                     # write_batch_performance=True,
                                     write_graph=True,
                                     write_images=False)

        root, ext = os.path.splitext(saved_weights_name)
        ckp_best_loss = ModelCheckpoint(root + "_bestLoss" + ext,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='min',
                                        period=1)
        ckp_saver = ModelCheckpoint(root + "_ckp" + ext,
                                    verbose=1,
                                    period=10)
        
        #en dessous on ne l'a plus utilisé pour les callbacks
        
        map_evaluator_cb = MapEvaluation(self, valid_generator,
                                         save_best=False,
                                         save_name=root + "_bestMap" + ext,
                                         tensorboard=tensorboard_cb,
                                         iou_threshold=iou_threshold,
                                         score_threshold=score_threshold)

        self._warmup_batches = train_times * len(train_generator) + len(valid_generator)
        if cosine_decay:
            total_steps = int(nb_epochs * len(train_generator) / batch_size)
            warmup_steps = int(len(train_generator) / batch_size)
            warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate,
                                                    total_steps=total_steps,
                                                    warmup_learning_rate=0.0,
                                                    warmup_steps=warmup_steps,
                                                    hold_base_rate_steps=0)

        if not isinstance(custom_callback, list):
            custom_callback = [custom_callback]
        callbacks = [ckp_best_loss, ckp_saver, tensorboard_cb, map_evaluator_cb] + custom_callback
        if early_stop:
            callbacks.append(early_stop_cb)
        if cosine_decay:
            callbacks.append(warm_up_lr)

        callbacks = [reduce_lr, early_stop_cb, ckp_best_loss] #finalement on n'utilise que ce callback

        #############################
        # Start the training process
        #############################

        history = self._model.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(train_generator) * train_times,
                                  epochs=nb_epochs,
                                  validation_data=valid_generator,
                                  validation_steps=len(valid_generator),
                                  callbacks=callbacks,
                                  workers=workers,
                                  max_queue_size=max_queue_size)
        
        pickle.dump(history, open( f"{self._saved_pickles_path}/history/history_{root + '_bestLoss' + ext}.p", "wb" ) )

    def predict(self, image, iou_threshold=0.5, score_threshold=0.5):

        input_image = self.resize(image)

        ### TFLite
        if self._tflite:

            # Extract details
            input_details = self._interpreter.get_input_details()
            output_details = self._interpreter.get_output_details()
            input_type = input_details[0]['dtype']

            # Convert frame to input type
            input_image = input_image.astype(input_type)

            # Predict
            self._interpreter.set_tensor(input_details[0]['index'], input_image)
            self._interpreter.invoke()
            netout = self._interpreter.get_tensor(output_details[0]['index'])[0]

        ### TF
        else:
            netout = self._model.predict(input_image)[0]

        boxes = decode_netout(netout, self._anchors, self._nb_class, score_threshold, iou_threshold)

        return boxes
    
    def resize(self, image):
        if len(image.shape) == 3 and self._gray_mode:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image[..., np.newaxis]
        elif len(image.shape) == 2 and not self._gray_mode:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 2:
            image = image[..., np.newaxis]

        image = cv2.resize(image, (self._input_size[1], self._input_size[0]))
        image = image[..., ::-1]  # make it RGB (it is important for normalization of some backends)

        image = self._feature_extractor.normalize(image)
        if len(image.shape) == 3:
            input_image = image[np.newaxis, :]
        else:
            input_image = image[np.newaxis, ..., np.newaxis]
        
        return input_image

    def create_optimizer(optimizer_config, default_lr):
        """
        Instanciate an optimizer corresponding to `optimizer_config` dict.
        """

        if not 'name' in optimizer_config.keys():
            raise Exception('Optimizer name not indicated')
        
        # Create learning-rate scheduler
        lr_scheduler = YOLO.create_lr_scheduler(optimizer_config['lr_scheduler'], default_lr)

        if optimizer_config['name'] == 'Adam':
            # Parse Adam arguments
            beta_1 = float(optimizer_config.get('beta_1', 0.9))
            beta_2 = float(optimizer_config.get('beta_2', 0.999))
            epsilon = float(optimizer_config.get('epsilon', 1e-08))
            decay = float(optimizer_config.get('decay', 0.0))

            # Instanciate Adam
            return Adam(
                    learning_rate=lr_scheduler,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=epsilon,
                    decay=decay
                )
        
        if optimizer_config['name'] == 'SGD':
            # Parse SGD arguments
            momentum = float(optimizer_config.get('momentum', 0.0))
            nesterov = bool(optimizer_config.get('nesterov', False))

            # Instanciate SGD
            return SGD(
                    learning_rate=lr_scheduler,
                    momentum=momentum,
                    nesterov=nesterov
                )

        if optimizer_config['name'] == 'RMSprop':
            # Parse RMSprop arguments
            rho = float(optimizer_config.get('rho', 0.9))
            momentum = float(optimizer_config.get('momentum', 0.0))
            epsilon = float(optimizer_config.get('epsilon', 1e-07))
            centered = optimizer_config.get('centered', False)

            # Instanciate RMSprop
            return RMSprop(
                    learning_rate=lr_scheduler,
                    rho=rho,
                    momentum=momentum,
                    epsilon=epsilon,
                    centered=centered
                )
        
        # Incorrect optimizer name
        raise Exception('Optimizer name \'%s\' is not valid, should be Adam, SGD or RMSprop' % optimizer_config['name'])

    def create_lr_scheduler(lr_scheduler_config, default_lr):
        """
        Intanciate learing-rate scheduler corresponding to `lr_scheduler_config` dict.
        """
        
        # Empty scheduler and None scheduler
        if not 'name' in lr_scheduler_config.keys():
            return default_lr
        if lr_scheduler_config['name'] in ('None', 'none'):
            return default_lr
        

        if lr_scheduler_config['name'] in ('CosineDecayRestarts', 'CDR'):
            # Parse CosineDecayRestarts arguments
            initial_learning_rate = float(lr_scheduler_config.get('initial_learning_rate', 1e-3))
            first_decay_steps = int(lr_scheduler_config.get('first_decay_steps', 1000))
            t_mul = float(lr_scheduler_config.get('t_mul', 2.0))
            m_mul = float(lr_scheduler_config.get('m_mul', 1.0))
            alpha = float(lr_scheduler_config.get('alpha', 0.0))

            # Instanciate CosineDecayRestarts
            return CosineDecayRestarts(
                    initial_learning_rate=initial_learning_rate,
                    first_decay_steps=first_decay_steps,
                    t_mul=t_mul,
                    m_mul=m_mul,
                    alpha=alpha
                )
        
        if lr_scheduler_config['name'] in ('ExponentialDecay', 'ED'):
            # Parse ExponentialDecay arguments
            initial_learning_rate = float(lr_scheduler_config.get('initial_learning_rate', 1e-3))
            decay_steps = int(lr_scheduler_config.get('decay_steps', 1000))
            decay_rate = float(lr_scheduler_config.get('decay_rate', 0.96))
            staircase = bool(lr_scheduler_config.get('staircase', False))

            # Intanciate ExponentialDecay
            return ExponentialDecay(
                    initial_learning_rate=initial_learning_rate,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    staircase=staircase
                )

        raise Exception('Learning-rate scheduler name \'%s\' is not valid, should be None, CosineDecayRestarts or ExponentialDecay' % lr_scheduler_config['name'])       


    @property
    def model(self):
        return self._model
