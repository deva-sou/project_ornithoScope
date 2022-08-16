

import matplotlib.pyplot as plt
import pickle
import argparse
import itertools


argparser = argparse.ArgumentParser(
    description='Plot training loss and validation loss hisotiry.')

argparser.add_argument(
    '-p',
    '--pick',
    default='data/pickles/history/history_data/saved_weights/new_weights/lr_fixé_all_data_Adam_OCS_new_data_cap200_v1_new_valid_and_train_data_bestLoss.h5.p',
    help='Path to pickle file.')


def _main_(args):
    pickle_file_path = args.pick

    with open(pickle_file_path, 'rb') as input_file:
            history = pickle.load(input_file)
        
    loss = history['loss']
    for i in range(len(loss)):
        if loss[i]>40:
            loss[i]=40
        loss = history['loss']
    
    val_loss = history['val_loss']
    for i in range(len(val_loss)):
        if val_loss[i]>40:
            val_loss[i]=40
        
    


    steps = [i for i in range(len(loss))]

    plt.plot(steps, loss, label='Training loss')
    plt.plot(steps, val_loss, label='Validation loss')
    
    plt.legend()
    plt.show()


if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)