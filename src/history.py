from copyreg import pickle
import matplotlib.pyplot as plt
import argparse
import json
import os
import pickle


argparser = argparse.ArgumentParser(
    description='Plot training loss and validation loss hisotiry.')

argparser.add_argument(
    '-c',
    '--conf',
    default='config/pre_config/ADAM_OCS_v0_full_sampling.json',
    help='Path to config file.')


def _main_(args):
    config_path = args.conf

    # Load config file as a dict
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    
    # Get the pickle history file based on config file
    root, ext = os.path.splitext(config['train']['saved_weights_name'])
    saved_pickle_path = config['data']['saved_pickles_path']
    pickle_path = f'{saved_pickle_path}/history/history_{root}_bestLoss{ext}.p'

    # Load history pickle
    with open(pickle_path, 'rb') as pickle_buffer:
        history = pickle.load(pickle_buffer)
        
    # Extract losses
    loss = history['loss']
    val_loss = history['val_loss']

    steps = [i for i in range(len(loss))]

    # Plot curves
    plt.plot(steps, loss, label='Training loss')
    plt.plot(steps, val_loss, label='Validation loss')
    
    # Modify figure params
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis((5, xmax, 0, val_loss[5]))
    plt.legend()

    # Save figure
    plt.savefig(pickle_path + '.jpg')


if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)
