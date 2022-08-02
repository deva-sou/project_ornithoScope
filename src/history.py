import matplotlib.pyplot as plt
import pickle
import argparse
import itertools
import os
import csv

argparser = argparse.ArgumentParser(
    description='Plot training loss and validation loss hisotiry.')

argparser.add_argument(
    '-p',
    '--pick',
    default='data/pickles/history/history_data/saved_weights/new_weights/Adam_OCS_batchsize8_1e-5à1e-2_bestLoss.h5.p', #insérer uh path issu des pickles
    help='Path to pickle file.')

#A CHANGER: NE PAS PLOTER TOUTES LES VALEURS DE LOSS DONT L ORDONNÉE EST AU DESSUS DE 300 SINON NIQUE L'ECHELLE

def _main_(args):
    pickle_file_path = args.pick

    with open(pickle_file_path, 'rb') as input_file:
            history = pickle.load(input_file)
        
    loss = history['loss']
    val_loss = history['val_loss']

    steps = [i for i in range(len(loss))]
    #on sauvegarde les graphes car sinon ne s'affiche pas
    plt.plot(steps, loss, label='loss')
    plt.plot(steps, val_loss, label='val_loss')
    plt.legend(["loss", "val_loss"])


    writepath = 'plot_lucien/Adam_OCS_%d.png'

    id = 0
    while os.path.exists(writepath % id):
        id += 1
    plt.savefig(writepath % id) #à modififier selon les documents
    plt.show()
    

if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)


