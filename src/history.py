import matplotlib.pyplot as plt
import pickle
import argparse
import itertools


argparser = argparse.ArgumentParser(
    description='Plot training loss and validation loss hisotiry.')

argparser.add_argument(
    '-p',
    '--pick',
    default='data/pickles/history/history_data/saved_weights/new_weights/test#_bestLoss.h5.p',
    help='Path to pickle file.')


def _main_(args):
    pickle_file_path = args.pick

    plt.figure('Histories')

    paths = ['data/pickles/history/history_data/saved_weights/new_weights/test#_bestLoss.h5.p',
            'data/pickles/history/history_data/saved_weights/new_weights/MobileNet_caped300_data_aug_v0_$#_bestLoss.h5.p']
            
    patches = [{'#' : [str(i) for i in range(1, 15)]},
                {'#' : ['0', '1', '2', '3'], '$' : ['ADAM', 'RMS']}]

    it = 0
    for path, patch in zip(paths, patches):
        for vals in itertools.product(*patch.values()):
            current_path = path
            for val, key in zip(vals, patch.keys()):
                current_path = val.join(current_path.split(key))

            try:
                with open(current_path, 'rb') as input_file:
                    history = pickle.load(input_file)
            except FileNotFoundError:
                print(current_path, "not found.")
                continue
                
            loss = history['loss']
            val_loss = history['val_loss']

            steps = [i for i in range(len(loss))]

            plt.plot(steps, val_loss, linestyle='-' if it < 10 else '--', label=vals)
            it += 1
    
    plt.legend()
    plt.show()


if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)



# import matplotlib.pyplot as plt
# import pickle
# import argparse


# argparser = argparse.ArgumentParser(
#     description='Plot training loss and validation loss hisotiry.')

# argparser.add_argument(
#     '-p',
#     '--pick',
#     default='data/pickles/history/history_data/saved_weights/new_weights/MobileNet_caped300_data_aug_v0_$#_bestLoss.h5.p',
#     help='Path to pickle file.')


# def _main_(args):
#     pickle_file_path = args.pick

#     patches = {'$' : ['ADAM', 'RMS'], '#' : ['0', '1', '2', '3', '4']}

#     loss = []
#     val_loss = []
#     steps = []
#     for key in patches.keys():
#         for val in patches[key]:
#             current path = val.join(pickle_file_path.split(key))

#             with open(pickle_file_path, 'rb') as input_file:
#                 history = pickle.load(input_file)
                
#             loss.append(history['loss'])
#             val_loss.append(history['val_loss'])

#             steps.append([i for i in range(len(loss))])

#     plt.figure('Histories')
#     plt.plot(steps, loss)
#     plt.plot(steps, val_loss)
#     plt.show()


# if __name__ == '__main__':
#     _args = argparser.parse_args()
#     _main_(_args)