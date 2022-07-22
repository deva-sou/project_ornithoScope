import matplotlib.pyplot as plt
import pickle
import argparse


argparser = argparse.ArgumentParser(
    description='Plot training loss and validation loss hisotiry.')

argparser.add_argument(
    '-p',
    '--pick',
    default='data/pickles/history/history_data/saved_weights/new_weights/MobileNet_caped300_data_aug_v0_RMS2_bestLoss.h5.p',
    help='Path to pickle file.')


def _main_(args):
    pickle_file_path = args.pick
    with open(pickle_file_path, 'rb') as input_file:
        history = pickle.load(input_file)
        
    loss = history['loss']
    val_loss = history['val_loss']

    steps = [i for i in range(len(loss))]

    plt.figure('Histories')
    plt.plot(steps, loss)
    plt.plot(steps, val_loss)
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