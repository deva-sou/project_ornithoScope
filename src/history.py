import matplotlib.pyplot as plt
import pickle


argparser = argparse.ArgumentParser(
    description='Plot training loss and validation loss hisotiry.')

argparser.add_argument(
    '-p',
    '--pick',
    default='data/pickles/history/history_data/saved_weights/new_weights/MobileNet_caped300_data_aug_v0_ADAM_bestLoss.h5.p',
    help='Path to pickle file.')


def _main_(args):
    pickle_file_path = args.pick
    with open(pickle_file_path, 'rb') as input_file:
        history = pickle.load(input_file)
        
    loss = histroy['loss']
    val_loss = hisotry['val_loss']

    steps = [i for i in range(len(loss))]

    plt.figure('Histories')
    plt.plot(steps, loss)
    plt.plot(steps, val_loss)
    plt.show()


if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)