import argparse
import os
import json
import sys
from discord import Webhook, File
from discord import RequestsWebhookAdapter


#ce code permet d'avoir une notif discord quand on lance un multi train à partir d'un tmux

argparser = argparse.ArgumentParser(
    description='Plot training loss and validation loss hisotiry.')

argparser.add_argument(
    '-c',
    '--conf',
    default='/home/acarlier/code/project_ornithoScope/src/config/benchmark_config/Mobilenet_sampling_valid_train_inat.json',
    help='Path to config file.')

CODE_QUOTES = '```'

def _main_(args):
    config_path = args.conf

    # Load config file as a dict
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    # Get evaluate outfile lines
    lines = [line for line in open(config_path + '.log', 'r').readlines()]
    class_line = lines.index('Class metrics:\n')
    bbox_line = lines.index('BBox metrics:\n')
    class_lines = lines[class_line:bbox_line]
    bbox_lines = lines[bbox_line:]
    
    # Get history output image
    root, ext = os.path.splitext(config['train']['saved_weights_name'])
    saved_pickle_path = config['data']['saved_pickles_path']
    pickle_path = f'{saved_pickle_path}/history/history_{root}_bestLoss{ext}.p'

    # Send message with the evaluate results and history image
    webhook = Webhook.from_url(
            "https://discordapp.com/api/webhooks/1009005803350540288/jJN-0ZFmrvuW3erbwuiOgqa0GmoHW7upHl4Zu2d-vppH2wrdk51Brp6uhZ5qZ8jTLUPu",
            adapter=RequestsWebhookAdapter())
    webhook.send(config_path)
    webhook.send(CODE_QUOTES + ''.join(bbox_lines) + CODE_QUOTES)
    webhook.send(CODE_QUOTES + ''.join(class_lines) + CODE_QUOTES)
    webhook.send(file=File(pickle_path + '.jpg'))


if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)