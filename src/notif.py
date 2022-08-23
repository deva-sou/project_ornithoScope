import argparse
import os
import json

from discord import Webhook, RequestsWebhookAdapter, File

argparser = argparse.ArgumentParser(
    description='Plot training loss and validation loss hisotiry.')

argparser.add_argument(
    '-c',
    '--conf',
    default='config/pre_config/ADAM_OCS_v0_full_sampling.json',
    help='Path to config file.')

CODE_QUOTES = '```'
URL = "https://discord.com/api/webhooks/1000055986528198767/sZhup-kBr9wqVxIN4vDb5sRUJ9D-7mXaSeZxWssmprWiMqeC3KbmeNGiDoIuyZU4lgWA"

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
    webhook = Webhook.from_url(URL, adapter=RequestsWebhookAdapter())
    webhook.send(config_path)
    webhook.send(CODE_QUOTES + ''.join(bbox_lines) + CODE_QUOTES)
    webhook.send(CODE_QUOTES + ''.join(class_lines) + CODE_QUOTES)
    webhook.send(file=File(pickle_path + '.jpg'))


if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)