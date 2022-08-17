import argparse
from itertools import count
import os


argparser = argparse.ArgumentParser(
  description='Generate multiple config files based on a template.')

argparser.add_argument(
  '-c',
  '--conf',
  default='config/benchmark_config/Template.json',
  type=str,
  help='Path to template configuration file.')


def _main_(args):
    config_path = args.conf
    config_folder, config_file = os.path.split(config_path)

    configs = [
        {'BACKEND': 'ResNet50', 'RHO': '224'},
        {'BACKEND': 'ResNet50', 'RHO': '192'},
        {'BACKEND': 'ResNet101', 'RHO': '224'},
        {'BACKEND': 'ResNet101', 'RHO': '192'},

        
    ]

    lines = ''
    with open(config_path, 'r') as config_buffer:
        for line in config_buffer.readlines():
            lines += line
        config_buffer.close()
    
    for config in configs:
        current_lines = lines
        for key in config:
            current_lines = current_lines.replace(key, config[key])
        
        current_config_file = '-'.join(config.values()) + '.json'
        with open(os.path.join(config_folder, current_config_file), 'w') as config_buffer:
            config_buffer.write(current_lines)
            config_buffer.close()
        
        




if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)