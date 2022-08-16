import argparse
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
    """
    Generate multiple config files based on a template.
    """
    config_path = args.conf
    config_folder, config_file = os.path.split(config_path)

    # Teamplate keyword to change
    configs = [
        {'BACKEND': 'MobileNetV2', 'ALPHA': '0.35', 'RHO': '192'},
        {'BACKEND': 'MobileNetV2', 'ALPHA': '0.50', 'RHO': '192'},
        {'BACKEND': 'MobileNetV2', 'ALPHA': '0.75', 'RHO': '192'},
        {'BACKEND': 'MobileNetV2', 'ALPHA': '1.00', 'RHO': '192'},
        {'BACKEND': 'MobileNetV2', 'ALPHA': '1.30', 'RHO': '192'},
        {'BACKEND': 'MobileNetV2', 'ALPHA': '1.40', 'RHO': '192'},
    ]

    lines = ''
    # Read file in lines
    with open(config_path, 'r') as config_buffer:
        for line in config_buffer.readlines():
            lines += line
        config_buffer.close()
    
    for config in configs:
        # Replace key words by its value in the current config
        current_lines = lines
        for key in config:
            current_lines = current_lines.replace(key, config[key])
        
        # Write output file
        current_config_file = '-'.join(config.values()) + '.json'
        with open(os.path.join(config_folder, current_config_file), 'w') as config_buffer:
            config_buffer.write(current_lines)
            config_buffer.close()


if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)