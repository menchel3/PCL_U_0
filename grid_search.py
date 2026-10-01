"""Run every configuration in a PCL-U config file exactly once."""

from __future__ import print_function

import argparse
import ast
import itertools
import os
import shlex
import subprocess
import sys


TRAIN_SCRIPT = 'train.py'


def load_config(path):
    """Load ``key=value`` entries using Python literal syntax."""
    config = {}
    with open(path, 'r') as config_file:
        for line_number, line in enumerate(config_file, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, separator, value = line.partition('=')
            if not separator:
                raise ValueError(
                    'Invalid configuration line %d in %s' %
                    (line_number, path))
            key = key.strip()
            if key in config:
                raise ValueError('Duplicate configuration key: %s' % key)
            config[key] = ast.literal_eval(value.strip())
    return config


def as_grid(config):
    return dict(
        (key, value if isinstance(value, list) else [value])
        for key, value in config.items())


def config_id(config):
    return ','.join(
        '%s:%s' % (key, str(config[key])) for key in sorted(config)).lower()


def read_used_config_ids(path):
    with open(path, 'r') as used_file:
        return set(line.strip() for line in used_file if line.strip())


def iter_grid(configs):
    """Yield the Cartesian product in a stable, reproducible order."""
    keys = sorted(configs)
    for values in itertools.product(*(configs[key] for key in keys)):
        yield dict(zip(keys, values))


def format_command(command):
    return ' '.join(shlex.quote(str(part)) for part in command)


def validate_data_files(cfg):
    datadir = cfg.get('datadir')
    if not datadir:
        raise ValueError('Configuration does not specify datadir')

    for key in ('dataform', 'data_test'):
        filename = cfg.get(key)
        if filename:
            path = os.path.join(datadir, filename)
            if not os.path.isfile(path):
                raise IOError('Configured data file does not exist: %s' % path)


def run(config_file):
    configs = as_grid(load_config(config_file))
    if not os.path.isfile(TRAIN_SCRIPT):
        raise IOError('Training script does not exist: %s' % TRAIN_SCRIPT)

    if len(configs.get('outdir', [])) != 1:
        raise ValueError('Grid configs must specify exactly one outdir')

    outdir = configs['outdir'][0]
    used_cfg_file = os.path.join(outdir, 'used_configs.txt')
    used_cfgs = (read_used_config_ids(used_cfg_file)
                 if os.path.isfile(used_cfg_file) else set())
    grid = list(iter_grid(configs))

    print('Grid contains %d configuration(s).' % len(grid))
    for index, cfg in enumerate(grid, 1):
        cfg_id = config_id(cfg)
        if cfg_id in used_cfgs:
            print('[%d/%d] Configuration already completed; skipping.' %
                  (index, len(grid)))
            continue

        command = [sys.executable, TRAIN_SCRIPT]
        for key in sorted(cfg):
            command.extend(['--%s' % key, str(cfg[key])])

        print('[%d/%d] %s' % (index, len(grid), format_command(command)))
        validate_data_files(cfg)
        os.makedirs(outdir, exist_ok=True)
        return_code = subprocess.call(command)
        if return_code != 0:
            raise RuntimeError(
                'Training failed with return code %d' % return_code)

        with open(used_cfg_file, 'a') as used_file:
            used_file.write('%s\n' % cfg_id)
        used_cfgs.add(cfg_id)


def main():
    parser = argparse.ArgumentParser(
        description='Run the full Cartesian grid from a PCL-U config exactly once.')
    parser.add_argument('config_file')
    args = parser.parse_args()
    run(args.config_file)


if __name__ == '__main__':
    main()
