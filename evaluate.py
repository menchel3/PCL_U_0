
#coding=utf-8
import sys
import os

import pickle as pickle

from pcl_u.logger import Logger as Log
Log.VERBOSE = True

import pcl_u.evaluation as evaluation
from pcl_u.plotting import plot_evaluation_bin, plot_evaluation_cont
from grid_search import load_config

def evaluate(config_file, overwrite=False, filters=None):

    if not os.path.isfile(config_file):
        raise Exception('Could not find config file at path: %s' % config_file)

    cfg = load_config(config_file)

    output_dir = cfg['outdir']

    if not os.path.isdir(output_dir):
        raise Exception('Could not find output at path: %s' % output_dir)

    data_train = cfg['datadir']+'/'+cfg['dataform']
    data_test = cfg['datadir']+'/'+cfg['data_test']
    outcome = cfg['outcome'].lower()
    if outcome not in ('binary', 'continuous'):
        raise ValueError("outcome must be 'binary' or 'continuous': %s" % outcome)
    binary = outcome == 'binary'

    # Evaluate results
    eval_path = '%s/evaluation.npz' % output_dir
    if overwrite or (not os.path.isfile(eval_path)):
        eval_results, configs = evaluation.evaluate(output_dir,
                                data_path_train=data_train,
                                data_path_test=data_test,
                                binary=binary)
        # Save evaluation
        pickle.dump((eval_results, configs), open(eval_path, "wb"))
    else:
        if Log.VERBOSE:
            print ('Loading evaluation results from %s...' % eval_path)
        # Load evaluation
        eval_results, configs = pickle.load(open(eval_path, "rb"))

    
    if binary:
        plot_evaluation_bin(eval_results, configs, output_dir, data_train, data_test, filters)
    else:
        plot_evaluation_cont(eval_results, configs, output_dir, data_train, data_test, filters)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print ('Usage: python evaluate.py <config_file> <overwrite (default 0)> <filters (optional)>')
    else:
        config_file = sys.argv[1]

        overwrite = False
        if len(sys.argv)>2 and sys.argv[2] == '1':
            overwrite = True

        filters = None
        if len(sys.argv)>3:
            filters = eval(sys.argv[3])

        evaluate(config_file, overwrite, filters=filters)
