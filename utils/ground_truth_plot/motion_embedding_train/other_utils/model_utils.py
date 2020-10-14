import os, enum, glob
import logging

from src import paths
import bash_utils as bash

logger = logging.getLogger(__name__)


class GanSwitch(enum.Enum):
    Discriminator = 0
    Generator = 1


def get_exp_name():
    return os.path.basename(os.path.abspath('.'))


def setup_dirs():
    bash.create_dir(paths.results_base_dir)
    bash.create_dir(paths.logs_base_dir)
    bash.create_dir(paths.saved_weights_dir)
    bash.create_dir(paths.all_weights_dir)
    bash.create_dir(paths.results_base_dir)
    bash.create_dir(paths.recon_dir)
    bash.create_dir(paths.gen_dir)
    bash.create_dir(paths.hist_dir)
    bash.create_dir(paths.temp_dir)
    bash.create_dir(paths.nn_dir)


def copy_weights_to_saved_dir(iter_no, iter_or_best='iter'):
    """
    Copied the Weights from weights/ to pretrained_weights/ given iteration number and label: 'all' or 'best'
    """
    files = os.listdir(paths.all_weights_dir)
    match_substr = '%s-%d' % (iter_or_best, iter_no)
    files = [f for f in files if match_substr in f]
    logger.info("Copying files:")
    for f in files:
        old_file_path = os.path.join(paths.all_weights_dir, f)
        cmd = 'cp -f %s %s' % (old_file_path, paths.saved_weights_dir)
        bash.exec_cmd(cmd)
    logger.info('Weights saved successfully...')


def get_most_recent_iter_no(dir_name='all', iter_or_best='iter', k=2):
    pattern = '*encoder_%s*.data*' % iter_or_best
    weights_dir = paths.weights_dir_paths[dir_name]
    filenames = glob.glob1(weights_dir, pattern)
    filepaths = [os.path.join(weights_dir, filename) for filename in filenames]
    files = sorted(filepaths, key=os.path.getmtime)
    # for f in files: print f
    if len(files) <= 1:
        msg = 'Fetch Error: {} files in {}'.format(len(files), weights_dir)
        logger.error(msg)
        raise Exception(msg)
    f = files[-k]
    iter_no = int(f[f.index('-') + 1:f.rindex('.')])
    return iter_no


def get_largest_iter_no(iter_or_best='best', dir='weights'):
    """
    Gets the most recent iteration number from weights/ dir of given label: ('best' or 'all')
    """
    files = os.listdir(dir)
    files = [f for f in files if iter_or_best in f]
    numbers = {int(f[f.index('-') + 1:f.index('.')]) for f in files}
    return max(numbers)


def copy_latest_weights_to_saved_dir(iter_or_best='best'):
    logger.info('Trying to copy latest weights...')
    latest_iter = get_most_recent_iter_no('all', iter_or_best)
    copy_weights_to_saved_dir(latest_iter, iter_or_best)
    return latest_iter


if __name__ == '__main__':
    it = get_most_recent_iter_no('all', 'iter', k=2)
    print it
