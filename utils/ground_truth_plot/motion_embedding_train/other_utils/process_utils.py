#!/bin/env python
from __future__ import print_function

import multiprocessing
import os, logging, time, signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

time_infinite = 10 ** 5

try:
    default_num_workers = multiprocessing.cpu_count() * 2
except:
    default_num_workers = 12


def run_worker_test(delay):
    print("In a worker process", os.getpid())
    time.sleep(delay)
    print('Done worker process')
    return 'Hello There'


def pool_map(task_func, tasks, num_workers=default_num_workers, timeout=None):

    time_infinite = 10 ** 5

    func_name = task_func.__name__

    if timeout is None:
        timeout = time_infinite

    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    pool = multiprocessing.Pool(num_workers)
    signal.signal(signal.SIGINT, original_sigint_handler)

    results = None
    try:
        logger.info("{}: Starting {} jobs...".format(func_name, len(tasks)))
        res = pool.map_async(task_func, tasks)
        logger.info("{}: Waiting for results...".format(func_name))
        results = res.get(timeout)  # Without the timeout this blocking call ignores all signals.
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        logger.info("{}: Normal Termination".format(func_name))
        pool.close()
    pool.join()
    return results


if __name__ == "__main__":
    results = pool_map(run_worker_test, [1.2, 1, 0.5, 0.4, 0.22, 2, 3], 3)
    for r in results:
        print(r)
