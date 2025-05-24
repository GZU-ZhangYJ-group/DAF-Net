# import logging
import time
import numpy as np


class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr

#
# initialized_logger = {}
# def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
#     """Get the root logger.
#
#     The logger will be initialized if it has not been initialized. By default a
#     StreamHandler will be added. If `log_file` is specified, a FileHandler will
#     also be added.
#
#     Args:
#         logger_name (str): root logger name. Default: 'basicsr'.
#         log_file (str | None): The log filename. If specified, a FileHandler
#             will be added to the root logger.
#         log_level (int): The root logger level. Note that only the process of
#             rank 0 is affected, while other processes will set the level to
#             "Error" and be silent most of the time.
#
#     Returns:
#         logging.Logger: The root logger.
#     """
#     logger = logging.getLogger(logger_name)
#     # if the logger has been initialized, just return it
#     if logger_name in initialized_logger:
#         return logger
#
#     format_str = '%(asctime)s %(levelname)s: %(message)s'
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(logging.Formatter(format_str))
#     logger.addHandler(stream_handler)
#     logger.propagate = False
#     # rank, _ = get_dist_info()
#     # if rank != 0:
#     #     logger.setLevel('ERROR')
#     # elif log_file is not None:
#     if log_file is not None:
#         logger.setLevel(log_level)
#         # add file handler
#         file_handler = logging.FileHandler(log_file, 'w')
#         file_handler.setFormatter(logging.Formatter(format_str))
#         file_handler.setLevel(log_level)
#         logger.addHandler(file_handler)
#     initialized_logger[logger_name] = True
#     return logger
#
