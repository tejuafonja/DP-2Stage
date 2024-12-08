import time
import os
import pickle
import argparse


def get_time():
    """Get current time"""
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def mkdir(path):
    """make dir if not exist"""
    os.makedirs(path, exist_ok=True)


def save_object(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output, protocol=2)


def load_object(filename):
    with open(filename, "rb") as input:
        return pickle.load(input)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
