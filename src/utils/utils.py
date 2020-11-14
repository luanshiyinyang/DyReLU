import os


def get_exp_num(log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    num = len(os.listdir(log_path))
    return os.path.join(log_path, "exp{}".format(num+1))