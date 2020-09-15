import os
from pathlib import Path


def mkdir(path, parents=True, exist_ok=True):
    Path(path).mkdir(parents=parents, exist_ok=exist_ok)


def mkdirs(paths, parents=True, exist_ok=True):
    for path in paths:
        mkdir(path, parents=parents, exist_ok=exist_ok)


def ls(rootpath, suffix=""):
    dirInfo = os.walk(rootpath)
    files = []
    for path, dir_list, file_list in dirInfo:
        for file_name in file_list:
            if file_name.endswith(suffix):
                files.append(os.path.join(path, file_name))
    return files


def print_to_file(path, content):
    with open(path, 'w') as f:
        f.write(content)