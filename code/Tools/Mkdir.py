from pathlib import Path


def mkdir(path, parents=True, exist_ok=True):
    Path(path).mkdir(parents=parents, exist_ok=exist_ok)


def mkdirs(paths, parents=True, exist_ok=True):
    for path in paths:
        mkdir(path, parents=parents, exist_ok=exist_ok)