import os


def ls(rootpath, suffix=""):
    dirInfo = os.walk(rootpath)
    files = []
    for path, dir_list, file_list in dirInfo:
        for file_name in file_list:
            if file_name.endswith(suffix):
                files.append(os.path.join(path, file_name))
    return files
