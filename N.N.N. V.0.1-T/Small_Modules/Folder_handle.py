import os
#Given folder, return name and path of each file inside
def check_in_folder(folder):
    if folder is None:
        return -1
    return os.listdir(folder)