import os

def get_file_size(filepath):
    return os.path.getsize(filepath)

def get_directory_size(directory):
    return sum(os.path.getsize(os.path.join(directory, f)) for f in os.listdir(directory))
