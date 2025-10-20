import os

def create_dir(name: str, path: str, overwrite: bool):
    dir = os.path.join(path, name)
    if overwrite and os.path.exists(dir) and os.path.isdir(dir):
        os.system(f"rm -rf {dir}")

    os.makedirs(dir)

    return dir


