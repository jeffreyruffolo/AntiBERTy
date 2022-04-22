import os
import urllib.request

import antiberty
from antiberty.utils.general import exists


def get_weights(dest="trained_models/"):
    project_path = os.path.dirname(os.path.realpath(antiberty.__file__))
    dest = os.path.join(project_path, dest)

    if exists(dest):
        print("Found existing weights at {}".format(dest))
        return dest

    url = "https://data.graylab.jhu.edu/AntiBERTy_md_smooth.tar.gz"
    tarfile = os.path.join(dest, "AntiBERTy_md_smooth.tar.gz")
    urllib.request.urlretrieve(url, dest)

    if not exists(tarfile):
        raise Exception("Could not download weights from {}".format(url))

    os.system(f"tar -xzf {tarfile} -C {dest}")

    return dest