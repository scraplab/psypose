import requests
import os
import os.path as osp
import pare
import torch
import zipfile
import shutil

pare_loc = osp.dirname(pare.__file__)
torch_loc = osp.dirname(torch.__file__)

pare_status = osp.exists(osp.join(pare_loc, 'data/dataset_folders'))

zip_address = 'https://www.dropbox.com/s/aeulffqzb3zmh8x/pare-github-data.zip'
zip_path = osp.join(pare_loc, 'pare-github-data.zip')

def install_pare_models():
    os.makedirs(osp.join(pare_loc, 'data/dataset_folders'))
    r = requests.get(zip_address)

    with open(zip_path, 'wb') as f:
        f.write(r.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(pare_loc)

    os.mkdir(osp.join(torch_loc, 'models'))
    shutil.move(osp.join(pare_loc, 'data/yolov3.weights'), osp.join(torch_loc, models, 'yolov3.weights'))

    os.remove(zip_path)




