from psypose import utils
from psypose.utils import PSYPOSE_DATA_DIR

import os.path as osp
import shutil
import atexit

from multi_person_tracker import MPT

def extract_tracks(path, image_folder=None):

    if not image_folder:
        image_folder = osp.join(PSYPOSE_DATA_DIR, pose.vid_name)

    ########## Run person tracking ##########
    image_folder, num_frames, img_shape = utils.video_to_images(vid_path, img_folder=image_folder, return_info=True)

    mpt = MPT(
        display=False,
        detector_type='yolo',
        batch_size=10,
        yolo_img_size=416
    )

    def clean_image_folder():
        if osp.exists(image_folder) and osp.isdir(image_folder):
            shutil.rmtree(image_folder)

    tracking_results = mpt(image_folder)

    # delete image folder
    atexit.register(clean_image_folder)
    shutil.rmtree(image_folder)

    return tracking_results


