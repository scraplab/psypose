from psypose import utils
from psypose.utils import PSYPOSE_DATA_DIR

import os.path as osp
import os
import shutil
import atexit
import glob
import warnings

from multi_person_tracker import MPT

def extract_tracks(path):

    video_needs_parsing = True
    if not image_folder:
        image_folder = osp.join(PSYPOSE_DATA_DIR, osp.basename(path)[:-4])
        try:
            os.makedirs(image_folder, exist_ok=False)
        except FileExistsError:
            expected_frames = utils.get_framecount(path)
            present_frames = len(glob.glob(osp.join(image_folder, '*.png')))
            if (expected_frames != present_frames) and present_frames:
                warnings.warn('Video only partially parsed to images. Deleting existing images and re-running ffmpeg...', UserWarning)
                shutil.rmtree(image_folder)
                os.makedirs(image_folder)
            elif (expected_frames != present_frames) and not present_frames:
                os.makedirs(image_folder, exist_ok=True)
            elif expected_frames == present_frames:
                video_needs_parsing=False

    ########## Run person tracking ##########
    if video_needs_parsing:
        image_folder, num_frames, img_shape = utils.video_to_images(path, img_folder=image_folder, return_info=True)
    else:
        num_frames, img_shape = utils.get_n_frames(path), utils.get_image_shape(path)

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
    atexit.register(clean_image_folder) # this will try to delete the folder/images if the script fails for some reason
    shutil.rmtree(image_folder) # and this deletes the folder if the script is successful

    return tracking_results


