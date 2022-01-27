from psypose import utils

import os.path as osp
import os
import shutil
import atexit
import glob
import warnings

from multi_person_tracker import MPT


def extract_tracks(path, image_folder=None):
    needs_parsing = True
    framecount = utils.get_framecount(path)
    image_folder
    if not image_folder:
        warnings.warn(
            "No image folder location chosen. Images will temporarily be saved to this system's root directory",
            UserWarning)
        image_folder = osp.join(utils.PSYPOSE_DATA_DIR, pose.vid_name)

        try:
            os.makedirs(image_folder, exist_ok=False)
        except FileExistsError:
            n_images = len(glob.glob(osp.join(image_folder, '*.png')))
            if n_images != framecount:
                warnings.warn('Video partially parsed to images. Deleting existing images and re-running ffmpeg...',
                              UserWarning)
                shutil.rmtree(path)  # delete the folder
                os.makedirs(image_folder, exist_ok=True)
            elif n_images == framecount:
                needs_parsing = False
                warnings.warn('Video previously parsed. Not re-running ffmpeg.', UserWarning)

    ########## Run person tracking ##########
    if needs_parsing:
        image_folder = utils.video_to_images(path, img_folder=image_folder, return_info=False)
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
    atexit.register(clean_image_folder)  # this will try to delete the folder/images if the script fails for some reason
    shutil.rmtree(image_folder)  # and this deletes the folder if the script is successful

    return tracking_results
