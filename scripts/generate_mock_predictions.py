import argparse
from pathlib import Path
import logging
import shutil
import numpy as np
import csv
import cv2
from sarrarp50.utils import TqdmLoggingHandler
from tqdm import tqdm



def mock_seg_img(msg, height, width):
    img = np.zeros((height,width,3), dtype=np.uint8)
    color = np.random.randint(0,10)#*10
    cv2.putText(img,
                f"{msg}", 
                (50,int(height/2)),# bottom left corner 
                cv2.FONT_HERSHEY_SIMPLEX, 
                5, # fond scale
                [color]*3,# color
                5, # thickness
                2) # linetype
    return img


def main(args):
    test_root_dir = Path(args.test_dir)
    test_video_dirs = [n for n in test_root_dir.iterdir() if n.is_dir()]
    dst_root_dir = Path(args.prediction_dir)
    logging.basicConfig(encoding='utf-8', level=logging.INFO,
                        handlers=[logging.FileHandler("mock_predictions.log"),
                        TqdmLoggingHandler()])

    try:
        dst_root_dir.mkdir()
    except FileExistsError as e:
        if not args.overwrite:
            logging.error(f"{dst_root_dir.resolve()} exist, please delete it  manually or call this script using thhe --overwirte flag to replace previous predictions")
            return 1
        try:
            shutil.rmtree(dst_root_dir)
        except NotADirectoryError as e:
            logging.error(f"{dst_root_dir.resolve()} is not a directory, please remove it manually")
            return 1

    # creating the output directories
    for video_dir in test_video_dirs:
        (dst_root_dir/video_dir.name/'segmentation').mkdir(parents=True)

    # creating mock action recognition predictions:
    for video_dir in tqdm(test_video_dirs, desc='generate mock action predicitons', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        rgb_ids = sorted([img_p.stem for img_p in (video_dir/'rgb').iterdir()])
        mock_action_labels = np.random.randint(8, size=len(rgb_ids))
        with open(dst_root_dir/video_dir.name/'action_discrete.txt', 'w') as f:
            csv_writer = csv.writer(f)
            for img_id, label in zip (rgb_ids, mock_action_labels):
                csv_writer.writerow([img_id, label])
            logging.info(f" generate mock action recognition predicitons for {video_dir.name}, at {dst_root_dir/video_dir.name/'action_discrete.txt'}")

    # create mock 3 channel 1920x1080 segmentation images

    for video_dir in tqdm(test_video_dirs, desc='generate mock segmentation predictions', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        rgb_names = sorted([img_p.name for img_p in (video_dir/'rgb').iterdir()])[::10]# processing one every 10 rgb images(10Hz) is equivilant to processing the video at 1Hz
        for name in tqdm(rgb_names, desc=f'segmentation files for {video_dir.name}', leave=False, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            cv2.imwrite(str(dst_root_dir/video_dir.name/'segmentation'/name), 
                        mock_seg_img(name, 1080, 1920))
        logging.info(f"generate mock segmentation predicitons for {video_dir.name}, under {dst_root_dir/video_dir.name/'segmentation'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', help='root_directory of the test set')
    parser.add_argument('prediction_dir', help='directory to store predictions')
    parser.add_argument('-o','--overwrite', help='overwrite the predictions of prediction_dir', action='store_true')
    
    SystemExit(main(parser.parse_args()))