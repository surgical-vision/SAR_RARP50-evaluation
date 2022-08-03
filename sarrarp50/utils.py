import cv2
from pathlib import Path
import numpy as np
import logging 
from tqdm import tqdm 


class TqdmLoggingHandler(logging.StreamHandler):

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)



def video_len(video_path):
    vid = cv2.VideoCapture(str(video_path))
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.release()
    return n_frames

def reference_sequence_len(ref_dir):
    n_action_frames = n_seg_frames = None
    if  (ref_dir/'video_left.avi').is_file():
        n_vid_frames = video_len(ref_dir/'video_left.avi')
        n_action_frames = int((n_vid_frames-1)//6)+1
        n_seg_frames = int((n_vid_frames-1)//60)+1
        print(n_vid_frames, n_action_frames, n_seg_frames)
    elif (ref_dir/'rgb').is_dir():
        n_action_frames = len(list((ref_dir/'rgb').iterdir()))
        print(n_action_frames)
        n_seg_frames = ((n_action_frames-1)//10)+1
    elif (ref_dir/'action_discrete.txt').is_file():# get number of frames from the reference action recognition file
        with open(ref_dir/'action_discrete.txt') as ar:
            n_action_frames = len(ar.readlines())
        n_seg_frames = n_action_frames//6
    elif (ref_dir/'segmentation').is_dir():
        n_seg_frames = len(list((ref_dir/'segmentation').iterdir()))
    
    return n_action_frames, n_seg_frames



def validate_prediction_dir(test_dir, prediction_dir, segmentation=True, actions=True):
    ref_video_dirs = sorted([d for d in Path(test_dir).glob('video_*') if d.is_dir()])
    pred_video_dirs = sorted([d for d in Path(prediction_dir).glob('video_*') if d.is_dir()])

    if len(ref_video_dirs)==0 or len(pred_video_dirs)==0:
        msg=f'{test_dir} or {prediction_dir} do not have child directories using the naming convention video_*'
        raise ValueError(msg)

    if len(ref_video_dirs) != len(pred_video_dirs):
        msg = f'prediction set and test set do not much in size {len(pred_video_dirs)}!={len(ref_video_dirs)}'
        raise FileNotFoundError(msg)
    


    for r_v, p_v in zip (ref_video_dirs, pred_video_dirs):
        if r_v.name != p_v.name:
            raise FileNotFoundError(f'video directory {p_v.name} is not present in the test directory or {r_v.name} is not present in the prediction directory')
            
    # validate action recognition prediciton files
    
    for r_v, p_v in zip (ref_video_dirs, pred_video_dirs):
        
        # find the length of the action labels based on the video file
        len_ref_action, len_ref_seg = reference_sequence_len(r_v)
        
        if segmentation:
            if not (p_v/'segmentation').is_dir():
                raise FileNotFoundError(f"{p_v/'segmentation'} is not a directory")
            if len(list((p_v/'segmentation').iterdir())) != len_ref_seg:
                l1 = [n.name for n in (p_v/'segmentation').iterdir()]
                unique = [n.name for n in (r_v/'segmentation').iterdir() if n.name not in l1]
                print(unique)
                raise FileNotFoundError(f"segmentation file validation failed. {r_v} contained {len_ref_seg} files while {p_v} contained {len(list((p_v/'segmentation').iterdir()))}")

        if actions:                
            try:
                with open(p_v/'action_discrete.txt') as ar:
                    len_pred_action = len(ar.readlines())
            except FileNotFoundError as e:
                raise FileNotFoundError(f"{p_v/'action_discrete.txt'} does not exist")


            if len_pred_action != len_ref_action:
                raise FileNotFoundError(f"segmentation file validation failed. {(r_v/'action_discrete.txt')} contained {len_ref_action} lines while {(p_v/'action_discrete.txt')} contained {len_pred_action} lines")
            
            # do a last check to see if submitted predictions correspond to reference files
            ref_action = np.genfromtxt(r_v/'action_discrete.txt', delimiter=',')
            pred_action = np.genfromtxt(p_v/'action_discrete.txt', delimiter=',')
            if not np.all(ref_action[:,0] == pred_action[:,0]):
                raise ValueError(f"action file validation failed. {(p_v/'action_discrete.txt')} contained predictions that do not correspond to {(r_v/'action_discrete.txt')}")
    return ref_video_dirs, pred_video_dirs