import argparse
from pathlib import Path
import logging
import cv2
import numpy as np
import pandas as pd
from sarrarp50.utils import TqdmLoggingHandler, validate_prediction_dir
from sarrarp50.metrics.segmentation import mNSD, mIoU
from sarrarp50.metrics.action_recognition import accuracy, f1k
from tqdm import tqdm 



def main(args):


    logging.basicConfig(encoding='utf-8', level=logging.INFO,
                        handlers=[logging.FileHandler(Path(args.prediction_dir)/"evaluation.log"),
                        TqdmLoggingHandler()])


    try:
        ref_video_dirs, pred_video_dirs = validate_prediction_dir(args.ref_dir, 
                                                                  args.prediction_dir,
                                                                  not args.ignore_actions,
                                                                  not args.ignore_segmentation)
        logging.info('file structure validation passed')
    except (FileNotFoundError, ValueError)as f_e:
        logging.error(f_e)
        return 1

    NSD_tau=10

    
    
    
    results={'video_id': [('_').join(video_dir.name.split('_')[1:]) for video_dir in ref_video_dirs]}
    if not args.ignore_segmentation:
        results['seg_mIoU']=[]
        results['seg_mNSD']=[]
    if not args.ignore_actions:
        results['ar_acc']=[]
        results['ar_f1@10']=[]
    for ref_dir, pred_dir in tqdm(zip(ref_video_dirs, pred_video_dirs),
                                  desc='evalute videos',
                                  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                                  total = len(ref_video_dirs)):
        if not args.ignore_segmentation:
            results['seg_mIoU'].append(np.array(mIoU(pred_dir, ref_dir, n_classes=9)).mean(axis=1).mean().item())
            logging.info(f"{ref_dir.name} segmentaiton mIoU: {results['seg_mIoU'][-1]}")
            results['seg_mNSD'].append(np.array(mNSD(pred_dir, ref_dir,n_classes=9, channel_tau=[NSD_tau]*9)).mean(axis=1).mean().item())
            logging.info(f"{ref_dir.name} segmentation mNSD: {results['seg_mNSD'][-1]}")
        if not args.ignore_actions:
            results['ar_acc'].append(accuracy(pred_dir, ref_dir))
            logging.info(f"{ref_dir.name} action recognition accuracy : {results['ar_acc'][-1]}")
            results['ar_f1@10'].append(f1k(pred_dir, ref_dir, k=10, n_classes=8))
            logging.info(f"{ref_dir.name} action recognition F1@10 : {results['ar_f1@10'][-1]}")

            
    df = pd.DataFrame(results)
    #aggreegate
    m_df = df.mean()


    if not args.ignore_segmentation:
        m_df['segmentation_score']= (df['seg_mIoU'].mean() * df['seg_mNSD'].mean())**(0.5)
    if not args.ignore_actions:
        m_df['action_score']= (df['ar_acc'].mean() * df['ar_f1@10'].mean())**(0.5)
            
    # dump files in the prediction folders
    df.to_csv(Path(args.prediction_dir)/'per_video_results.csv', index=False)
    m_df.to_csv(Path(args.prediction_dir)/'final_results.csv')
    logging.info("per video results")
    logging.info(df)
    logging.info("final results")
    logging.info(m_df)

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument('ref_dir')
    parser.add_argument('prediction_dir')
    parser.add_argument( '--ignore_actions', action='store_true')
    parser.add_argument( '--ignore_segmentation', action='store_true')
    args = parser.parse_args()
    if (args.ignore_actions and args.ignore_segmentation):
        parser.error('--ignore_actions and --ignore_segmentation flags cannot be set at the same time')
    SystemExit(main(args))