import matplotlib
from audio_to_multiple_pose_gan.config import create_parser, get_config
from common.pose_logic_lib import translate_keypoints
matplotlib.use('Agg')
import os
import argparse
import pandas as pd
import numpy as np
from audio_to_multiple_pose_gan.dataset import generate_batch, get_processor
from tqdm import tqdm
import logging
from logging import getLogger
logging.basicConfig()
logger = getLogger("model.logger")
from audio_to_multiple_pose_gan.static_model_factory import Audio2PoseGANS
import torch
from common.consts import RIGHT_BODY_KEYPOINTS, LEFT_BODY_KEYPOINTS, LEFT_HAND_KEYPOINTS, \
    RIGHT_HAND_KEYPOINTS, POSE_SAMPLE_SHAPE, G_SCOPE, D_SCOPE, SR, FRAMES_PER_SAMPLE, AUDIO_SHAPE

device = "cuda" if torch.cuda.is_available() else "cpu"
def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default="~/Desktop/AI-Experiments/speechandtext2gesture/Gestures/train.csv", help="DATAPATH")
    parser.add_argument('--speaker', '-s', type=int, default="shelly")
    parser.add_argument('--output_path', '-op', type=int, default="~/Desktop/AI-Experiments/speechandtext2gesture")
    args = parser.parse_args()
    return args

def predict_df(df, cfg, shift_pred=(900, 290), shift_gt=(1900, 280)):
        keypoints1_list = []
        keypoints2_list = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                (keypoints1, keypoints2) = predict_row(row, cfg, shift_pred=shift_pred,shift_gt=shift_gt)
            except Exception as e:
                logger.exception(e)
                continue

            keypoints1_list.append(keypoints1)
            keypoints2_list.append(keypoints2)

        return np.array(keypoints1_list), np.array(keypoints2_list)

def predict_row(row, cfg, shift_pred=(0, 0), shift_gt=(0, 0)):
        process_row, decode_pose = get_processor(cfg)
        x, y = process_row(row)
        """
        これがまじで意味わからん
        """
        # pred_pose = G_model(X)をしているに等しい
        x = torch.tensor(x).float().to(device)
        pred_pose = G_model()
        return post_process(pred_pose, y, shift_gt, shift_pred, decode_pose, row)

def post_process(res, y, shift_gt, shift_pred, decode_pose, row):
    return post_process_output(res[0], decode_pose, shift_pred, row['speaker']), post_process_output(y, decode_pose, shift_gt, row['speaker'])

def post_process_output(res, decode_pose, shift, speaker):
    return decode_pose(res, shift, speaker) #decode_pose(pose_Y, shift = [0,0], speaker="shelly").shape = (64, 2, 49) Y.shape (64, 98)が(64, 2, 49)に変換された。

def save_prediction_video(df, keypoints1_list, keypoints2_list, args.output_path):
    #videoをsaveする関数を組む

if __name__ == "__main__":
    args = create_args()
    G_model = Audio2PoseGANS(1, POSE_SAMPLE_SHAPE[-1]).to(device)
    data_csv: str = args.dataset
    df = pd.read_csv(data_csv)
    cfg: dict = {"processor": "audio_to_pose", "input_shape": [None, AUDIO_SHAPE]} #processorはaudio_to_pose_inferenceかもしれない
    keypoints1_list, keypoints2_list = predict_df(df, cfg, [0,0], [0,0])
    keypoints1_list = translate_keypoints(keypoints1_list, [900, 290])
    keypoints2_list = translate_keypoints(keypoints2_list, [1900, 280])

    save_prediction_video(df, keypoints1_list, keypoints2_list, args.output_path)