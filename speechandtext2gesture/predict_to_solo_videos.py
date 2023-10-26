import matplotlib
from common.pose_logic_lib import translate_keypoints
from common.pose_plot_lib import save_video
matplotlib.use('Agg')
import os
import argparse
import pandas as pd
import numpy as np
from audio_to_multiple_pose_gan.dataset import get_processor
from tqdm import tqdm
import logging
from logging import getLogger
logging.basicConfig()
logger = getLogger("model.logger")
from audio_to_multiple_pose_gan.static_model_factory import Audio2PoseGANS
from audio_to_multiple_pose_gan.static_model_factory import Audio2PoseGANSTransformer
import torch
from common.consts import POSE_SAMPLE_SHAPE, AUDIO_SHAPE
import warnings
warnings.simplefilter('ignore')
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default="~/Desktop/AI-Experiments/speechandtext2gesture/Gestures/train.csv", help="DATAPATH")
    parser.add_argument('--speaker', '-s', type=str, default="shelly")
    parser.add_argument('--output_path', '-op', type=str, default="pred_videos")
    parser.add_argument('--param_path', '-p', type = str, default = "params/G_transformermodel_last.pth")
    args = parser.parse_args()
    return args
@torch.no_grad
def predict_df(G_model, df, cfg, shift_pred=(900, 290), shift_gt=(1900, 280), limit = 1e9):
        keypoints1_list = []
        keypoints2_list = []

        for i, row in tqdm(df.iterrows(), total=limit):
            if limit < i:
                 break
            try:
                (keypoints1, keypoints2) = predict_row(G_model, row, cfg, shift_pred=shift_pred,shift_gt=shift_gt)
            except Exception as e:
                logger.exception(e)
                continue
            print(keypoints1.shape)
            keypoints1_list.append(keypoints1)
            keypoints2_list.append(keypoints2)
        return np.array(keypoints1_list), np.array(keypoints2_list)
@torch.no_grad
def predict_row(G_model, row, cfg, shift_pred=(0, 0), shift_gt=(0, 0)):
        process_row, decode_pose = get_processor(cfg)
        x, y = process_row(row)
        """
        これがまじで意味わからん
        """
        # pred_pose = G_model(X)をしているに等しい
        x = torch.tensor(x).float().to(device)
        x = x.unsqueeze(0)
        pred_pose = G_model(x)
        pred_pose = pred_pose.cpu().detach().numpy()
        return post_process(pred_pose, y, shift_gt, shift_pred, decode_pose, row)
@torch.no_grad
def post_process(res, y, shift_gt, shift_pred, decode_pose, row):
    return post_process_output(res[0], decode_pose, shift_pred, row['speaker']), post_process_output(y, decode_pose, shift_gt, row['speaker'])
@torch.no_grad
def post_process_output(res, decode_pose, shift, speaker):
    return decode_pose(res, shift, speaker) #decode_pose(pose_Y, shift = [0,0], speaker="shelly").shape = (64, 2, 49) Y.shape (64, 98)が(64, 2, 49)に変換された。
@torch.no_grad
def save_prediction_video(df, keypoints_pred, output_path, limit = None):
    #videoをsaveする関数を組む
    if limit == None:
        limit = len(df)
    for i in tqdm(range(min(len(df), limit))):
        try:
            row = df.iloc[i]
            keypoints1 = keypoints_pred[i]
            dir_name = os.path.join(output_path, "pred_videos")
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            interval_id = row["interval_id"]
            output_fn = os.path.join(dir_name, f"{interval_id}.mp4")
            save_video(dir_name, keypoints1, output_fn, delete_tmp=False)
        except Exception as e:
                logger.exception(e)

if __name__ == "__main__":
    args = create_args()
    G_model = Audio2PoseGANSTransformer(1, POSE_SAMPLE_SHAPE[-1]).to(device)
    G_model = torch.compile(G_model)
    G_model.load_state_dict(torch.load(args.param_path))
    G_model.eval()
    G_model.transformerencoder.train()
    data_csv: str = args.dataset
    df = pd.read_csv(data_csv)
    cfg: dict = {"processor": "audio_to_pose", "input_shape": [None, AUDIO_SHAPE]} #processorはaudio_to_pose_inferenceかもしれない
    keypoints1_list, keypoints2_list = predict_df(G_model, df, cfg, [0,0], [0,0], limit = 1)
    del G_model
    gc.collect()
    torch.cuda.empty_cache()
    keypoints1_list = translate_keypoints(keypoints1_list, [1500, 500])
    limit = len(keypoints1_list)
    save_prediction_video(df, keypoints1_list, args.output_path, limit = limit)