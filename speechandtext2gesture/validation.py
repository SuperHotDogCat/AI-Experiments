import glob
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
from common.consts import POSE_SAMPLE_SHAPE, AUDIO_SHAPE, SR
import warnings
warnings.simplefilter('ignore')
import gc
from common.audio_repr import raw_repr
import numpy as np
from torch import nn
from common.pose_logic_lib import normalize_relative_keypoints, preprocess_to_relative, decode_pose_normalized_keypoints, get_pose, decode_pose_normalized_keypoints_no_scaling, decode_pose_normalized_keypoints_new_keypoints
from audio_to_multiple_pose_gan.torch_layers import to_motion_delta, keypoints_to_train, keypoints_regloss

NUM_DATA = 25 #事情により25個のデータでしか検証ができない
device = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speaker', '-s', type=str, default="shelly")
    parser.add_argument('--output_path', '-op', type=str, default="pred_videos")
    parser.add_argument('--model_type', '-mt', type=str, default="normal")
    parser.add_argument('--param_path', '-p', type = str, default = "params/G_model_last.pth")
    args = parser.parse_args()
    return args

from common.consts import POSE_SAMPLE_SHAPE, AUDIO_SHAPE, SR

if __name__ == "__main__":
    args = create_args()
    valid_X_path = "Gestures/test_256_all/shelly/audio/*"
    valid_Y_path = "Gestures/test_256_all/shelly/npz_pose/*"
    valid_X_path = glob.glob(valid_X_path)
    valid_Y_path = glob.glob(valid_Y_path)
    valid_X_path.sort()
    valid_Y_path.sort()
    valid_X_path = valid_X_path[:NUM_DATA]
    valid_Y_path = valid_Y_path[:NUM_DATA]
    if args.model_type == "transformer":
        G_model = Audio2PoseGANSTransformer(1, POSE_SAMPLE_SHAPE[-1]).to(device) #つまりPOSESAMPLESHAPE[-1] = 98
        G_model.eval()
        G_model.transformerencoder.train()
        G_model = torch.compile(G_model)
        G_model.load_state_dict(torch.load("params/G_transformermodel_last.pth"))
    elif args.model_type == "normal":
        G_model = Audio2PoseGANS(1, POSE_SAMPLE_SHAPE[-1]).to(device)
        G_model.eval()
        G_model = torch.compile(G_model)
        G_model.load_state_dict(torch.load("params/G_model_last.pth"))
    else:
        raise SyntaxError(f"{args.model_type}は実装されていません")
    val_loss = 0
    val_pose_loss = 0
    val_motion_loss = 0
    for n_data in tqdm(range(NUM_DATA)):
        audio_x_p = valid_X_path[n_data]
        pose_y_p = valid_Y_path[n_data]
        audio_x, sr = raw_repr(audio_x_p, sr = SR)
        audio_x = audio_x[np.newaxis,:] #(1, 67727)とかかな
        audio_x= torch.tensor(audio_x).float().to(device)
        if audio_x.size(1) != 67268:
            continue
        pose_y = np.load(valid_Y_path[n_data])
        pose_y = next(iter(pose_y.values()))
        pose_y = np.delete(pose_y, [7,8,9], axis = 2)
        pose_y = preprocess_to_relative(pose_y)
        pose_y = normalize_relative_keypoints(pose_y, speaker="shelly")
        pose_y = pose_y[np.newaxis,:,:]
        pose_y = torch.tensor(pose_y).float().to(device)

        with torch.no_grad():
            pred_pose_y = G_model(audio_x)
            pose_y_motion = to_motion_delta(pose_y)
            pred_pose_y_motion = to_motion_delta(pred_pose_y)
            val_motion_loss += keypoints_regloss(pose_y_motion, pred_pose_y_motion, "l1")
            val_pose_loss += keypoints_regloss(pose_y, pred_pose_y, "l1")
            val_loss += keypoints_regloss(pose_y, pred_pose_y, "l1") + keypoints_regloss(pose_y_motion, pred_pose_y_motion, "l1")
    print(f"{args.model_type} pose valid loss: {val_pose_loss.detach().cpu().item() / NUM_DATA}")
    print(f"{args.model_type} motion valid loss: {val_motion_loss.detach().cpu().item() / NUM_DATA}")
    print(f"{args.model_type} valid loss: {val_loss.detach().cpu().item() / NUM_DATA}")

# normal valid loss: 0.4792803192138672
"""
normal pose valid loss: 0.4294252777099609
normal motion valid loss: 0.04985506057739258
normal valid loss: 0.4792803192138672
"""

#transformerは結果がブレる
"""
transformer pose valid loss: 0.3952843475341797
transformer motion valid loss: 0.05003711700439453
transformer valid loss: 0.4453215026855469

transformer pose valid loss: 0.39730979919433596
transformer motion valid loss: 0.05027887344360352
transformer valid loss: 0.4475886535644531

transformer pose valid loss: 0.3961880874633789
transformer motion valid loss: 0.05015779495239258
transformer valid loss: 0.4463459396362305
"""