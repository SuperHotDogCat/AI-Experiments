from audio_to_multiple_pose_gan.config import get_config
from audio_to_multiple_pose_gan.dataset import load_train, generate_batch, get_processor
#get_modelの実装は任せる
from audio_to_multiple_pose_gan.static_model_factory import Audio2PoseGANS, D_patchgan
from audio_to_multiple_pose_gan.torch_layers import to_motion_delta, keypoints_to_train, keypoints_regloss
from common.audio_lib import save_audio_sample
from common.audio_repr import raw_repr
from common.consts import RIGHT_BODY_KEYPOINTS, LEFT_BODY_KEYPOINTS, LEFT_HAND_KEYPOINTS, \
    RIGHT_HAND_KEYPOINTS, POSE_SAMPLE_SHAPE, G_SCOPE, D_SCOPE, SR, FRAMES_PER_SAMPLE, AUDIO_SHAPE
from common.evaluation import compute_pck
from common.pose_logic_lib import translate_keypoints, get_sample_output_by_config
from common.pose_plot_lib import save_side_by_side_video, save_video_from_audio_video
from torch import optim
import torch
import pandas as pd
import argparse

def get_training_keypoints():
    training_keypoints = []
    training_keypoints.extend(RIGHT_BODY_KEYPOINTS)
    training_keypoints.extend(LEFT_BODY_KEYPOINTS)
    for i in range(5):
        training_keypoints.extend(RIGHT_HAND_KEYPOINTS(i))
        training_keypoints.extend(LEFT_HAND_KEYPOINTS(i))
    training_keypoints = sorted(list(set(training_keypoints)))
    return training_keypoints

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=100, help='EPOCHS')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='BATCH_SIZE')
    args = parser.parse_args()
    return args

"""
To do train_loop関数を作る
"""

if __name__ == "__main__":
    args = make_args()
    """
    model, optimizers定義
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G_model = Audio2PoseGANS(1, POSE_SAMPLE_SHAPE[-1]).to(device) #つまりPOSESAMPLESHAPE[-1] = 98
    D_model = D_patchgan(in_channels=FRAMES_PER_SAMPLE).to(device) #つまりFRAMESPERSAMPLE=64

    optimizer_g = optim.Adam(G_model.parameters(), lr = 1e-4) #初期値にしたがってlr = 1e-4としている。
    optimizer_d = optim.Adam(D_model.parameters(), lr = 1e-4)

    train_csv: str = "~/Desktop/AI-Experiments/speechandtext2gesture/Gestures/train.csv" #ウルトラハードコーディングだがしょうがない
    df = pd.read_csv(train_csv)

    cfg: dict = {"processor": "audio_to_pose", "input_shape": [None, AUDIO_SHAPE]}
    process_row, decode_pose = get_processor(cfg)
    #row = df.sample(n=1).iloc[0]
    #audio_X, pose_Y = generate_batch(df, process_row, batch_size)  でデータを取り出す。
    """
    model, optimizers定義終了
    """