from audio_to_multiple_pose_gan.dataset import generate_batch, get_processor
#get_modelの実装は任せる
from audio_to_multiple_pose_gan.static_model_factory import Audio2PoseGANSTransformer, D_patchgan
from audio_to_multiple_pose_gan.torch_layers import to_motion_delta, keypoints_regloss
from common.consts import RIGHT_BODY_KEYPOINTS, LEFT_BODY_KEYPOINTS, LEFT_HAND_KEYPOINTS, \
    RIGHT_HAND_KEYPOINTS, POSE_SAMPLE_SHAPE, FRAMES_PER_SAMPLE, AUDIO_SHAPE
from torch import optim
import torch
import pandas as pd
import argparse
from torch import nn
import warnings
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    parser.add_argument('--dataset_path', '-d', type=str, default="~/Desktop/AI-Experiments/speechandtext2gesture/Gestures/train.csv", help="DATAPATH")
    parser.add_argument('--param_path', '-p', type = str, default = None)
    args = parser.parse_args()
    return args

"""
To do train_loop関数を作る
"""

def train(G_model, D_model, optimizer_g, optimizer_d, df, process_row,batch_size,device):
    """
    1 epoch計算する
    """
    optimizer_d.zero_grad()
    audio_X, pose_Y = generate_batch(df, process_row, batch_size)
    audio_X = torch.tensor(audio_X).float().to(device)
    pose_Y = torch.tensor(pose_Y).float().to(device)
    pose_Y /= 100.0 #訓練効率化のため
    fake_pose_Y = G_model(audio_X)
    #事情によりkeypoints to trainはなしで
    pose_Y_input = torch.concatenate([pose_Y,to_motion_delta(pose_Y)], dim =1)
    fake_pose_Y_input = torch.concatenate([fake_pose_Y,to_motion_delta(fake_pose_Y)], dim =1)
    D_loss = 0
    D_loss += torch.pow( torch.ones(size=(pose_Y_input.size(0), 1)).to(device) - D_model(pose_Y_input),2 ).mean()
    D_loss += 1e-4 * torch.pow(torch.zeros((fake_pose_Y_input.size(0), 1)).to(device) - D_model(fake_pose_Y_input), 2 ).mean()
    D_loss.backward()
    optimizer_d.step()

    optimizer_g.zero_grad()
    audio_X, pose_Y = generate_batch(df, process_row, batch_size)
    audio_X = torch.tensor(audio_X).float().to(device)
    pose_Y = torch.tensor(pose_Y).float().to(device)
    pose_Y /= 100.0 #訓練効率化のため
    pose_Y_motion = to_motion_delta(pose_Y)
    fake_pose_Y = G_model(audio_X)
    fake_pose_Y_motion = to_motion_delta(fake_pose_Y)
    fake_pose_Y_input = torch.concatenate([fake_pose_Y,to_motion_delta(fake_pose_Y)], dim =1)
    g_gan_loss = torch.pow(torch.ones(size=(fake_pose_Y_input.size(0),1)).to(device) - D_model(fake_pose_Y_input), 2 ).mean()
    G_loss = keypoints_regloss(pose_Y, fake_pose_Y, "l1") + keypoints_regloss(pose_Y_motion, fake_pose_Y_motion, "l1") + 1e-4 * g_gan_loss
    G_loss.backward()
    optimizer_g.step()

    return G_loss.cpu().detach().item(), D_loss.cpu().detach().item()

if __name__ == "__main__":
    args = make_args()
    """
    model, optimizers定義
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G_model = Audio2PoseGANSTransformer(1, POSE_SAMPLE_SHAPE[-1]+6).to(device) #つまりPOSESAMPLESHAPE[-1] = 98
    D_model = D_patchgan(in_channels=FRAMES_PER_SAMPLE+FRAMES_PER_SAMPLE-1, linear_size=26).to(device) #つまりFRAMESPERSAMPLE=64でto_motion_deltaを組み合わせた動きとする
    G_model = torch.compile(G_model)
    D_model = torch.compile(D_model)
    if args.param_path != None:
        G_model.load_state_dict(torch.load(args.param_path))

    optimizer_g = optim.Adam(G_model.parameters(), lr = 1e-4) #初期値にしたがってlr = 1e-4としている。
    optimizer_d = optim.Adam(D_model.parameters(), lr = 1e-4)
    print("EPOCHS: ", args.epochs)
    train_csv: str = args.dataset_path #ウルトラハードコーディングだがしょうがない
    df = pd.read_csv(train_csv)
    cfg: dict = {"processor": "audio_to_pose_new_keypoints", "input_shape": [None, AUDIO_SHAPE], "new_keypoints":True}
    process_row, decode_pose = get_processor(cfg)
    #row = df.sample(n=1).iloc[0]
    #audio_X, pose_Y = generate_batch(df, process_row, args.batch_size)  でデータを取り出す。
    """
    model, optimizers定義終了
    """
    G_losses = [0] * args.epochs
    D_losses = [0] * args.epochs
    epochs = [0] * args.epochs
    for epoch in tqdm(range(args.epochs)):
        G_loss, D_loss = train(G_model, D_model, optimizer_g, optimizer_d, df, process_row,args.batch_size,device)
        G_losses[epoch] = G_loss
        D_losses[epoch] = D_loss
        epochs[epoch] = epoch + 1
        if (epoch+1) % 100 == 0:
            print(f"epoch: {epoch + 1}, G_loss: {G_loss}, D_loss: {D_loss}")
        if (epoch+1) % 1000 == 0:
            torch.save(G_model.state_dict(), f"params/G_transformermodel_NK_{epoch+1}.pth")
    torch.save(G_model.state_dict(), "params/G_transformermodel_NK_last.pth")
    torch.save(D_model.state_dict(), "params/D_transformermodel_NK_last.pth")
    plt.plot(epochs, G_losses)
    plt.plot(epochs, D_losses)
    plt.legend(["Generator Loss", "Discriminator Loss"])
    plt.savefig("TransformerLossGraphNK.png")
    plt.show()
