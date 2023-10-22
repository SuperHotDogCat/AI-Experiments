from audio_to_multiple_pose_gan.config import get_config
from audio_to_multiple_pose_gan.dataset import load_train, generate_batch, get_processor
#get_modelの実装は任せる
#from audio_to_multiple_pose_gan.static_model_factory import get_model
from audio_to_multiple_pose_gan.torch_layers import to_motion_delta, keypoints_to_train, keypoints_regloss
from common.audio_lib import save_audio_sample
from common.audio_repr import raw_repr
from common.consts import RIGHT_BODY_KEYPOINTS, LEFT_BODY_KEYPOINTS, LEFT_HAND_KEYPOINTS, \
    RIGHT_HAND_KEYPOINTS, POSE_SAMPLE_SHAPE, G_SCOPE, D_SCOPE, SR
from common.evaluation import compute_pck
from common.pose_logic_lib import translate_keypoints, get_sample_output_by_config
from common.pose_plot_lib import save_side_by_side_video, save_video_from_audio_video

class PoseGAN(nn.Module):
    def __init__(self, args, seq_len = 64):
        super().__init__()
        """
        pose shape: (batch_size, seq_len, 98)

        """
    def forward(self, audio, real_pose):
        """
        audio: shape = (batch, 67267) 予定
        real_pose: shape = (batch, 64, 98) 予定

        real_poseはkey_points_to_train関数で処理されて
        training_real_poseとtraining_real_pose_motionという正解ラベルになる。

        audioはaudio2poseでmel_spectogram関数に通してから処理しているのでこのままでもよい
        """
    def train(self, ):

    def get_training_keypoints(self):
        training_keypoints = []
        training_keypoints.extend(RIGHT_BODY_KEYPOINTS)
        training_keypoints.extend(LEFT_BODY_KEYPOINTS)
        for i in range(5):
            training_keypoints.extend(RIGHT_HAND_KEYPOINTS(i))
            training_keypoints.extend(LEFT_HAND_KEYPOINTS(i))
        training_keypoints = sorted(list(set(training_keypoints)))
        return training_keypoints