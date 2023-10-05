import torch
import torch.nn as nn
from torch import cuda      

# GPU上で計算を行う場合は，この変数を非Noneの整数、又は"cuda"にする
gpu_id = "cuda" if torch.cuda.is_available() else "cpu"

# 言語モデル用ニューラルネットワークの定義
class LanguageModelLSTM(nn.Module):
    def __init__(self, source_vocabulary_size, embed_size=100):
        # パラメータを chainer.Chain に渡す
        super(LanguageModelLSTM, self).__init__()
        self.W_x_hi = nn.Embedding(source_vocabulary_size, embed_size)
        self.W_lstm=nn.LSTMCell(embed_size, embed_size)
        # ここのhrはタプル(hidden state, cell state)
        self.hr = None
        self.W_hr_y=nn.Linear(embed_size, source_vocabulary_size) #最後softmaxに通すための層
        if gpu_id is not None:
            self.device = torch.device(gpu_id)
            self.to(self.device)
        else:
            self.device = torch.device('cpu')
        self.embed_size = embed_size
        self.reset_state()
    def reset_state(self):
        # 隠れ層の状態をリセットする
        self.hr = torch.zeros(1, self.embed_size).to(self.device)
        self.cr = torch.zeros(1, self.embed_size).to(self.device)

    def forward(self, word):
        if gpu_id is not None:
            word = word.to(self.device)
        # ここを実装する
        # wordはcur_wordである。
        hi = self.W_x_hi(word)
        self.hr, self.cr = self.W_lstm(hi, (self.hr, self.cr))
        y = self.W_hr_y(self.cr)
        return y
