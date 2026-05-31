import torch
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
import torch.nn.functional as F

def init_sub_block(
    input_dim,
    model_dim,
    kernel_size=3,
    stride=2,
):
    return nn.Sequential(
        nn.Conv1d(input_dim, model_dim, kernel_size, stride, padding=1),
        nn.ReLU(),
    )


def calc_length(
    lengths, padding=1, kernel_size=3, stride=2, ceil_mode=False, repeat_num=1
):
    add_pad: float = (padding * 2) - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)

class PhoneCNNStack(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.Conv2d = nn.Conv2d(1, 1, 3, 1, 1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.Conv2d(x)
        x = x.squeeze(1)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.reLU(x)
        x = self.drop_out(x)
        return x


class PhoneRNNStack(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.bilstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim//2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x, _ = self.bilstm(x)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop_out(x)

        return x


class Phonetic_encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.CNN = PhoneCNNStack(hidden_dim=hidden_dim)
        self.RNN = PhoneRNNStack(hidden_dim=hidden_dim)

    def forward(self, x):
        x = self.CNN(x)
        x = self.RNN(x)
        return x



    
class Linguistic_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(256, 64)
        self.fc1 = nn.Linear(128, 2304)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=64, bidirectional=True, batch_first=True)
        self.fc3 = nn.Linear(128, 2304)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.long()
        x = self.embedding(x)
        o, _ = self.bilstm(x)
        y = self.fc1(o)
        x = self.fc3(o)
        return x, y
    
class PL(Wav2Vec2PreTrainedModel):
    def __init__(self, config, hidden_dim=768, vocab_size: int = 123):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.Phonetic_encoder = Phonetic_encoder(hidden_dim=hidden_dim)
        self.Linguistic_encoder = Linguistic_encoder()

        self.fc1 = nn.Linear(hidden_dim*2, vocab_size, bias=True)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, 16, batch_first=True, kdim=2304, vdim=2304)

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, input_values, linguistic):
        phonetic = self.wav2vec2(input_values, attention_mask=None)[0]
        phonetic = self.Phonetic_encoder(phonetic)  
        h_k, h_v = self.Linguistic_encoder(linguistic)

        # KHÔNG CẮT GỌT phonetic theo linguistic nữa!
        # Để MultiheadAttention tự làm việc: Query=phonetic (T frames), Key=h_k (N frames)
        attn_output, _ = self.multihead_attn(phonetic, h_k, h_v)
        
        before_Linear = torch.cat((attn_output, phonetic), dim=2)
        output = self.fc1(before_Linear)
        return output