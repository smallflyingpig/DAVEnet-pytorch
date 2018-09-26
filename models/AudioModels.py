# Author: David Harwath
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        
class Davenet(nn.Module):
    def __init__(self, embedding_dim=1024, rnn_type="lstm"):
        super(Davenet, self).__init__()
        self.embedding_dim = embedding_dim
        
        self.rnn_type = rnn_type
        self.rnn_layers = 1
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        # (1024, seq_len)
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))
        if rnn_type=="rnn":
            self.RNN = nn.RNN(embedding_dim, embedding_dim, num_layers=self.rnn_layers)
        elif rnn_type=="lstm":
            self.RNN = nn.LSTM(embedding_dim, embedding_dim, num_layers=self.rnn_layers)
        else:
            raise NotImplementedError
        

    def init_hidden(self, x):
        batch_size = x.shape[0]
        rtn = (torch.zeros(self.rnn_layers, batch_size, self.embedding_dim, device=x.device),
                torch.zeros(self.rnn_layers, batch_size, self.embedding_dim, device=x.device))
        
        return rtn 
        
    def forward(self, x, nframes):
        # (batch, channel, 40, seq_len)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        # (batch, channel, sel_len)
        x = x.squeeze(2)

        self.h0 = self.init_hidden(x)
        x_pad = pack_padded_sequence(x, nframes, batch_first=True)
        output, hidden = self.RNN(x_pad, self.h0)
        output = pad_packed_sequence(output, batch_first=True)[0]
        words_emb = output.transpose(1, 2)
        sent_emb = hidden[0].transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.embedding_dim)  #(batch_size, embedding_dim)
        return x, sent_emb
        