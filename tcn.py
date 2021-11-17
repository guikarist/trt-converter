import torch
from torch import nn
from torch.nn.utils import weight_norm

__all__ = ['TCN', 'TCNWithTwoInputs', 'TCNWithEnsembles']


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # conv1 gets (batch, num_features, seq_len)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        pred = self.linear(output[:, -1, :])
        return pred


class TCNWithTwoInputs(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, num_words=10000):
        super(TCNWithTwoInputs, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.embedding_layer = nn.Embedding(num_words, 1)

    def forward(self, x1, x2):
        embedded_x2 = self.embedding_layer(x2)
        embedded_x2 = torch.reshape(embedded_x2, embedded_x2.size()[:-1])
        embedded_x2 = embedded_x2.repeat(1, x1.size()[1], 1)
        x = torch.cat((x1, embedded_x2), 2)
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        pred = self.linear(output[:, -1, :])
        return pred


class TCNWithEnsembles(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, num_ensembles=10):
        super(TCNWithEnsembles, self).__init__()
        self.tcn_s = nn.ModuleList([TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout) for _ in
                                    range(num_ensembles)])
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        transposed_x = x.transpose(1, 2)
        output = torch.stack([tcn(transposed_x).transpose(1, 2) for tcn in self.tcn_s], dim=0).mean(dim=0)
        pred = self.linear(output[:, -1, :])
        return pred
