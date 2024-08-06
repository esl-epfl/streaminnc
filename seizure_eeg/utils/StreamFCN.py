import torch.nn as nn
import torch

# import numpy as np

class StreamNet(nn.Module):
    def __init__(self, in_channels=22, overlap = 256):
        super(StreamNet, self).__init__()
        # first convolutional block
        n_filters = 128
        self.conv1 = nn.Conv1d(
            in_channels, n_filters, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu1 = nn.ReLU()
        # first pooling
        self.pool1 = nn.MaxPool1d(kernel_size=4, padding=0)

        # second convolutional block
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.relu2 = nn.ReLU()
        # second pooling
        self.pool2 = nn.MaxPool1d(kernel_size=4, padding=0)

        # third convolutional block
        self.conv3 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_filters)
        self.relu3 = nn.ReLU()
        # second pooling
        self.pool3 = nn.MaxPool1d(kernel_size=4, padding=0)

        self.fc1 = nn.Conv1d(n_filters, 100, kernel_size=16, padding=0)
        self.fc2 = nn.Conv1d(100, 2, kernel_size=1, padding=0)
        
        self.overlap = overlap
        
    def init_buffer(self, x):
        self.conv1_out = self.relu1(self.bn1(self.conv1(x)))
        self.pool1_out = self.pool1(self.conv1_out)
        
        self.conv2_out = self.relu2(self.bn2(self.conv2(self.pool1_out)))
        self.pool2_out = self.pool2(self.conv2_out)
        
        self.conv3_out = self.relu3(self.bn3(self.conv3(self.pool2_out)))
        self.pool3_out = self.pool3(self.conv3_out)
        
        out = self.fc1(self.pool3_out)
        out = self.fc2(out)
        out = out.transpose(0, 1)  # nxbxt
        n, b, t = out.size()
        out = out.contiguous()
        out = out.view(-1, t * b)  # nx(t*b)
        out = out.t()  # (t*b)xn
        
        return out
        

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        
        out = torch.cat((self.conv1_out[..., self.overlap:], 
                         out[..., -self.overlap : ]), dim = -1)
        self.conv1_out = out
        
        out = self.pool1(out)
        
        
        out = self.relu2(self.bn2(self.conv2(out)))
        out = torch.cat((self.conv2_out[..., self.overlap // 4:], 
                         out[..., -self.overlap // 4 : ]), dim = -1)
        
        self.conv2_out = out
        
        out = self.pool2(out)
        
        out = self.relu3(self.bn3(self.conv3(out)))
        out = torch.cat((self.conv3_out[..., (self.overlap // 4) // 4:], 
                         out[..., -(self.overlap // 4) // 4: ]), dim = -1)
        self.conv3_out = out
        
        out = self.pool3(out)

        out = self.fc1(out)
        out = self.fc2(out)
        out = out.transpose(0, 1)  # nxbxt
        n, b, t = out.size()
        out = out.contiguous()
        out = out.view(-1, t * b)  # nx(t*b)
        out = out.t()  # (t*b)xn
        

        return out


class ApproximateStreamNet(nn.Module):
    def __init__(self, in_channels=22, overlap = 256):
        super(ApproximateStreamNet, self).__init__()
        # first convolutional block
        n_filters = 128
        self.conv1 = nn.Conv1d(
            in_channels, n_filters, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu1 = nn.ReLU()
        # first pooling
        self.pool1 = nn.MaxPool1d(kernel_size=4, padding=0)

        # second convolutional block
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.relu2 = nn.ReLU()
        # second pooling
        self.pool2 = nn.MaxPool1d(kernel_size=4, padding=0)

        # third convolutional block
        self.conv3 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_filters)
        self.relu3 = nn.ReLU()
        # second pooling
        self.pool3 = nn.MaxPool1d(kernel_size=4, padding=0)

        self.fc1 = nn.Conv1d(n_filters, 100, kernel_size=16, padding=0)
        self.fc2 = nn.Conv1d(100, 2, kernel_size=1, padding=0)
        
        self.overlap = overlap
        
    def init_buffer(self, x):
        self.conv1_out = self.relu1(self.bn1(self.conv1(x)))
        self.pool1_out = self.pool1(self.conv1_out)
        
        self.conv2_out = self.relu2(self.bn2(self.conv2(self.pool1_out)))
        self.pool2_out = self.pool2(self.conv2_out)
        
        self.conv3_out = self.relu3(self.bn3(self.conv3(self.pool2_out)))
        self.pool3_out = self.pool3(self.conv3_out)
        
        out = self.fc1(self.pool3_out)
        out = self.fc2(out)
        out = out.transpose(0, 1)  # nxbxt
        n, b, t = out.size()
        out = out.contiguous()
        out = out.view(-1, t * b)  # nx(t*b)
        out = out.t()  # (t*b)xn
        
        return out
        

    def forward(self, x):
        
        
        mask = torch.zeros_like(x)
        mask[..., -self.overlap:] = 1
        
        out = self.relu1(self.bn1(self.conv1(x * mask)))
        
        out = torch.cat((self.conv1_out[..., self.overlap:], 
                         out[..., -self.overlap : ]), dim = -1)
        self.conv1_out = out
        
        out = self.pool1(out)
        
        mask = torch.zeros_like(out)
        mask[..., -self.overlap // 4:] = 1
        
        out = self.relu2(self.bn2(self.conv2(out * mask)))
        out = torch.cat((self.conv2_out[..., self.overlap // 4:], 
                         out[..., -self.overlap // 4 : ]), dim = -1)
        
        self.conv2_out = out
        
        out = self.pool2(out)
        
        mask = torch.zeros_like(out)
        mask[..., -(self.overlap // 4) // 4:] = 1
        
        out = self.relu3(self.bn3(self.conv3(out * mask)))
        out = torch.cat((self.conv3_out[..., (self.overlap // 4) // 4:], 
                         out[..., -(self.overlap // 4) // 4: ]), dim = -1)
        self.conv3_out = out
        
        out = self.pool3(out)

        out = self.fc1(out)
        out = self.fc2(out)
        out = out.transpose(0, 1)  # nxbxt
        n, b, t = out.size()
        out = out.contiguous()
        out = out.view(-1, t * b)  # nx(t*b)
        out = out.t()  # (t*b)xn        

        return out



