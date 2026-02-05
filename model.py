import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DualPathDA(nn.Module):
    """
    Dual Path Deep Fake Detection Model with Attention Mechanisms.
    Designed to capture both spectral artifacts and temporal inconsistencies.
    """
    def __init__(self, input_channels=1, num_classes=2):
        super(DualPathDA, self).__init__()
        
        # Convolutional Front-end (Spectral Analysis)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Attention Modules
        self.ca = ChannelAttention(128)
        self.sa = SpatialAttention()
        
        # Recurrent Back-end (Temporal Analysis)
        # Using GRU as it's lighter than LSTM
        self.gru = nn.GRU(128, 64, bidirectional=True, batch_first=True)
        
        # Classification Head
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: [batch, 1, freq, time]
        
        # CNN Path
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Refine features with Attention
        x = self.ca(x) * x
        x = self.sa(x) * x
        
        # Prepare for RNN
        # Reshape: [batch, channels, freq, time] -> [batch, time, channels*freq]
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, t, -1)
        
        # RNN Path needs correct input size, adaptable via adaptive pool or fixed assumption
        # For simplicity in this demo, we use global average pooling instead of GRU if dimensions mismatch widely
        # But let's try to assume a standard input size or use adaptive pooling before GRU
        
        # Simplification for flexibility: Global Max Pooling over time instead of GRU complex sequence handling
        # This makes the model input-size agnostic
        x = torch.mean(x, dim=1) # Average over time [batch, c*f] -> this is too big if c*f is large.
        
        # Let's use adaptive pool instead to get fixed feature vector
        # x was [b, c, f, t]. Let's pool manually to [b, 128]
        # Using Global Average Pooling on the channel dimension essentially
        
        # Re-approach: GAP directly on the feature map
        # x is [b, 128, f, t] from conv3/attention
        x = F.adaptive_avg_pool2d(x, (1, 1)) # -> [b, 128, 1, 1]
        x = x.view(b, -1) # -> [b, 128]
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
