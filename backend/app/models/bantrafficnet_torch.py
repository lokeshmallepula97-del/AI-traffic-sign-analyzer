"""
BanTrafficNet PyTorch Implementation
Matches the exact layer parameters and layer dimensions of BanTrafficNet from the reference paper:
- Scientific Reports (2026) 16:31000
- 3 Convolutional Blocks with valid padding (kernel_size=3, padding=0)
- Input: 96x96x3 RGB, normalized [0, 1]
- Dropout: 0.25 on blocks 2 & 3, 0.50 on FC layer
- Output: num_classes Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BanTrafficNetTorch(nn.Module):
    def __init__(self, num_classes: int = 56):
        super(BanTrafficNetTorch, self).__init__()
        self.num_classes = num_classes
        
        # Conv Block 1: 96x96 -> 94x94 -> 92x92 -> 46x46
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0)   # 94x94
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=0)  # 92x92
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)        # 46x46
        
        # Conv Block 2: 46x46 -> 44x44 -> 42x42 -> 21x21
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=0)  # 44x44
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=0)  # 42x42
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)        # 21x21
        self.drop1 = nn.Dropout2d(0.25)
        
        # Conv Block 3: 21x21 -> 19x19 -> 17x17 -> 8x8
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=0) # 19x19
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=0)# 17x17
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)        # 8x8
        self.drop2 = nn.Dropout2d(0.25)
        
        # Flatten: 8 * 8 * 256 = 16,384
        self.flatten = nn.Flatten()
        
        # Fully Connected Block
        self.fc1 = nn.Linear(16384, 256)
        self.drop3 = nn.Dropout(0.50)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x expected shape: (B, 3, 96, 96), normalized [0, 1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.drop1(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.drop2(x)
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    net = BanTrafficNetTorch(num_classes=56)
    dummy = torch.randn(1, 3, 96, 96)
    out = net(dummy)
    params = count_parameters(net)
    print("PyTorch BanTrafficNet (56 classes):")
    print(f"Output shape: {out.shape}")
    print(f"Total Parameters: {params:,}")
    expected = 4629152 + (257 * 56)
    print(f"Expected theoretical: {expected:,}")
    assert params == expected, f"Param mismatch: {params} vs {expected}"
    print("Exact parameter count: VERIFIED!")
