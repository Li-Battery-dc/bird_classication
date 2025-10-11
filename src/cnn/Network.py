import torch
import torch.nn as nn

class CNNNetwork(nn.Module):
    def __init__(self, num_classes=200, image_size=224):
        super(CNNNetwork, self).__init__()
        self.linear_input_size = image_size // 8  # after two 2x2 pooling layers
        self.cov = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112 * 112

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56 * 56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 * 28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * self.linear_input_size * self.linear_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.cov(x)
        x = self.classifier(x)
        return x