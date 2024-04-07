import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Prep Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Layer 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride = 1, padding = 1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Layer 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride = 1, padding = 1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Layer 3
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride = 1, padding = 1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv_fc = nn.Sequential(
            nn.MaxPool2d(4, 4)
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # Prep Layer
        x = self.conv1(x)

        # Layer 1
        x = self.conv2(x)
        r1 = self.res1(x)
        r1 = self.res1(r1)
        x1 = x + r1

        #Layer 2
        x1 = self.conv3(x1)

        # Layer 3
        x1 = self.conv4(x1)
        r2 = self.res2(x1)
        r2 = self.res2(r2)
        x2 = x1 + r2

        x2 = self.conv_fc(x2)

        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc(x2)
        x2 = F.log_softmax(x2, dim=1)

        return x2