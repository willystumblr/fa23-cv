import torch
import torch.nn as nn
import torchvision.models as models
from .Resnet50 import model_resnet50
from .ClassicalModel import SobelEdgeDetector


class model_resnet50_with_sobel(nn.Module):
    def __init__(self, weights=None):
        super(model_resnet50_with_sobel, self).__init__()
        self.sobel = SobelEdgeDetector()  # (N, 1, 224, 224)
        self.resnet = model_resnet50(weights=weights)  # (N, 133*3)
        self.fc_sobel = nn.Linear(224 * 224, 133 * 3)
        self.fc_resnet = nn.Linear(133 * 3, 133 * 3)
        self.fc_combine = nn.Linear(133 * 3 * 2, 133 * 3)

    def forward(self, x):
        x1 = self.resnet(x)
        x1 = self.fc_resnet(x1)

        x2 = self.sobel(x)
        x2 = x2.reshape(x2.shape[0], -1)
        x2 = self.fc_sobel(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc_combine(x)

        return x


class model_resnet50_4(nn.Module):
    def __init__(self, num_keypoint=133, weights=None):
        super(model_resnet50_4, self).__init__()
        self.encoder = models.resnet50(weights=weights)

        pretrained_conv1_weights = self.encoder.conv1.weight.clone()
        self.encoder.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.encoder.conv1.weight[:, :3, :, :] = pretrained_conv1_weights

        self.encoder.fc = nn.Linear(2048, 133 * 3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(7 * 7 * 512 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.outlayer1 = nn.Linear(1024, num_keypoint * 3)

    def forward(self, x):
        # x = (x - 0.45) / 0.225
        x[:, :3, :, :] = (x[:, :3, :, :] - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = x.reshape(x.shape[0], -1)
        x1 = self.relu(self.fc1(x))
        x1 = self.relu(self.fc2(x1))
        x1 = self.outlayer1(x1)
        return x1

class model_resnet50_with_sift(nn.Module):
    def __init__(self, weights=None, num_keypoints=133, num_sift_features=128):
        super(model_resnet50_with_sift, self).__init__()
        
        self.resnet = model_resnet50(weights=weights)
        # Define a processing layer for SIFT features
        self.sift_processor = nn.Sequential(
            nn.Linear(num_keypoints * num_sift_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        # Final fully connected layer that combines both streams
        self.fc = nn.Linear(512 + (num_keypoints * 3), num_keypoints * 3)
        
    def forward(self, x, sift_features):
        # Process the image with the ResNet stream
        image_features = self.resnet(x)

        # Flatten and process the SIFT features
        # sift_features = sift_features.view(sift_features.size(0), -1)
        sift_features = self.sift_processor(sift_features)

        # Concatenate the image and SIFT features
        combined_features = torch.cat((image_features, sift_features), dim=1)

        # Final prediction
        output = self.fc(combined_features)
        return output

class model_resnet50_4_with_sobel(nn.Module):
    def __init__(self, weights=None):
        super(model_resnet50_4_with_sobel, self).__init__()
        self.sobel = SobelEdgeDetector()  # (N, 1, 224, 224)
        self.resnet = model_resnet50_4(weights=weights)  # (N, 133*3)

    def forward(self, x):
        edges = self.sobel(x)
        x = torch.cat((x, edges), dim=1)
        x = self.resnet(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = model_resnet50_4_with_sobel().to(device)
    batch_size = 1
    a = torch.rand(batch_size, 3, 224, 224).to(device)
    b = net(a)
    print(b.shape)

    from thop import profile

    flops, params = profile(net, inputs=(a,), verbose=False)
    print(f"FLOPs: {flops / 1e9} billion")
    print(f"Parameters: {params / 1e6} million")
    print(
        f"Trainable Parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6} million"
    )
