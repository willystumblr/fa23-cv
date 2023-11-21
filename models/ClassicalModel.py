import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np

class SobelEdgeDetector(nn.Module):
    def __init__(self):
        super(SobelEdgeDetector, self).__init__()

        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.weight_x = nn.Parameter(sobel_x, requires_grad=False)
        self.weight_y = nn.Parameter(sobel_y, requires_grad=False)

    def forward(self, x):
        x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x_gray = x_gray.unsqueeze(1)  # (batch_size, 1, height, width)

        x_sobel_x = F.conv2d(x_gray, self.weight_x, padding=1)
        x_sobel_y = F.conv2d(x_gray, self.weight_y, padding=1)

        edge_strength = torch.sqrt(x_sobel_x**2 + x_sobel_y**2)
        return edge_strength


class SIFTFeatureExtractor(nn.Module):
    def __init__(self, num_keypoints=133):
        super(SIFTFeatureExtractor, self).__init__()
        self.num_keypoints = num_keypoints
        self.sift = cv2.SIFT_create(self.num_keypoints)

    def forward(self, x):
        # Assuming x is a CPU tensor of shape (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape
        x_np = x.numpy().transpose(0, 2, 3, 1)  # Convert to numpy and shape (batch_size, height, width, channels)

        # Placeholder for SIFT features
        sift_features = []

        for i in range(batch_size):
            # Convert to grayscale
            gray_image = cv2.cvtColor(x_np[i], cv2.COLOR_BGR2GRAY)
            # Detect SIFT features and compute descriptors.
            keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)

            # Here you need to decide how to handle the keypoints and descriptors.
            # For instance, you can create a fixed-size feature map or a tensor based on the descriptors
            # You might need to normalize or otherwise process the descriptors before using them

            # Example: Create a fixed-size padded descriptor tensor
            if descriptors is not None:
                if descriptors.shape[0] < self.num_keypoints:
                    # If we have fewer than num_keypoints descriptors, pad with zeros
                    descriptors = np.vstack(
                        [descriptors, np.zeros((self.num_keypoints - descriptors.shape[0], descriptors.shape[1]))]
                    )
                elif descriptors.shape[0] > self.num_keypoints:
                    # If we have more than num_keypoints descriptors, truncate
                    descriptors = descriptors[:self.num_keypoints, :]
            else:
                # If no descriptors were found, create a zero tensor
                descriptors = np.zeros((self.num_keypoints, 128))  # 128 is the length of SIFT descriptors

            sift_features.append(descriptors)

        # Convert list of numpy arrays to a single tensor
        sift_tensor = torch.tensor(np.stack(sift_features), dtype=torch.float32)

        # Now you have a batch of SIFT descriptors as a tensor
        return sift_tensor



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = SobelEdgeDetector().to(device)
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
