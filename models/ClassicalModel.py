import torch
import torch.nn.functional as F
import torch.nn as nn


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
