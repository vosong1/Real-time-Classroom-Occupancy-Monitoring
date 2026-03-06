import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===== Mini CNN (hard-code) =====
class MiniBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)  # /2

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)  # /2

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)  # /2

    def forward(self, x):
        f1 = self.conv1(x)      # (1,16,H,W)
        x = self.pool1(f1)      # (1,16,H/2,W/2)

        f2 = self.conv2(x)      # (1,32,H/2,W/2)
        x = self.pool2(f2)      # (1,32,H/4,W/4)

        f3 = self.conv3(x)      # (1,64,H/4,W/4)
        x = self.pool3(f3)      # (1,64,H/8,W/8)
        return f1, f2, f3


def preprocess(img_path, size=640):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img_rgb, x


def show_feature_grid(feat, title, max_ch=16):
    # feat: (1,C,H,W)
    feat = feat[0].detach().cpu().numpy()  # C,H,W
    C = feat.shape[0]
    n = min(max_ch, C)
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(10, 8))
    for i in range(n):
        fm = feat[i]
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(fm, cmap="viridis")
        ax.set_title(f"ch {i}")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    IMG_PATH = "test.jpg"  # đổi tên ảnh của bạn ở đây
    img_rgb, x = preprocess(IMG_PATH, size=640)

    model = MiniBackbone()
    model.eval()

    with torch.no_grad():
        f1, f2, f3 = model(x)

    print("f1:", tuple(f1.shape), "| f2:", tuple(f2.shape), "| f3:", tuple(f3.shape))

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title("Input Image")
    plt.axis("off")
    plt.show()

    show_feature_grid(f1, "conv1 feature maps (low-level edges)")
    show_feature_grid(f2, "conv2 feature maps (mid-level textures)")
    show_feature_grid(f3, "conv3 feature maps (higher-level patterns)")