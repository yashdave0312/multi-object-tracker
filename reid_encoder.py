import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small
import numpy as np

class ReIDEncoder:
    def __init__(self, device="cpu"):
        self.device = device
        model = mobilenet_v3_small(weights="DEFAULT")
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.eval().to(device)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def encode(self, frame, bboxes):
        embeddings = []
        for (x1, y1, x2, y2) in bboxes:
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0:
                embeddings.append(None)
                continue
            t = self.transform(crop).unsqueeze(0).to(self.device)
            feat = self.model(t).squeeze().cpu().numpy()
            embeddings.append(feat / (np.linalg.norm(feat) + 1e-6))
        return embeddings