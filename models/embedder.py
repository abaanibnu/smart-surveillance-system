import os
import torch
import numpy as np
from PIL import Image

# Monkey-patch torch.load for PyTorch 2.6+ compatibility with TorchScript weights
_orig_torch_load = torch.load
def _custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _custom_torch_load

import open_clip

class ClipEmbedder:
    """
    Implements DFD Level 1.3:
    - P1.8: CLIP Image Encoder (generates normalized 512-dimensional visual vector)
    - P2.3: CLIP Text Encoder (converts query text into 512-dim embedding)
    """

    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def embed_image(self, image_input) -> np.ndarray:
        """
        Encodes a single image file or PIL Image into a normalized 512-dim visual vector.
        """
        if isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = Image.open(str(image_input)).convert("RGB")

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor)
            # L2 normalize so dot product equals cosine similarity
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().astype(np.float32)

    def embed_image_batch(self, image_paths: list) -> np.ndarray:
        """
        Encodes a batch of images into (N, 512) normalized vectors.
        """
        tensors = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(self.preprocess(img))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Encodes query text into a normalized 512-dim text embedding vector.
        """
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().astype(np.float32)
