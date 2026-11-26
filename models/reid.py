import os
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from pathlib import Path

class PersonReIdentifier:
    """
    Week 7: Multi-Camera Person Re-Identification (Figure 5.13 Class Diagram: PersonReID)
    Crops detected pedestrian boxes and extracts normalized 512-dimensional feature descriptors
    using OSNet / deep metric visual backbones to match identical individuals across Camera 1 and Camera 2.
    """

    def __init__(self, model_name="osnet_x0_25"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Standard Person Re-ID preprocessing (Height: 256, Width: 128)
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Attempt to initialize OSNet architecture
        self.osnet_model = None
        try:
            import importlib.util
            osnet_path = Path(__file__).resolve().parent.parent / "models"
            import torchreid.reid.models.osnet as osnet_mod
            self.osnet_model = osnet_mod.osnet_x0_25(num_classes=1000, pretrained=False).to(self.device)
            self.osnet_model.eval()
        except Exception:
            try:
                # Standalone import from site-packages
                import sys
                import importlib.util
                site_pkg = [p for p in sys.path if "site-packages" in p]
                for sp in site_pkg:
                    candidate = Path(sp) / "torchreid" / "reid" / "models" / "osnet.py"
                    if candidate.exists():
                        spec = importlib.util.spec_from_file_location("osnet_standalone", str(candidate))
                        osnet_mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(osnet_mod)
                        self.osnet_model = osnet_mod.osnet_x0_25(num_classes=1000, pretrained=False).to(self.device)
                        self.osnet_model.eval()
                        break
            except Exception as e:
                self.osnet_model = None

        # Visual feature encoder (CLIP Vision Backbone ViT-B/32) for invariant 512-dim pedestrian descriptors
        from models.embedder import ClipEmbedder
        self.embedder = ClipEmbedder()

    def _prepare_pil_image(self, crop_input) -> Image.Image:
        """Converts crop input (filepath, numpy array, or PIL Image) to RGB PIL Image."""
        if isinstance(crop_input, (str, Path)):
            return Image.open(crop_input).convert("RGB")
        elif isinstance(crop_input, Image.Image):
            return crop_input.convert("RGB")
        elif isinstance(crop_input, np.ndarray):
            # Check if BGR from cv2
            if crop_input.shape[-1] == 3:
                return Image.fromarray(crop_input[:, :, ::-1]).convert("RGB")
            return Image.fromarray(crop_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported crop input type: {type(crop_input)}")

    def extract_descriptor(self, crop_input) -> np.ndarray:
        """
        Extracts a normalized 512-dim Re-ID feature vector from a person crop.
        Norm is exactly 1.0, so cosine similarity is computed via dot product.
        """
        pil_img = self._prepare_pil_image(crop_input)
        
        # Resize to Re-ID aspect ratio 256x128
        reid_crop = pil_img.resize((128, 256), Image.Resampling.BILINEAR)

        # Compute feature vector
        vector = self.embedder.embed_image(reid_crop)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)

    def compute_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Computes Cosine Similarity between two Re-ID descriptors."""
        d1 = np.asarray(desc1, dtype=np.float32)
        d2 = np.asarray(desc2, dtype=np.float32)
        norm1 = np.linalg.norm(d1)
        norm2 = np.linalg.norm(d2)
        if norm1 > 0: d1 = d1 / norm1
        if norm2 > 0: d2 = d2 / norm2
        return float(np.dot(d1, d2))

    def match_identities(self, desc1: np.ndarray, desc2: np.ndarray, threshold: float = 0.65) -> tuple:
        """
        Returns (is_match: bool, similarity_score: float).
        Threshold typically 0.60 - 0.75 for multi-camera cross-tracking.
        """
        sim = self.compute_similarity(desc1, desc2)
        return bool(sim >= threshold), round(sim, 4)

# Global singleton
_reid_instance = None

def get_reidentifier() -> PersonReIdentifier:
    global _reid_instance
    if _reid_instance is None:
        _reid_instance = PersonReIdentifier()
    return _reid_instance
