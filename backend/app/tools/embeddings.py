"""
Embedding Generators Tool
=========================
Generates multimodal embeddings for text, images, and audio.
- Text: BioBERT via sentence-transformers
- Image: CLIP via transformers (fallback from BiomedCLIP due to open_clip issues)
- Audio: CLAP via laion_clap
"""

import os
import warnings
from typing import Optional, List
import numpy as np
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger("embeddings")

# ============================================================================
# TEXT EMBEDDINGS (BioBERT)
# ============================================================================

class TextEmbedder:
    """Generate text embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"):
        self.model = None
        self.model_name = model_name
        self.dimension = 768
    
    def _load_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading text model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info("Text model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load text model: {e}")
                self.model = None
    
    def embed(self, text: Optional[str]) -> Optional[List[float]]:
        if not text or not text.strip():
            return None
        
        try:
            self._load_model()
            if self.model is None:
                return None
            
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return None

# ============================================================================
# IMAGE EMBEDDINGS (CLIP via transformers)
# ============================================================================

class ImageEmbedder:
    """Generate image embeddings using CLIP via transformers library"""
    
    # Using standard CLIP as fallback - BiomedCLIP requires open_clip which has import issues
    MODEL_NAME = "openai/clip-vit-base-patch32"
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.dimension = 512
    
    def _load_model(self):
        if self.model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch
                
                logger.info(f"Loading CLIP model: {self.MODEL_NAME}")
                self.model = CLIPModel.from_pretrained(self.MODEL_NAME)
                self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
                self.model.eval()
                
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(self.device)
                
                logger.info(f"CLIP loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load CLIP: {e}")
                self.model = None
    
    def embed(self, image_path: Optional[str]) -> Optional[List[float]]:
        if not image_path or not os.path.exists(image_path):
            return None
        
        try:
            self._load_model()
            if self.model is None or self.processor is None:
                return None
            
            import torch
            from PIL import Image
            
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.squeeze().cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Image embedding failed for {image_path}: {e}")
            return None
    
    def embed_batch(self, image_paths: List[str]) -> Optional[List[float]]:
        """Embed multiple images and return average embedding"""
        if not image_paths:
            return None
            
        embeddings = []
        for path in image_paths:
            emb = self.embed(path)
            if emb:
                embeddings.append(np.array(emb))
        
        if not embeddings:
            return None
            
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding.tolist()

# ============================================================================
# AUDIO EMBEDDINGS (CLAP)
# ============================================================================

class AudioEmbedder:
    """Generate audio embeddings using CLAP"""
    
    def __init__(self):
        self.model = None
        self.dimension = 512
    
    def _load_model(self):
        if self.model is None:
            try:
                import laion_clap
                logger.info("Loading CLAP model...")
                self.model = laion_clap.CLAP_Module(enable_fusion=False)
                self.model.load_ckpt()
                logger.info("CLAP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CLAP: {e}")
                self.model = None
    
    def embed(self, audio_path: Optional[str]) -> Optional[List[float]]:
        if not audio_path or not os.path.exists(audio_path):
            return None
        
        try:
            self._load_model()
            if self.model is None:
                return None
            
            audio_embed = self.model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=False)
            return audio_embed[0].tolist()
            
        except Exception as e:
            logger.error(f"Audio embedding failed for {audio_path}: {e}")
            return None

# ============================================================================
# UNIFIED EMBEDDER
# ============================================================================

class MultiModalEmbedder:
    """Unified interface for all embedding types"""
    
    def __init__(self):
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
        self.audio_embedder = AudioEmbedder()
    
    def embed_text(self, text: Optional[str]) -> Optional[List[float]]:
        return self.text_embedder.embed(text)
    
    def embed_image(self, image_path: Optional[str]) -> Optional[List[float]]:
        return self.image_embedder.embed(image_path)
    
    def embed_images(self, image_paths: List[str]) -> Optional[List[float]]:
        """Embed multiple images and return average"""
        return self.image_embedder.embed_batch(image_paths)
    
    def embed_audio(self, audio_path: Optional[str]) -> Optional[List[float]]:
        return self.audio_embedder.embed(audio_path)
