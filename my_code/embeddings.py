"""
Embedding Generators for Patient Memory Agent
==============================================
Generates multimodal embeddings for text, images, and audio.

Models:
- Text: sentence-transformers (medical model)
- Image: CLIP (OpenAI)
- Audio: CLAP (Contrastive Language-Audio Pretraining)

Each generator handles missing data gracefully.
"""

import os
import warnings
from typing import Optional, List
import numpy as np
from PIL import Image

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TEXT EMBEDDINGS
# ============================================================================

class TextEmbedder:
    """Generate text embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"):
        """
        Initialize text embedder.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model = None
        self.model_name = model_name
        self.dimension = 768  # BioBERT actual dimension
    
    def _load_model(self):
        """Lazy load the model"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[Text Embedder] Loading model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                print(f"[Text Embedder] ✓ Model loaded, dimension: {self.dimension}")
            except Exception as e:
                print(f"[Text Embedder] ✗ Failed to load model: {e}")
                self.model = None
    
    def embed(self, text: Optional[str]) -> Optional[List[float]]:
        """
        Generate text embedding.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not text or text.strip() == "":
            print("[Text Embedder] ⚠ No text provided")
            return None
        
        try:
            self._load_model()
            
            if self.model is None:
                return None
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Convert to list
            embedding_list = embedding.tolist()
            
            print(f"[Text Embedder] ✓ Generated embedding, dim: {len(embedding_list)}")
            return embedding_list
            
        except Exception as e:
            print(f"[Text Embedder] ✗ Embedding failed: {e}")
            return None


# ============================================================================
# IMAGE EMBEDDINGS
# ============================================================================

class ImageEmbedder:
    """Generate image embeddings using CLIP"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize image embedder.
        
        Args:
            model_name: CLIP model name
        """
        self.model = None
        self.processor = None
        self.model_name = model_name
        self.dimension = 512  # CLIP ViT-B/32 dimension
    
    def _load_model(self):
        """Lazy load the model"""
        if self.model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                print(f"[Image Embedder] Loading model: {self.model_name}")
                self.model = CLIPModel.from_pretrained(self.model_name)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
                print(f"[Image Embedder] ✓ Model loaded, dimension: {self.dimension}")
            except Exception as e:
                print(f"[Image Embedder] ✗ Failed to load model: {e}")
                self.model = None
    
    def embed(self, image_path: Optional[str]) -> Optional[List[float]]:
        """
        Generate image embedding.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not image_path:
            print("[Image Embedder] ⚠ No image path provided")
            return None
        
        if not os.path.exists(image_path):
            print(f"[Image Embedder] ✗ Image not found: {image_path}")
            return None
        
        try:
            self._load_model()
            
            if self.model is None or self.processor is None:
                return None
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Generate embedding
            outputs = self.model.get_image_features(**inputs)
            embedding = outputs.detach().numpy()[0]
            
            # Convert to list
            embedding_list = embedding.tolist()
            
            print(f"[Image Embedder] ✓ Generated embedding for {os.path.basename(image_path)}, dim: {len(embedding_list)}")
            return embedding_list
            
        except Exception as e:
            print(f"[Image Embedder] ✗ Embedding failed for {image_path}: {e}")
            return None
    
    def embed_multiple(self, image_paths: List[str]) -> Optional[List[float]]:
        """
        Generate averaged embedding from multiple images.
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            Averaged embedding vector, or None if all failed
        """
        if not image_paths:
            return None
        
        embeddings = []
        for path in image_paths:
            emb = self.embed(path)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            print("[Image Embedder] ⚠ All image embeddings failed")
            return None
        
        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        print(f"[Image Embedder] ✓ Averaged {len(embeddings)} image embeddings")
        return avg_embedding


# ============================================================================
# AUDIO EMBEDDINGS
# ============================================================================

class AudioEmbedder:
    """Generate audio embeddings using CLAP"""
    
    def __init__(self):
        """Initialize audio embedder"""
        self.model = None
        self.dimension = 512  # CLAP dimension
    
    def _load_model(self):
        """Lazy load the model"""
        if self.model is None:
            try:
                import laion_clap
                print(f"[Audio Embedder] Loading CLAP model")
                self.model = laion_clap.CLAP_Module(enable_fusion=False)
                self.model.load_ckpt()  # Load default checkpoint
                print(f"[Audio Embedder] ✓ Model loaded, dimension: {self.dimension}")
            except Exception as e:
                print(f"[Audio Embedder] ✗ Failed to load model: {e}")
                print(f"[Audio Embedder] ℹ CLAP may not be installed or checkpoint missing")
                self.model = None
    
    def embed(self, audio_path: Optional[str]) -> Optional[List[float]]:
        """
        Generate audio embedding.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not audio_path:
            print("[Audio Embedder] ⚠ No audio path provided")
            return None
        
        if not os.path.exists(audio_path):
            print(f"[Audio Embedder] ✗ Audio file not found: {audio_path}")
            return None
        
        try:
            self._load_model()
            
            if self.model is None:
                print("[Audio Embedder] ⚠ Model not available, skipping audio embedding")
                return None
            
            # Generate embedding
            audio_embed = self.model.get_audio_embedding_from_filelist(
                x=[audio_path],
                use_tensor=False
            )
            
            # Convert to list
            embedding_list = audio_embed[0].tolist()
            
            print(f"[Audio Embedder] ✓ Generated embedding for {os.path.basename(audio_path)}, dim: {len(embedding_list)}")
            return embedding_list
            
        except Exception as e:
            print(f"[Audio Embedder] ✗ Embedding failed for {audio_path}: {e}")
            return None


# ============================================================================
# UNIFIED EMBEDDER
# ============================================================================

class MultiModalEmbedder:
    """
    Unified interface for generating all modality embeddings.
    Lazy loads models as needed.
    """
    
    def __init__(self):
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
        self.audio_embedder = AudioEmbedder()
    
    def embed_text(self, text: Optional[str]) -> Optional[List[float]]:
        """Generate text embedding"""
        return self.text_embedder.embed(text)
    
    def embed_image(self, image_path: Optional[str]) -> Optional[List[float]]:
        """Generate image embedding"""
        return self.image_embedder.embed(image_path)
    
    def embed_images(self, image_paths: List[str]) -> Optional[List[float]]:
        """Generate averaged embedding from multiple images"""
        return self.image_embedder.embed_multiple(image_paths)
    
    def embed_audio(self, audio_path: Optional[str]) -> Optional[List[float]]:
        """Generate audio embedding"""
        return self.audio_embedder.embed(audio_path)


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EMBEDDING GENERATORS - DEMO")
    print("=" * 70 + "\n")
    
    embedder = MultiModalEmbedder()
    
    # Test text embedding
    print("Testing text embedding...")
    text_emb = embedder.embed_text("Patient presents with skin rash on forearm")
    if text_emb:
        print(f"✓ Text embedding generated, dimension: {len(text_emb)}")
        print(f"  First 5 values: {text_emb[:5]}")
    
    print("\n" + "=" * 70)
    print("✅ Embedding generators ready!")
    print("Note: Image and audio tests require actual files")
    print("=" * 70)
