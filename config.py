import torch
import spacy


class Config:
    def __init__(self):
        self.device = self._get_device()
        self._configure_spacy()

    def _get_device(self):
        """Determine the best available device (GPU or CPU)"""
        if torch.cuda.is_available():
            print("✅ Using GPU acceleration")
            return torch.device("cuda")
        else:
            print("⚠️ GPU not available, using CPU")
            return torch.device("cpu")

    def _configure_spacy(self):
        """Configure spaCy to use GPU if available"""
        if self.device.type == 'cuda':
            spacy.prefer_gpu()


# Global configuration instance
config = Config()