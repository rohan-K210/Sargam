from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class LayerOutput:
    """Base output for all pipeline layers."""
    success: bool
    data: Any
    metadata: Dict
    errors: list


class PipelineLayer(ABC):
    """Abstract base class for pipeline layers."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def process(self, audio: np.ndarray, sr: int, context: Dict) -> LayerOutput:
        """Process audio and return layer output."""
        pass
    
    def validate_input(self, audio: np.ndarray, sr: int) -> bool:
        """Validate input audio."""
        return audio is not None and len(audio) > 0 and sr > 0
    
    def __call__(self, audio: np.ndarray, sr: int, context: Dict) -> LayerOutput:
        if not self.validate_input(audio, sr):
            return LayerOutput(
                success=False,
                data=None,
                metadata={},
                errors=[f"{self.name}: Invalid input audio"]
            )
        return self.process(audio, sr, context)