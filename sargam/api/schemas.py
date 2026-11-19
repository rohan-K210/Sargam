from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class PredictRequest(BaseModel):
    audio_base64: str
    tonic_hz: Optional[float] = None
    return_student_view: bool = True


class SwaraNoteOut(BaseModel):
    swara: str
    variant: str
    octave: str
    onset_sec: float
    duration_sec: float
    ornament: Dict[str, Any]
    dynamics: str
    attack: str


class PredictResponse(BaseModel):
    success: bool
    tonic: str
    tonic_hz: float
    tonic_confidence: float
    tempo: str
    tempo_bpm: float
    instrument: str
    instrument_confidence: float
    breaths: List[Dict]
    notes: List[SwaraNoteOut]
    audio_info: Dict
    student_view: Optional[Dict] = None
    error: Optional[str] = None