import pytest
import numpy as np
from src.pipeline.orchestrator import create_pipeline


def test_pipeline_creation():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.layers) == 6


def test_pitch_tonic_layer():
    from src.pipeline.layer1_pitch_tonic import PitchTonicLayer
    layer = PitchTonicLayer({})
    assert layer.name == "PitchTonicLayer"


def test_timing_layer():
    from src.pipeline.layer2_timing import TimingLayer
    layer = TimingLayer({})
    assert layer.name == "TimingLayer"


def test_gamaka_layer():
    from src.pipeline.layer3_gamaka import GamakaLayer
    layer = GamakaLayer({})
    assert layer.name == "GamakaLayer"


def test_swara_mapper():
    from src.mapping.swara_mapper import SwaraMapper
    mapper = SwaraMapper()
    
    # Test Hz to cents
    cents = mapper.hz_to_cents(440, 440)
    assert cents == 0.0
    
    cents = mapper.hz_to_cents(880, 440)
    assert abs(cents - 1200) < 1
    
    # Test svara mapping
    svara, variant, target = mapper.cents_to_svara(200)
    assert svara == 'Re'
    assert variant == 'komal'


def test_tonic_detector():
    from src.models.tonic_detector import detect_tonic_from_pitch, hz_to_note_name
    
    freqs = np.array([440, 440, 466, 440, 440])  # A4
    conf = np.array([0.9, 0.9, 0.8, 0.9, 0.9])
    voiced = np.array([True]*5)
    
    tonic_hz, tonic_conf, tonic_note = detect_tonic_from_pitch(freqs, conf, voiced)
    assert abs(tonic_hz - 440) < 5
    assert tonic_note == 'A4'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])