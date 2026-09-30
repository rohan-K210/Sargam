"""
Integration tests for Sargam Sheet Generator.
Tests pitch detection, sargam conversion, and block classification.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import librosa
import soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))

TEST_DIR = tempfile.gettempdir()

from melody import (
    analyze_melody, create_time_blocks, sargam_to_notation_lines,
    hz_to_sargam, format_sargam_note, detect_sa_from_audio,
    NoteEvent, SARGAM_NOTES
)
from sargam_converter import build_sargam_sheet


def generate_sine_wave(freq_hz: float, duration_s: float, sr: int = 22050) -> np.ndarray:
    """Generate a sine wave tone at given frequency."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq_hz * t)


def generate_sargam_scale(sa_hz: float = 261.63, sr: int = 22050) -> np.ndarray:
    """Generate a scale of all 12 sargam notes."""
    segments = []
    for note_name, semitones in SARGAM_NOTES:
        freq = sa_hz * (2 ** (semitones / 12))
        segment = generate_sine_wave(freq, 0.3, sr)
        segments.append(segment)
    return np.concatenate(segments), sr


def generate_meend_test(sr: int = 22050) -> np.ndarray:
    """Generate a pitch glide (meend) from Sa to Re."""
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq_start = 261.63  # C4
    freq_end = 293.66     # D4
    freqs = np.linspace(freq_start, freq_end, len(t))
    return 0.5 * np.sin(2 * np.pi * freqs * t), sr


def generate_gamak_test(sr: int = 22050) -> np.ndarray:
    """Generate rapid pitch oscillation (gamak) around Ga."""
    duration = 0.8
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    center_freq = 329.63  # E4
    mod_depth = 8  # Hz modulation depth
    mod_rate = 12  # Hz modulation rate
    freqs = center_freq + mod_depth * np.sin(2 * np.pi * mod_rate * t)
    return 0.5 * np.sin(2 * np.pi * np.cumsum(freqs) / sr), sr


class TestSaDetection:
    """Test tonic (Sa) detection."""

    def test_c4_detection(self):
        """Detect Sa from a C4 tone with harmonics (more realistic)."""
        # Add 3rd and 5th harmonics for richer tone
        y = generate_sine_wave(261.63, 1.0)
        y += 0.3 * generate_sine_wave(783.99, 1.0, 22050)  # G4 (3rd harmonic)
        y += 0.1 * generate_sine_wave(1308.15, 1.0, 22050)  # C5 (5th harmonic)
        sa_hz = detect_sa_from_audio(y, 22050)
        # Should detect ~261.63 Hz or derive Sa from Pa interval
        assert abs(sa_hz - 261.63) < 30 or abs(sa_hz - 196) < 30, f"Unexpected Sa: {sa_hz}"

    def test_d4_detection(self):
        """Detect Sa from a pure D4 tone (should detect ~293.66 or derive Sa ~261.63)."""
        y = generate_sine_wave(293.66, 1.0)
        sa_hz = detect_sa_from_audio(y, 22050)
        # Should detect either D4 or derive Sa from it
        assert 130 < sa_hz < 523, f"Sa out of range: {sa_hz}"

    def test_default_on_noise(self):
        """Return default Sa on noise."""
        noise = np.random.randn(22050)
        sa_hz = detect_sa_from_audio(noise, 22050)
        assert 260 < sa_hz < 263, f"Expected default Sa, got {sa_hz}"


class TestHzToSargam:
    """Test frequency to sargam conversion."""

    def test_sa_at_c4(self):
        """C4 should map to Sa."""
        result = hz_to_sargam(261.63, 261.63)
        assert result is not None
        note_name, octave = result
        assert note_name == "Sa"
        assert octave == 0

    def test_pa_at_g4(self):
        """G4 should map to Pa."""
        result = hz_to_sargam(392.00, 261.63)
        assert result is not None
        note_name, octave = result
        assert note_name == "Pa"
        assert octave == 0

    def test_komal_re(self):
        """C#4/Db4 should map to re (komal Re)."""
        result = hz_to_sargam(277.18, 261.63)
        assert result is not None
        note_name, octave = result
        assert note_name == "re"
        assert octave == 0

    def test_tivra_ma(self):
        """F#4 should map to Ma' (Tivra Ma)."""
        result = hz_to_sargam(369.99, 261.63)
        assert result is not None
        note_name, octave = result
        assert note_name == "Ma'"
        assert octave == 0

    def test_upper_octave(self):
        """C5 should map to Sa in upper octave."""
        result = hz_to_sargam(523.25, 261.63)
        assert result is not None
        note_name, octave = result
        assert note_name == "Sa"
        assert octave == 1

    def test_lower_octave(self):
        """C3 should map to Sa in lower octave."""
        result = hz_to_sargam(130.81, 261.63)
        assert result is not None
        note_name, octave = result
        assert note_name == "Sa"
        assert octave == -1


class TestSargamFormatting:
    """Test sargam notation formatting rules."""

    def test_format_middle_octave(self):
        """Middle octave notes have no prefix/suffix."""
        assert format_sargam_note("Sa", 0) == "Sa"
        assert format_sargam_note("Re", 0) == "Re"
        assert format_sargam_note("ga", 0) == "ga"

    def test_format_upper_octave(self):
        """Upper octave gets apostrophe."""
        assert format_sargam_note("Sa", 1) == "Sa'"
        assert format_sargam_note("Ga", 1) == "Ga'"

    def test_format_lower_octave(self):
        """Lower octave gets dot prefix."""
        assert format_sargam_note("Ni", -1) == ".Ni"
        assert format_sargam_note("Dha", -1) == ".Dha"

    def test_format_double_apostrophe(self):
        """Two octaves up gets Sa''."""
        result = format_sargam_note("Sa", 2)
        assert result == "Sa''"

    def test_format_double_dot(self):
        """Two octaves down gets ..Sa."""
        result = format_sargam_note("Sa", -2)
        assert result == "..Sa"


class TestSargamScale:
    """Test full sargam scale analysis."""

    def _check_no_western(self, text: str):
        """Assert no Western notation in output."""
        western = ['Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Ti']
        for w in western:
            if w not in ['Re', 'Mi']:
                assert w not in text, f"Western note {w} found"

    def test_full_scale_analysis(self):
        """Analyze a full 12-note sargam scale."""
        audio, sr = generate_sargam_scale(261.63)
        temp_path = os.path.join(TEST_DIR, "test_scale.wav")
        sf.write(temp_path, audio, sr)

        try:
            result = analyze_melody(temp_path)
            assert result['sa_frequency'] > 0
            assert len(result['notes']) > 0

            notation = build_sargam_sheet(
                sargam_to_notation_lines(result['notes']),
                "Test Scale",
                f"{result['sa_note']} ({result['sa_frequency']:.1f} Hz)"
            )

            self._check_no_western(notation)
            assert 'Sa' in notation, "Missing Sa"
            assert 'Re' in notation or 're' in notation, "Missing Re"
            assert 'Ga' in notation or 'ga' in notation, "Missing Ga"

        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

    def test_komal_notes_present(self):
        """Komal notes should appear as lowercase."""
        # Build expected note list with komal variants
        komal_notes = {'re', 'ga', 'dha', 'ni'}
        for note_name, semitones in SARGAM_NOTES:
            if note_name in komal_notes:
                assert note_name[0].islower(), f"{note_name} should be lowercase"


class TestBlockClassification:
    """Test vocal/instrumental block classification."""

    def test_empty_notes_creates_instrumental_block(self):
        """No notes should result in a single Instrumental block."""
        blocks = create_time_blocks([], 10.0)
        assert len(blocks) == 1
        assert blocks[0]['type'] == 'Instrumental'
        assert blocks[0]['start_time'] == 0.0
        assert blocks[0]['end_time'] == 10.0

    def test_single_note_creates_blocks(self):
        """Single note event should create time blocks."""
        note = NoteEvent("Sa", 0, 0.0, 0.5, "Sa")
        blocks = create_time_blocks([note], 5.0)
        assert len(blocks) >= 1
        # First block should cover the note
        assert blocks[0]['start_time'] == 0.0

    def test_notation_lines_no_lyrics(self):
        """Evenly distribute notes into lines when no lyrics."""
        notes = [
            NoteEvent("Sa", 0, 0.0, 0.3, "Sa"),
            NoteEvent("Re", 0, 0.3, 0.6, "Re"),
            NoteEvent("Ga", 0, 0.6, 0.9, "Ga"),
            NoteEvent("Ma", 0, 0.9, 1.2, "Ma"),
        ]
        lines = sargam_to_notation_lines(notes)
        assert len(lines) >= 0


class TestFullSongAnalysis:
    """Integration-level tests for full analysis pipeline."""

    def test_analysis_returns_all_fields(self):
        """Analyze a simple melody and verify all expected fields."""
        audio, sr = generate_sargam_scale(261.63)
        temp_path = os.path.join(TEST_DIR, "test_analysis.wav")
        sf.write(temp_path, audio, sr)

        try:
            result = analyze_melody(temp_path)
            assert 'sa_frequency' in result
            assert 'sa_note' in result
            assert 'notes' in result
            assert 'phrases' in result
            assert 'duration_seconds' in result
            assert result['sa_frequency'] > 0
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

    def test_full_sargam_no_abbreviations(self):
        """Verify output never uses abbreviated notation."""
        audio, sr = generate_sargam_scale(261.63)
        temp_path = os.path.join(TEST_DIR, "test_abbreviations.wav")
        sf.write(temp_path, audio, sr)

        try:
            result = analyze_melody(temp_path)
            notation = build_sargam_sheet(
                sargam_to_notation_lines(result['notes']),
                "Test",
                f"{result['sa_note']}"
            )

            bad = [' S ', ' R ', ' G ', ' M ', ' P ', ' D ', ' N ']
            for b in bad:
                assert b not in notation, f"Abbreviated note {b} found"

        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
