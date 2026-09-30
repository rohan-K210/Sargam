"""
Gemini TTS integration for singing sargam notation.
Uses Gemini 2.5 Flash TTS Preview to generate audio with correct pitch and melody.
"""

import os
from typing import Optional
from google import genai
from google.genai import types


def get_gemini_client():
    """Initialize Gemini client with API key."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def sing_sargam_line(sargam_text: str, lyrics: str = "",
                     voice_name: str = "Kore") -> Optional[bytes]:
    """
    Generate singing audio for a sargam line using Gemini TTS.
    
    Args:
        sargam_text: The sargam notation text to sing
        lyrics: Optional lyrics context
        voice_name: Gemini TTS voice to use
    
    Returns:
        WAV audio bytes, or None if generation fails
    """
    client = get_gemini_client()
    if client is None:
        return None
    
    try:
        # Build a descriptive prompt for the singing
        if lyrics:
            prompt = (
                f"Sing these Indian sargam notes for the phrase '{lyrics}': {sargam_text}. "
                f"Sing each note clearly at its correct pitch. "
                f"Make it sound like a melodic vocal warm-up with proper Sa Re Ga Ma intonation. "
                f"Keep a steady tempo. Sing only the note names, not the lyrics."
            )
        else:
            prompt = (
                f"Sing these Indian classical sargam notes: {sargam_text}. "
                f"Sing each note clearly at its correct pitch - Sa, Re, Ga, Ma, Pa, Dha, Ni. "
                f"Make it sound melodic and clear for a flute student to follow. "
                f"Keep a moderate tempo."
            )
        
        # Use the TTS model
        model_name = "gemini-2.5-flash-native-audio-preview"
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            )
        )
        
        # Extract audio data from response
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type and "audio" in part.inline_data.mime_type:
                return part.inline_data.data
        
        return None
        
    except Exception as e:
        print(f"Gemini TTS error: {e}")
        return None



