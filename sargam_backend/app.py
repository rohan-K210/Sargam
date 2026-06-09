import gradio as gr
import librosa
import numpy as np
from google import genai
import os

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def hz_to_note(frequency):
    if frequency <= 0:
        return None
    A4 = 440.0
    semitone = 12 * np.log2(frequency / A4)
    note_number = round(semitone) % 12
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    return notes[note_number % 12]

def analyze_audio(filepath):
    y, sr = librosa.load(filepath, duration=60)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512)
    notes_data = []
    times = librosa.times_like(pitches, sr=sr, hop_length=512)

    for i in range(0, pitches.shape[1], 5):
        idx = magnitudes[:, i].argmax()
        pitch = pitches[idx, i]
        magnitude = magnitudes[idx, i]
        if pitch > 80 and magnitude > 0.01:
            note = hz_to_note(pitch)
            if note:
                notes_data.append({
                    "time": round(float(times[i]), 2),
                    "note": note,
                    "loudness": round(float(magnitude), 3)
                })

    return notes_data[:200]

def build_gemini_prompt(notes_data):
    notes_str = ", ".join([f"{n['note']}({n['time']}s, vol:{n['loudness']})" for n in notes_data])
    prompt = f"""
You are an Indian classical music expert. Convert these Western music notes to Indian Sargam notation for a flute player.

Notes detected (note, time in seconds, volume): {notes_str}

Convert this to Sargam notation with these rules:
- Map notes: C=Sa, D=Re, E=Ga, F=Ma, G=Pa, A=Dha, B=Ni
- Mark Komal (flat) notes with lowercase: re, ga, dha, ni
- Mark Tivra Ma with Ma'
- Group notes into meaningful phrases
- Show loudness as (soft), (medium), (loud)
- Show note duration based on time gaps

Return the output in this format:
1. SARGAM TEXT: Write the full notation line by line
2. PHRASES: Break into musical phrases (4-8 notes each)
3. TIPS FOR FLUTE: Any specific fingering or breathing tips

Keep it clear and simple for a beginner flute player.
"""
    return prompt

def convert_to_sargam(audio_file):
    try:
        if audio_file is None:
            return "Please upload an audio file."

        notes_data = analyze_audio(audio_file)

        if not notes_data:
            return "Could not detect any notes in the audio. Please try a clearer recording."

        prompt = build_gemini_prompt(notes_data)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=convert_to_sargam,
    inputs=gr.Audio(type="filepath", label="Upload Song (MP3, WAV, etc.)"),
    outputs=gr.Textbox(label="Sargam Sheet", lines=20),
    title="🎵 Sargam Sheet Generator",
    description="Upload any song and get Sa Re Ga Ma Pa notation for flute players!"
)

demo.launch()
