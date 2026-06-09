from flask import Flask, request, jsonify
import librosa
import numpy as np
from google import genai
import os
import tempfile

app = Flask(__name__)

# Put your Gemini API key here
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
    y, sr = librosa.load(filepath, duration=60)  # max 60 seconds

    # Pitch detection
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512)

    notes_data = []
    times = librosa.times_like(pitches, sr=sr, hop_length=512)

    for i in range(0, pitches.shape[1], 5):  # sample every 5 frames
        idx = magnitudes[:, i].argmax()
        pitch = pitches[idx, i]
        magnitude = magnitudes[idx, i]

        if pitch > 80 and magnitude > 0.01:  # filter noise
            note = hz_to_note(pitch)
            if note:
                notes_data.append({
                    "time": round(float(times[i]), 2),
                    "note": note,
                    "loudness": round(float(magnitude), 3)
                })

    return notes_data[:200]  # send max 200 notes to Gemini

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

@app.route('/convert', methods=['POST'])
def convert_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']

    # Save to temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Analyze audio
        notes_data = analyze_audio(tmp_path)

        if not notes_data:
            return jsonify({"error": "Could not detect any notes in audio"}), 400

        # Call Gemini
        prompt = build_gemini_prompt(notes_data)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return jsonify({
            "success": True,
            "sargam": response.text,
            "notes_detected": len(notes_data)
        })

    finally:
        os.unlink(tmp_path)  # clean up temp file

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)