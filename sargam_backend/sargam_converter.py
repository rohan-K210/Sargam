"""
Sargam notation converter - transforms melody analysis into formatted output.
Handles formatting rules, lyrics alignment, and output generation.
"""

from typing import List, Dict


def build_sargam_sheet(notation_lines: List[Dict], song_name: str = "", 
                       sa_info: str = "") -> str:
    """
    Build a complete text representation of the sargam sheet.
    """
    lines = []
    
    if song_name:
        lines.append(f"\U0001f3b5 {song_name}")
        lines.append("=" * 50)
    
    if sa_info:
        lines.append(f"\U0001f3bc Sa = {sa_info}")
        lines.append("-" * 50)
    
    lines.append("")
    
    for i, line in enumerate(notation_lines):
        lyrics = line.get('lyrics', '')
        sargam = line.get('sargam', '')
        
        if lyrics:
            lines.append(f"Lyrics: {lyrics}")
        
        if sargam:
            lines.append(f"Sargam: {sargam}")
        
        lines.append("")
    
    return '\n'.join(lines)


def generate_gemini_sargam_analysis_prompt(
    raw_notes: List[Dict],
    lyrics: str = "",
    song_name: str = "",
    sa_candidates: List[Dict] = None,
    beat_times: List[float] = None,
    downbeat_times: List[float] = None,
    ornaments_data: Dict = None,
    duration_seconds: float = 0.0
) -> str:
    """
    Generate a prompt for Gemini to analyze the detected notes and produce
    a clean sargam transcription organized by time blocks.
    
    Enhanced with:
    - Beat-aligned note events (onset beat number, position in bar)
    - Ornament markers (meend/gamak)
    - Vocal vs accompaniment energy per block
    - Multiple Sa candidates with confidence scores
    - Song structure hints (verse/chorus from downbeats)
    - Output strict JSON with time blocks covering entire duration
    """
    notes_text = []
    for n in raw_notes[:300]:
        if isinstance(n, dict):
            extra = ""
            if n.get('beat_number', -1) >= 0:
                extra += f" (beat {n['beat_number']}, pos {n.get('position_in_bar', '?')}/4)"
            if n.get('ornament'):
                extra += f" [{n['ornament']}]"
            notes_text.append(
                f"[{n.get('start_time', 0):.2f}s-{n.get('end_time', 0):.2f}s]{extra} {n.get('note', n.get('display', '?'))}"
            )
    
    notes_str = '\n'.join(notes_text)
    
    lyrics_section = ""
    if lyrics:
        # Limit lyrics to 1000 chars
        lyrics_truncated = lyrics[:1000]
        lyrics_section = f"\n**Lyrics provided (may need alignment):**\n{lyrics_truncated}\n"
    
    # Sa candidates section
    sa_candidates_section = ""
    if sa_candidates:
        sa_lines = []
        for sc in sa_candidates[:5]:
            sa_lines.append(f"  - {sc.get('note_name', '?')} ({sc.get('frequency', 0):.1f} Hz) — confidence: {sc.get('confidence', 0):.2f}")
        sa_candidates_section = "\n**Detected Sa (tonic) candidates (ranked by confidence):**\n" + "\n".join(sa_lines) + "\n"
    
    # Beat structure section
    beat_section = ""
    if beat_times:
        beat_count = len(beat_times)
        downbeat_count = len(downbeat_times) if downbeat_times else 0
        beat_section = f"\n**Beat tracking:** {beat_count} beats detected, {downbeat_count} downbeats (structure hints: {' / '.join([f'{t:.1f}s' for t in downbeat_times[:8]])})\n"
    
    # Ornaments section
    ornament_section = ""
    if ornaments_data:
        meend_count = len(ornaments_data.get('meend', []))
        gamak_count = len(ornaments_data.get('gamak', []))
        if meend_count > 0 or gamak_count > 0:
            ornament_section = f"\n**Ornaments detected:** {meend_count} meends (glides), {gamak_count} gamaks (oscillations). Mark with ~ for meend and ~~~ for gamak.\n"
    
    duration_info = f"\n**Total duration for analysis:** {duration_seconds:.1f} seconds (limited to 60s).\n" if duration_seconds > 0 else ""
    
    prompt = f"""You are an expert in Indian classical music (Hindustani/Carnatic) transcription.

{song_name and f'A song titled "{song_name}"' or 'A song'} has been analyzed and the following melodic notes were detected with timestamps:
{sa_candidates_section}
{beat_section}
{ornament_section}
{duration_info}

**Detected notes:**
{notes_str}
{lyrics_section}
## Task: Convert to Time-Blocked Sargam Notation

Your job is to organize the entire song into **continuous time blocks from start to finish**. Every second must be covered — no gaps.

### Step 1: Divide into time blocks
Group the detected notes into meaningful time blocks based on the music structure. Each block should be 2-10 seconds long. Use the downbeat positions as hints for section boundaries (verse/chorus/instrumental).

### Step 2: Classify each block
For each block, determine whether it is:
- "Vocal" — the singer is singing (melodic, varied notes with lyrics)
- "Instrumental" — instrumental passage, filler, or pause (sparse, repetitive, or no notes)

### Step 3: Convert to Indian Sargam
For each block, convert the notes to proper Indian Sargam notation:
- Full note names: Sa Re Ga Ma Pa Dha Ni
- Komal (flat) notes: re, ga, dha, ni (lowercase)
- Tivra Ma: Ma'
- Upper octave: add apostrophe (Sa', Re', Ga', Ma', Pa', Dha', Ni')
- Lower octave: add dot prefix (.Ni, .Dha, .Pa, .Ma, .Ga, .Re, .Sa)
- Long held notes: ALL CAPS (SA, RE, GA, MA, PA, DHA, NI)
- Dynamics: (soft) or (loud) at end of relevant notes
- Ornaments: add ~ after notes with meend (Sa~), add ~~~ after notes with gamak (Ga~~~)

### Step 4: Provide lyrics for vocal blocks
If the block is "Vocal", provide the most likely lyrics for that time range.
Use the provided lyrics if available and applicable.

### Step 5: Beat alignment
For each note in the sargam, indicate the beat position when possible:
- Use | for bar lines every 4 beats
- Use - for continued notes within a beat

### Output Format
Return ONLY valid JSON (no markdown, no code fences, no extra text):
{{
  "blocks": [
    {{
      "start_time": 0.0,
      "end_time": 2.0,
      "type": "Instrumental",
      "sargam": "Sa Re Ga Ma Pa Dha Ni",
      "lyrics": "",
      "structure_hint": "intro"
    }},
    {{
      "start_time": 2.0,
      "end_time": 5.0,
      "type": "Vocal",
      "sargam": "SA RE ga MA Pa Dha",
      "lyrics": "Tum hi ho",
      "structure_hint": "verse"
    }}
  ],
  "song_duration": 30.0,
  "recommended_sa": "C4 (261.6 Hz)",
  "confidence": "high"
}}

Cover the ENTIRE song duration. No gaps. Each block's start_time should match the previous block's end_time.
The output_time is {duration_seconds:.1f} seconds — ensure the last block covers up to this time."""
    
    return prompt
