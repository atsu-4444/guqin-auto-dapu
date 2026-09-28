import pandas as pd
import numpy as np
import json
import os
from midiutil import MIDIFile
from music21 import converter

# New Gemini SDK
from google import genai

# =================================================================
# Initialization & Setup
# =================================================================

# Retrieve API key from terminal environment variables
api_key = os.environ.get("GOOGLE_API_KEY")

try:
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        # Fallback if the environment variable is not set
        client = genai.Client()
except Exception as e:
    print(f"Warning: Failed to initialize Gemini Client. Error: {e}")
    client = None

SEMITONES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

# =================================================================
# Helper Functions
# =================================================================

def _quantize_duration(duration, step=0.25):
    return round(duration / step) * step

def _pitch_to_midi_note(pitch_name):
    if not pitch_name or not isinstance(pitch_name, str) or len(pitch_name) < 2:
        return None
    
    pitch_name = pitch_name.strip().upper()
    note_letter = pitch_name[0]
    if note_letter not in SEMITONES: return None
    
    accidental, octave_start_index = 0, 1
    if len(pitch_name) > 1:
        if pitch_name[1] == '#': accidental, octave_start_index = 1, 2
        elif pitch_name[1] == 'B': accidental, octave_start_index = -1, 2
        
    try:
        octave = int(pitch_name[octave_start_index:])
        return 12 * (octave + 1) + SEMITONES[note_letter] + accidental
    except (ValueError, IndexError):
        return None

def _calculate_pitch(string, hui, sound_type, tuning_map):
    try:
        hui_for_key = hui if isinstance(hui, str) and not hui.replace('.', '').isdigit() else (int(float(hui)) if float(hui) == int(float(hui)) else float(hui))
        key = f"{sound_type}_string{int(string)}_hui{hui_for_key}"
        return tuning_map.get(key, None)
    except (ValueError, TypeError):
        return None

# =================================================================
# AI Music Expression Generation
# =================================================================

def generate_music_expression_with_ai(title, lyrics, note_ids):
    """
    Generate musical expression (tempo, duration, velocity) using LLM based on song context.
    """
    if not client:
        print("Gemini API key is missing. Returning default values.")
        return 60, "API Key Missing", [(nid, 1.0, 80) for nid in note_ids]

    prompt = f"""
You are a professional Guqin player. Please generate performance data based on the following information.

# Song Information
- Title: "{title}"
- Lyrics/Image: "{lyrics}"
- Number of notes: {len(note_ids)}

# Rules
1. Determine the tempo (BPM) between 40 and 120.
2. Determine the "duration" (0.25-2.0) and "velocity" (30-120) for each ID.
3. Output ONLY in JSON format:
{{
  "tempo": integer,
  "description": "A short description of the atmosphere (in Japanese)",
  "notes": [ {{ "id": "note ID", "duration": number, "velocity": number }} ]
}}

# Target ID List
{json.dumps(note_ids)}
"""

    try:
        # Generate content using the new SDK method
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        text = response.text.strip()
        json_str = text.split('```json')[1].split('```')[0] if '```json' in text else text.split('```')[1].split('```')[0] if '```' in text else text
            
        parsed_data = json.loads(json_str)
        ai_notes_map = {item['id']: item for item in parsed_data.get('notes', [])}
        
        note_sequence = [(nid, ai_notes_map.get(nid, {}).get('duration', 1.0), ai_notes_map.get(nid, {}).get('velocity', 80)) for nid in note_ids]
        return parsed_data.get('tempo', 60), parsed_data.get('description', "AI generation"), note_sequence

    except Exception as e:
        print(f"AI Generation Error: {e}")
        return 60, "Fallback setting due to error", [(nid, 1.0, 80) for nid in note_ids]

# =================================================================
# Main MIDI Pipeline
# =================================================================

def generate_midi_file(jzp_ids, title, lyrics, output_path, csv_path, json_path):
    """
    Map recognized sequence to frequencies and generate a MIDI file.
    """
    try:
        df = pd.read_csv(csv_path, encoding='cp932')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='utf-8')
        
    with open(json_path, 'r', encoding='utf-8') as f:
        tuning_map = json.load(f)

    if 'id' in df.columns:
        df.set_index('id', inplace=True)
    
    song_tempo, description, note_sequence = generate_music_expression_with_ai(title, lyrics, jzp_ids)
    
    last_string, last_hui, last_sound_type = None, None, None
    final_score = []
    
    for note_id, _, _ in note_sequence:
        if note_id not in df.index:
            final_score.append(None)
            continue
            
        nd = df.loc[note_id]
        c_sound = nd.get('sound_type') if pd.notna(nd.get('sound_type')) and nd.get('sound_type') != 'context' else ('an\'yin' if last_sound_type == 'an\'yin' else 'san\'yin')
        c_hui = nd.get('hui') if pd.notna(nd.get('hui')) and str(nd.get('hui')) != 'null' else (last_hui if last_sound_type == 'an\'yin' else 0)
        c_string = nd.get('string') if pd.notna(nd.get('string')) else last_string
            
        pitch = nd.get('pitch') if 'pitch' in nd and pd.notna(nd['pitch']) else None
        
        if pd.isna(pitch):
            if str(c_sound).startswith("an'yin") and isinstance(c_string, str) and ',' in c_string:
                s_list, h_list, st_list = c_string.split(','), str(c_hui).split(','), str(c_sound).split(',')
                max_len = max(len(s_list), len(h_list), len(st_list))
                pitches = [_calculate_pitch(s.strip(), h.strip(), st.strip(), tuning_map) for s, h, st in zip((s_list*max_len)[:max_len], (h_list*max_len)[:max_len], (st_list*max_len)[:max_len])]
                pitch = ','.join([p for p in pitches if p]) if any(pitches) else None
            else:
                pitch = _calculate_pitch(c_string, c_hui, c_sound, tuning_map)
        
        final_score.append(pitch.split(',') if isinstance(pitch, str) and ',' in pitch else pitch)
        last_string, last_hui, last_sound_type = c_string, c_hui, c_sound

    midi = MIDIFile(1)
    midi.addTempo(0, 0, song_tempo)
    midi.addProgramChange(0, 0, 0, 106)
    midi.addTrackName(0, 0, title.encode('latin-1', errors='ignore').decode('latin-1'))

    current_time = 0
    for i, (note_id, duration, volume) in enumerate(note_sequence):
        note_info = final_score[i]
        q_dur = _quantize_duration(duration)
        
        if note_info is None:
            current_time += q_dur
            continue
            
        radical = str(df.loc[note_id].get('radical_name', '')) if note_id in df.index else ""
        if '綽' in radical:
            midi.addPitchWheelEvent(0, 0, current_time, -4096)
            midi.addPitchWheelEvent(0, 0, current_time + 0.2, 0)
        elif '注' in radical:
            midi.addPitchWheelEvent(0, 0, max(current_time, current_time + q_dur - 0.25), -4096)
            midi.addPitchWheelEvent(0, 0, current_time + q_dur, 0)
        else:
            midi.addPitchWheelEvent(0, 0, current_time, 0)

        for p in (note_info if isinstance(note_info, list) else [note_info]):
            midi_p = _pitch_to_midi_note(p)
            if midi_p: midi.addNote(0, 0, midi_p, current_time, q_dur, volume)
        
        current_time += q_dur

    with open(output_path, "wb") as f:
        midi.writeFile(f)

    return description

def open_score_viewer(midi_path):
    """
    Launch default music notation software (e.g., MuseScore) to view MIDI.
    """
    if os.path.exists(midi_path):
        converter.parse(midi_path).show()