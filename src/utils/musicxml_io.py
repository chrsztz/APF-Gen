from typing import List, Tuple, Any

from music21 import converter

from src.data.parser import NoteEvent
from src.utils.midi_io import parse_midi
import tempfile
import os

def parse_musicxml(path: str) -> Tuple[List[NoteEvent], Any]:
    """
    Parse MusicXML by converting to MIDI first to ensure identical timing/seconds handling
    as the training data format.
    Returns (events, music21_score_object).
    """
    # 1. Load MusicXML via music21
    score = converter.parse(path)
    
    # 2. Write to a temporary MIDI file
    # We use a temp file because parse_midi expects a path
    fd, temp_midi_path = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    
    try:
        score.write("midi", fp=temp_midi_path)
        
        # 3. Parse that MIDI file using our robust MIDI parser
        events = parse_midi(temp_midi_path)
        
    finally:
        # Clean up
        if os.path.exists(temp_midi_path):
            os.remove(temp_midi_path)
            
    return events, score


def write_fingerings_to_musicxml(score, fingerings: List[int], events: List[NoteEvent], out_path: str):
    """
    Write predicted fingerings back to a MusicXML file.
    This is tricky because 'events' come from a flattened MIDI representation.
    We need to match them back to the music21 score objects.
    """
    # Create a mapping of (onset_sec, pitch_midi) -> fingering
    # Note: onset times might float-drift, so we used a tolerance or fuzzy match if needed.
    # However, since we just converted score->midi->events, the seconds SHOULD align closely.
    
    # Let's build a map:
    # key: (approx_onset, midi) -> finger
    # We'll truncate onset to 3 decimal places for matching key
    finger_map = {}
    for ev, f in zip(events, fingerings):
        key = (round(ev.onset, 3), ev.midi)
        # Handle chords/collisions: append or overwrite?
        # A simple overwrite is usually okay for single note melodies.
        # But for chords, multiple notes share onset. the key includes midi, so it is unique per note.
        finger_map[key] = f
        
    # Traverse the score and apply fingerings
    # We need secondsMap again to get absolute timing of notes in the score structure
    try:
        # score.flat.secondsMap is safe
        sec_map = score.flat.secondsMap
    except:
        # If flat fails, just recurse (but timing is hard without secondsMap)
        print("Warning: Could not generate secondsMap for writing. Fingerings may be skipped.")
        return

    from music21 import articulations

    for item in sec_map:
        el = item['element']
        onset = item['offsetSeconds']
        
        if 'Note' in el.classes:
            midi_val = el.pitch.midi
            key = (round(onset, 3), midi_val)
            if key in finger_map:
                f = finger_map[key]
                # Apply fingering
                # In music21, fingering is an articulation
                fg = articulations.Fingering(f)
                el.articulations.append(fg)
                
        elif 'Chord' in el.classes:
            for p in el.pitches:
                midi_val = p.midi
                key = (round(onset, 3), midi_val)
                if key in finger_map:
                    f = finger_map[key]
                    # Chords in music21 can have articulations, but per-note fingering is... tricky.
                    # Music21 often attaches articulations to the Chord object, confusing which note it applies to.
                    # But per-note fingering is supported in XML.
                    # music21's Chord doesn't easily support per-pitch-articulation.
                    # We might just attach it to the chord and hope the layout works, or skip complex chords for now.
                    # Alternative: Text annotation?
                    fg = articulations.Fingering(f)
                    el.articulations.append(fg)

    score.write("musicxml", fp=out_path)
