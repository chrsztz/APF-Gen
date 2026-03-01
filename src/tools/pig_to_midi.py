import argparse
import os
import sys

import pretty_midi

# Add src to path to import local modules if needed, but we will try to be standalone or import correctly
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.data.parser import parse_fingering_file

def pig_to_midi(input_path: str, output_path: str, initial_tempo: float = 120.0):
    """
    Convert a PIG txt file to a MIDI file using pretty_midi.
    
    Args:
        input_path: Path to input PIG txt file
        output_path: Path to output MIDI file
        initial_tempo: Initial tempo in BPM (default: 120.0)
        
    PIG format:
        (note ID) (onset time) (offset time) (spelled pitch) (onset velocity) (offset velocity) (channel) (finger number)
        - Times are in seconds
        - Channel: 0 = right hand, 1 = left hand
    """
    piece_id, events = parse_fingering_file(input_path)
    
    # Create a PrettyMIDI object with initial tempo
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
    
    # Create two instruments: one for right hand, one for left hand
    # Program 0 = Acoustic Grand Piano
    instrument_rh = pretty_midi.Instrument(program=0, name="Right Hand")
    instrument_lh = pretty_midi.Instrument(program=0, name="Left Hand")
    
    # Sort events by onset time to ensure proper ordering
    events.sort(key=lambda x: x.onset)
    
    # Process each event and add to appropriate instrument
    for ev in events:
        # Create a Note object
        # pretty_midi.Note(velocity, pitch, start, end)
        # Times are in seconds - perfect match for our PIG format!
        note = pretty_midi.Note(
            velocity=ev.vel_on,
            pitch=ev.midi,
            start=ev.onset,
            end=ev.offset
        )
        
        # Add to appropriate instrument based on channel
        if ev.channel == 0:
            instrument_rh.notes.append(note)
        else:
            instrument_lh.notes.append(note)
    
    # Add instruments to the MIDI data
    # Only add instruments that have notes
    if instrument_rh.notes:
        midi_data.instruments.append(instrument_rh)
    if instrument_lh.notes:
        midi_data.instruments.append(instrument_lh)
    
    # Write to MIDI file
    midi_data.write(output_path)
    
    print(f"Converted {input_path} -> {output_path}")
    print(f"  Total events: {len(events)}")
    print(f"  Right hand notes: {len(instrument_rh.notes)}")
    print(f"  Left hand notes: {len(instrument_lh.notes)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input PIG txt file")
    parser.add_argument("--output", required=True, help="Output MIDI file")
    args = parser.parse_args()
    
    pig_to_midi(args.input, args.output)
