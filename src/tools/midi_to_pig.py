import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.utils.midi_io import parse_midi
from src.data.parser import NoteEvent

def midi_to_pig(input_path: str, output_path: str):
    """
    Parse a MIDI file using the system's parse_midi function and write to PIG format.
    
    PIG format specification:
        (note ID) (onset time) (offset time) (spelled pitch) (onset velocity) (offset velocity) (channel) (finger number)
        
    Where:
        - onset/offset time: in seconds (based on MIDI tempo)
        - spelled pitch: e.g., C4, F#3, Ab5 (A4 ≃ 440Hz)
        - onset velocity: dynamic from MIDI data
        - offset velocity: fixed at 80
        - channel: 0 = right hand, 1 = left hand
        - finger: 0 for unassigned (to be filled by fingering model)
    """
    events = parse_midi(input_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Write header matching PIG dataset format
        f.write("//Version: PianoFingering_v170101_Generated\n")
        
        for ev in events:
            # Format: (note ID) (onset) (offset) (pitch) (vel_on) (vel_off) (channel) (finger)
            # Tab-separated values as per PIG dataset specification
            line = f"{ev.idx}\t{ev.onset:.6g}\t{ev.offset:.6g}\t{ev.pitch_str}\t" \
                   f"{ev.vel_on}\t{ev.vel_off}\t{ev.channel}\t{ev.finger}\n"
            f.write(line)
            
    print(f"Parsed {input_path} -> {output_path} with {len(events)} events.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input MIDI file")
    parser.add_argument("--output", required=True, help="Output PIG txt file")
    args = parser.parse_args()
    
    midi_to_pig(args.input, args.output)
