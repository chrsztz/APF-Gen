from typing import List

from music21 import converter

from src.data.parser import NoteEvent

# Simple heuristic split pitch (middle C = 60)
DEFAULT_SPLIT_PITCH = 60


def _append_event(events: List[NoteEvent], onset: float, dur: float, midi: int, pitch_str: str, channel: int):
    offset = onset + dur
    ev = NoteEvent(
        idx=len(events),
        onset=onset,
        offset=offset,
        pitch_str=pitch_str,
        midi=midi,
        vel_on=64,
        vel_off=64,
        channel=channel,
        finger=0,
    )
    events.append(ev)


def parse_midi(path: str, split_pitch: int = DEFAULT_SPLIT_PITCH) -> List[NoteEvent]:
    """
    Parse MIDI file to NoteEvent list.
    If multiple parts, map first part to RH (0), second to LH (1).
    If single part, split by pitch threshold.
    Expands chords into separate notes sharing onset/duration.
    """
    score = converter.parse(path)
    events: List[NoteEvent] = []
    parts = score.parts if score.parts else [score.flat]

    if len(parts) >= 2:
        part_map = {0: 0, 1: 1}
        for p_idx, part in enumerate(parts[:2]):
            channel = part_map.get(p_idx, 0)
            for n in part.recurse().notes:
                onset = float(n.offset)
                dur = float(n.duration.quarterLength)
                if getattr(n, "isChord", False):
                    for p in n.pitches:
                        _append_event(events, onset, dur, int(p.midi), p.nameWithOctave, channel)
                else:
                    _append_event(events, onset, dur, int(n.pitch.midi), n.pitch.nameWithOctave, channel)
    else:
        for n in score.flat.notes:
            onset = float(n.offset)
            dur = float(n.duration.quarterLength)
            if getattr(n, "isChord", False):
                for p in n.pitches:
                    channel = 0 if p.midi >= split_pitch else 1
                    _append_event(events, onset, dur, int(p.midi), p.nameWithOctave, channel)
            else:
                midi = int(n.pitch.midi)
                pitch_str = n.pitch.nameWithOctave
                channel = 0 if midi >= split_pitch else 1
                _append_event(events, onset, dur, midi, pitch_str, channel)

    events.sort(key=lambda e: (e.onset, e.midi))
    for i, ev in enumerate(events):
        ev.idx = i
    return events

