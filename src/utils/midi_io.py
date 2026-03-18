"""MIDI parsing with rich metadata extraction for accurate MusicXML export."""

from typing import Any, Dict, List, Tuple

import pretty_midi

from src.data.parser import NoteEvent

DEFAULT_SPLIT_PITCH = 60
OFFSET_VELOCITY = 80

# Chromatic pitch class → circle-of-fifths position (for the major key on that root)
CHROMATIC_TO_FIFTHS = [0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5]


def _midi_number_to_pitch_string(midi_num: int) -> str:
    # Match C++ PitchToSitch() convention used to generate PIG dataset files:
    # pitch class 3 → "Eb" (not "D#"), pitch class 10 → "Bb" (not "A#")
    note_names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
    octave = (midi_num // 12) - 1
    note = note_names[midi_num % 12]
    return f"{note}{octave}"


def _append_event(
    events: List[NoteEvent],
    onset: float,
    offset: float,
    midi_val: int,
    channel: int,
    vel_on: int,
):
    pitch_str = _midi_number_to_pitch_string(midi_val)
    ev = NoteEvent(
        idx=len(events),
        onset=onset,
        offset=offset,
        pitch_str=pitch_str,
        midi=midi_val,
        vel_on=vel_on,
        vel_off=OFFSET_VELOCITY,
        channel=channel,
        finger=0,
    )
    events.append(ev)


def _read_non_drum_channels(path: str) -> List[int]:
    """Return actual MIDI channel numbers for non-drum instruments in first-seen order.

    pretty_midi builds its instrument list in the order channels are first encountered,
    so this list aligns index-for-index with pretty_midi's non-drum instrument list.
    Uses mido, which is a hard dependency of pretty_midi.
    """
    import mido
    seen: List[int] = []
    try:
        mid = mido.MidiFile(path)
        for track in mid.tracks:
            for msg in track:
                ch = getattr(msg, 'channel', None)
                if ch is None or ch == 9:  # skip meta messages and drum channel
                    continue
                if msg.type in ('note_on', 'note_off') and ch not in seen:
                    seen.append(ch)
    except Exception:
        pass
    return seen


def _parse_notes(
    midi_data: pretty_midi.PrettyMIDI, split_pitch: int,
    midi_channels: List[int] = None,
) -> List[NoteEvent]:
    """Extract NoteEvent list from PrettyMIDI (timestamps in seconds).

    midi_channels: actual MIDI channel numbers indexed by non-drum instrument position,
    obtained via _read_non_drum_channels(). Mirrors C++ PianoRoll::ReadMIDIFile which
    uses (status_byte % 16) as the channel field written into PIG .txt files.
    """
    events: List[NoteEvent] = []
    non_drum = [inst for inst in midi_data.instruments if not inst.is_drum]
    has_multiple = len(non_drum) >= 2

    for inst_idx, instrument in enumerate(non_drum):
        if has_multiple:
            if midi_channels and inst_idx < len(midi_channels):
                channel_id = midi_channels[inst_idx]
            else:
                channel_id = inst_idx  # fallback: assume index == channel
        else:
            channel_id = -1

        for note in instrument.notes:
            ch = (
                channel_id
                if channel_id >= 0
                else (0 if note.pitch >= split_pitch else 1)
            )
            _append_event(events, note.start, note.end, note.pitch, ch, note.velocity)

    events.sort(key=lambda e: (e.onset, e.midi))
    for i, ev in enumerate(events):
        ev.idx = i
    return events


def _extract_midi_meta(
    midi_data: pretty_midi.PrettyMIDI, events: List[NoteEvent]
) -> Dict[str, Any]:
    """Extract rich metadata from MIDI for accurate MusicXML export.

    Returns dict with:
        resolution: int – MIDI ticks per quarter note
        onset_ticks: List[float] – per-note onset in MIDI ticks (tempo-aware)
        offset_ticks: List[float] – per-note offset in MIDI ticks
        tempo_changes: List[(tick, bpm)]
        time_sig_changes: List[(tick, numerator, denominator)]
        key_sig_changes: List[(tick, fifths, mode_str)]
    """
    resolution = midi_data.resolution

    # Per-note tick positions via the authoritative tempo map
    onset_ticks = [midi_data.time_to_tick(ev.onset) for ev in events]
    offset_ticks = [midi_data.time_to_tick(ev.offset) for ev in events]

    # --- Tempo changes ---
    tempo_changes: List[Tuple[float, float]] = []
    try:
        times, tempi = midi_data.get_tempo_changes()
        for t, bpm in zip(times, tempi):
            tempo_changes.append((midi_data.time_to_tick(t), float(bpm)))
    except Exception:
        pass
    if not tempo_changes:
        tempo_changes = [(0.0, 120.0)]

    # --- Time signature changes ---
    time_sig_changes: List[Tuple[float, int, int]] = []
    for ts in midi_data.time_signature_changes:
        tick = midi_data.time_to_tick(ts.time)
        time_sig_changes.append((tick, int(ts.numerator), int(ts.denominator)))
    if not time_sig_changes:
        time_sig_changes = [(0.0, 4, 4)]

    # --- Key signature changes ---
    key_sig_changes: List[Tuple[float, int, str]] = []
    for ks in midi_data.key_signature_changes:
        tick = midi_data.time_to_tick(ks.time)
        key_number = int(ks.key_number)
        if key_number >= 12:
            # Minor key
            pc = key_number - 12
            rel_major_pc = (pc + 3) % 12
            fifths = CHROMATIC_TO_FIFTHS[rel_major_pc]
            mode = "minor"
        else:
            fifths = CHROMATIC_TO_FIFTHS[key_number]
            mode = "major"
        key_sig_changes.append((tick, fifths, mode))

    return {
        "resolution": resolution,
        "onset_ticks": onset_ticks,
        "offset_ticks": offset_ticks,
        "tempo_changes": tempo_changes,
        "time_sig_changes": time_sig_changes,
        "key_sig_changes": key_sig_changes,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_midi(
    path: str, split_pitch: int = DEFAULT_SPLIT_PITCH
) -> List[NoteEvent]:
    """Parse MIDI → NoteEvent list (seconds-based, for model inference)."""
    midi_data = pretty_midi.PrettyMIDI(path)
    midi_channels = _read_non_drum_channels(path)
    return _parse_notes(midi_data, split_pitch, midi_channels)


def parse_midi_with_meta(
    path: str, split_pitch: int = DEFAULT_SPLIT_PITCH
) -> Tuple[List[NoteEvent], Dict[str, Any]]:
    """Parse MIDI → (events, rich_meta) for both inference and MusicXML export."""
    midi_data = pretty_midi.PrettyMIDI(path)
    midi_channels = _read_non_drum_channels(path)
    events = _parse_notes(midi_data, split_pitch, midi_channels)
    meta = _extract_midi_meta(midi_data, events)
    return events, meta
