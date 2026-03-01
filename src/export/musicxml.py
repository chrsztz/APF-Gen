"""Generate MusicXML with fingering annotations using **music21**.

Delegates all music-theory logic to music21:
  - Enharmonic pitch spelling based on key signature
  - Automatic beaming of 8th/16th/32nd notes
  - Anacrusis (pickup measure) handling
  - Proper chord representation
  - Bass clef for LH, treble clef for RH
  - Clean MusicXML output
"""

import os
from collections import defaultdict
from fractions import Fraction
from typing import Any, Dict, List, Optional

from music21 import (
    articulations,
    chord,
    clef,
    duration,
    key,
    metadata,
    meter,
    note,
    pitch,
    stream,
    tempo,
)

from src.data.parser import NoteEvent
from src.data.features import IDX_TO_FINGER

# Onset/offset quantization grid: 1/8 quarter note = 32nd note.
# Using 1/8 (instead of finer grids like 1/24) guarantees that every
# gap between notes is a multiple of 1/8, which is always a standard
# musical duration that music21 can export.
_GRID = 0.125  # 32nd note

# Minimum duration
_MIN_DUR = 0.125  # 32nd note

# Exhaustive list of quarter-note lengths that music21 can cleanly export
# to MusicXML.  Used by _sanitize_durations as a safety net.
_SAFE_QL: List[Fraction] = sorted([
    Fraction(1, 8),    # 32nd
    Fraction(1, 6),    # triplet 16th
    Fraction(3, 16),   # dotted 32nd
    Fraction(1, 4),    # 16th
    Fraction(1, 3),    # triplet 8th
    Fraction(3, 8),    # dotted 16th
    Fraction(1, 2),    # 8th
    Fraction(2, 3),    # triplet quarter
    Fraction(3, 4),    # dotted 8th
    Fraction(1, 1),    # quarter
    Fraction(4, 3),    # triplet half
    Fraction(3, 2),    # dotted quarter
    Fraction(2, 1),    # half
    Fraction(3, 1),    # dotted half
    Fraction(4, 1),    # whole
    Fraction(6, 1),    # dotted whole
    Fraction(8, 1),    # breve
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap(qn: float) -> float:
    """Snap any quarter-note value to the 32nd-note grid."""
    return round(qn / _GRID) * _GRID


def _nearest_safe(ql: float) -> Fraction:
    """Return the nearest safe Fraction duration for a given quarter length."""
    if ql <= 0:
        return _SAFE_QL[0]
    target = Fraction(ql).limit_denominator(100)
    return min(_SAFE_QL, key=lambda x: abs(x - target))


def _sanitize_durations(part_stream):
    """Force every note/rest to a duration music21 can export to MusicXML.

    makeMeasures() can generate rests with non-standard durations (e.g. 5/24
    of a quarter note) that crash the MusicXML serialiser.  This snaps them
    to the nearest safe standard value using exact Fraction arithmetic.
    """
    for el in part_stream.recurse().notesAndRests:
        ql = el.duration.quarterLength
        safe = _nearest_safe(ql)
        if abs(float(safe) - ql) > 0.001:
            try:
                el.duration = duration.Duration(quarterLength=safe)
            except Exception:
                el.duration = duration.Duration(quarterLength=Fraction(1, 4))


def _respell(midi_num: int, key_fifths: int) -> pitch.Pitch:
    """Return a *music21* Pitch with enharmonic spelling matching the key.

    * Flat keys (fifths < 0) → prefer flat spellings  (A# → Bb)
    * Sharp keys (fifths > 0) → prefer sharp spellings (Eb → D#)
    * Neutral (fifths == 0) → use music21 default
    """
    p = pitch.Pitch(midi=midi_num)
    if p.accidental is not None:
        if key_fifths < 0 and p.accidental.alter > 0:
            enh = p.getEnharmonic()
            if enh.accidental is None or enh.accidental.alter <= 0:
                return enh
        elif key_fifths > 0 and p.accidental.alter < 0:
            enh = p.getEnharmonic()
            if enh.accidental is None or enh.accidental.alter >= 0:
                return enh
    return p


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predictions_to_musicxml(
    events: List[NoteEvent],
    finger_indices: List[int],
    out_path: str,
    title: str = "Generated Fingering",
    midi_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a MusicXML score with fingering, beaming, and proper spelling.

    Parameters
    ----------
    events : list[NoteEvent]
    finger_indices : list[int]   – model-output class indices
    out_path : str               – destination .musicxml path
    title : str
    midi_meta : dict | None
        Rich metadata from ``parse_midi_with_meta``.
        When *None*, falls back to rough estimation.
    """
    if not midi_meta:
        midi_meta = {}

    resolution = midi_meta.get("resolution", 480)
    onset_ticks = midi_meta.get("onset_ticks", [])
    offset_ticks = midi_meta.get("offset_ticks", [])
    ts_changes = midi_meta.get("time_sig_changes", [(0, 4, 4)])
    ks_changes = midi_meta.get("key_sig_changes", [])
    tempo_changes = midi_meta.get("tempo_changes", [(0, 120)])

    # tick → quarter-note offset
    def t2q(tick: float) -> float:
        return tick / resolution

    # ------------------------------------------------------------------
    # Key-signature lookup (for pitch spelling throughout the piece)
    # ------------------------------------------------------------------
    ks_points = [(t2q(tick), fifths) for tick, fifths, _ in ks_changes]
    if not ks_points:
        ks_points = [(0.0, 0)]

    def _key_at(oq: float) -> int:
        """Return active key-fifths at quarter-note offset *oq*."""
        result = ks_points[0][1]
        for qn, fifths in ks_points:
            if qn <= oq + 0.001:
                result = fifths
            else:
                break
        return result

    # ------------------------------------------------------------------
    # Anacrusis detection
    # Many MIDI files encode pickups as a short initial time-signature
    # (e.g. 1/8) followed by the real one (12/8).
    # ------------------------------------------------------------------
    anacrusis = False
    anacrusis_qn = 0.0
    main_ts_num, main_ts_den = ts_changes[0][1], ts_changes[0][2]

    if len(ts_changes) >= 2:
        first_meas_qn = ts_changes[0][1] * 4.0 / ts_changes[0][2]
        second_meas_qn = ts_changes[1][1] * 4.0 / ts_changes[1][2]
        if first_meas_qn < second_meas_qn:
            anacrusis = True
            anacrusis_qn = first_meas_qn
            main_ts_num = ts_changes[1][1]
            main_ts_den = ts_changes[1][2]

    # ------------------------------------------------------------------
    # Build note data  (onset_qn, dur_qn, midi, channel, finger)
    # ------------------------------------------------------------------
    note_data: List[tuple] = []
    for i, (ev, f_idx) in enumerate(zip(events, finger_indices)):
        if onset_ticks:
            ot, ft = onset_ticks[i], offset_ticks[i]
        else:
            # Fallback: seconds → ticks estimation
            bpm = tempo_changes[0][1] if tempo_changes else 120
            spq = 60.0 / bpm if bpm > 0 else 0.5
            ot = ev.onset / spq * resolution
            ft = ev.offset / spq * resolution

        oq = max(0, _snap(t2q(ot)))
        end_q = _snap(t2q(ft))
        dq = max(_MIN_DUR, end_q - oq)
        finger = IDX_TO_FINGER.get(int(f_idx), 0)
        note_data.append((oq, dq, ev.midi, ev.channel, finger))

    rh_notes = [(o, d, m, f) for o, d, m, ch, f in note_data if ch == 0]
    lh_notes = [(o, d, m, f) for o, d, m, ch, f in note_data if ch == 1]

    # ------------------------------------------------------------------
    # Build a music21 Part for one hand
    # ------------------------------------------------------------------
    def _build_part(
        notes_list: list,
        clef_obj,
        part_name: str,
        add_tempo: bool,
    ) -> stream.Part:
        part = stream.Part(id=part_name.replace(" ", ""))
        part.partName = part_name
        part.insert(0, clef_obj)

        # ---- Time signature(s) ------------------------------------------
        if anacrusis:
            # Show the main TS from the start; the pickup duration is
            # encoded via paddingLeft on the first Measure (see below).
            part.insert(0, meter.TimeSignature(f"{main_ts_num}/{main_ts_den}"))
            for tick, num, den in ts_changes[2:]:
                part.insert(t2q(tick), meter.TimeSignature(f"{num}/{den}"))
        else:
            for tick, num, den in ts_changes:
                part.insert(t2q(tick), meter.TimeSignature(f"{num}/{den}"))

        # ---- Key signature(s) -------------------------------------------
        for tick, fifths, mode_str in ks_changes:
            ks = key.KeySignature(fifths)
            part.insert(t2q(tick), ks)

        # ---- Tempo marks (first part only) ------------------------------
        if add_tempo:
            for tick, bpm in tempo_changes:
                part.insert(t2q(tick), tempo.MetronomeMark(number=round(bpm)))

        # ---- Notes / Chords --------------------------------------------
        # Group by quantized onset for chord detection
        onset_groups: Dict[float, list] = defaultdict(list)
        for oq, dq, midi_num, finger in notes_list:
            onset_groups[oq].append((dq, midi_num, finger))

        for oq in sorted(onset_groups.keys()):
            group = onset_groups[oq]
            kf = _key_at(oq)

            if len(group) == 1:
                dq, midi_num, f = group[0]
                p = _respell(midi_num, kf)
                n = note.Note(p, duration=duration.Duration(dq))
                if f != 0:
                    n.articulations.append(articulations.Fingering(abs(f)))
                part.insert(oq, n)
            else:
                # Build a Chord
                pitches: List[pitch.Pitch] = []
                max_dur = 0.0
                fingers: List[int] = []
                for dq, midi_num, f in sorted(group, key=lambda x: x[1]):
                    pitches.append(_respell(midi_num, kf))
                    max_dur = max(max_dur, dq)
                    fingers.append(f)
                c = chord.Chord(pitches, duration=duration.Duration(max_dur))
                for f in fingers:
                    if f != 0:
                        c.articulations.append(articulations.Fingering(abs(f)))
                part.insert(oq, c)

        # ---- Post-processing -------------------------------------------
        part.makeMeasures(inPlace=True)

        # Safety net: clamp any bizarre rest/note durations that
        # makeMeasures may have created due to floating-point drift.
        _sanitize_durations(part)

        # Handle anacrusis: tell music21 the first measure is a pickup
        if anacrusis:
            measures = list(part.getElementsByClass(stream.Measure))
            if measures:
                main_meas_qn = main_ts_num * 4.0 / main_ts_den
                measures[0].paddingLeft = main_meas_qn - anacrusis_qn

        # Automatic beaming per measure
        for m in part.getElementsByClass(stream.Measure):
            try:
                m.makeBeams(inPlace=True)
            except Exception:
                pass  # rare edge cases

        # Let music21 decide courtesy accidentals
        part.makeAccidentals(inPlace=True)

        return part

    # ------------------------------------------------------------------
    # Assemble Score
    # ------------------------------------------------------------------
    rh_part = _build_part(rh_notes, clef.TrebleClef(), "Right Hand", add_tempo=True)
    lh_part = _build_part(lh_notes, clef.BassClef(), "Left Hand", add_tempo=False)

    s = stream.Score()
    md = metadata.Metadata()
    md.title = title
    s.metadata = md
    s.insert(0, rh_part)
    s.insert(0, lh_part)

    # Write
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    s.write("musicxml", fp=out_path)
    return out_path
