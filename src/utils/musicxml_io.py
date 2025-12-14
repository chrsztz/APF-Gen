import xml.etree.ElementTree as ET
from typing import List

from src.data.parser import NoteEvent, pitch_to_midi

DIV_FALLBACK = 480
GRID = DIV_FALLBACK // 4  # 1/16 grid


def quantize(val: float, grid: int) -> int:
    return int(round(val / grid) * grid)


def _get_tempo(root: ET.Element, default: float = 120.0) -> float:
    for per_min in root.iter("per-minute"):
        try:
            return float(per_min.text)
        except (TypeError, ValueError):
            continue
    return default


def _get_divisions(measure: ET.Element, current: int) -> int:
    div_el = measure.find("./attributes/divisions")
    if div_el is not None and div_el.text:
        try:
            return int(div_el.text)
        except ValueError:
            pass
    return current


def _midi_from_pitch(pitch_el: ET.Element):
    step = pitch_el.findtext("step", default="C")
    alter_text = pitch_el.findtext("alter")
    alter = int(alter_text) if alter_text else 0
    octave = int(pitch_el.findtext("octave", default="4"))
    accidental = "#" if alter > 0 else "b" if alter < 0 else ""
    pitch_str = f"{step}{accidental}{octave}"
    return pitch_str, pitch_to_midi(pitch_str)


def parse_musicxml(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    tempo_bpm = _get_tempo(root)
    events: List[NoteEvent] = []

    parts = root.findall("part")
    if not parts:
        return events, tree

    for p_idx, part in enumerate(parts):
        channel_default = 0 if p_idx == 0 else 1
        time_sec = 0.0
        divisions = DIV_FALLBACK
        for measure in part.findall("measure"):
            divisions = _get_divisions(measure, divisions)
            sec_per_div = 60.0 / tempo_bpm / divisions
            for note in measure.findall("note"):
                if note.find("rest") is not None:
                    dur = int(note.findtext("duration", default="0"))
                    time_sec += dur * sec_per_div
                    continue
                pitch_el = note.find("pitch")
                if pitch_el is None:
                    continue
                duration = int(note.findtext("duration", default="0"))
                is_chord = note.find("chord") is not None
                staff_text = note.findtext("staff")
                channel = channel_default
                if staff_text:
                    if staff_text.strip() == "1":
                        channel = 0
                    elif staff_text.strip() == "2":
                        channel = 1
                onset_div = quantize(time_sec / sec_per_div, GRID)
                onset_sec = onset_div * sec_per_div
                dur_q = max(GRID, quantize(duration, GRID))
                offset_sec = onset_sec + dur_q * sec_per_div
                pitch_str, midi = _midi_from_pitch(pitch_el)
                ev = NoteEvent(
                    idx=len(events),
                    onset=onset_sec,
                    offset=offset_sec,
                    pitch_str=pitch_str,
                    midi=midi,
                    vel_on=64,
                    vel_off=64,
                    channel=channel,
                    finger=0,
                )
                ev.note_ref = note  # keep reference for writing back (same tree)
                events.append(ev)
                if not is_chord:
                    time_sec = offset_sec
            # handle backup/forward
            for bk in measure.findall("backup"):
                dur = int(bk.findtext("duration", default="0"))
                time_sec -= dur * sec_per_div
            for fw in measure.findall("forward"):
                dur = int(fw.findtext("duration", default="0"))
                time_sec += dur * sec_per_div

    events.sort(key=lambda e: (e.onset, e.midi))
    for i, ev in enumerate(events):
        ev.idx = i
    return events, tree


def write_fingerings_to_musicxml(tree: ET.ElementTree, fingerings: List[int], events: List[NoteEvent], out_path: str):
    for ev, fing in zip(events, fingerings):
        note_el = getattr(ev, "note_ref", None)
        if note_el is None:
            continue
        notations = note_el.find("notations")
        if notations is None:
            notations = ET.SubElement(note_el, "notations")
        technical = notations.find("technical")
        if technical is None:
            technical = ET.SubElement(notations, "technical")
        fing_el = technical.find("fingering")
        if fing_el is None:
            fing_el = ET.SubElement(technical, "fingering")
        fing_el.text = str(fing)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

