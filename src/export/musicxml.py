import math
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

from src.data.parser import NoteEvent
from src.data.features import IDX_TO_FINGER

DIVISIONS = 480
DEFAULT_TIME = (4, 4)  # beats, beat-type


def pitch_components(pitch: str) -> Tuple[str, int, int]:
    step = pitch[0]
    rest = pitch[1:]
    alter = 0
    octave = 4
    if rest:
        if "#" in rest:
            alter += rest.count("#")
        if "b" in rest:
            alter -= rest.count("b")
        digits = "".join([c for c in rest if c.isdigit()])
        if digits:
            octave = int(digits)
    return step, alter, octave


def note_type_and_dots(ticks: int) -> Tuple[str, int]:
    base_map = {
        DIVISIONS * 4: "whole",
        DIVISIONS * 2: "half",
        DIVISIONS: "quarter",
        DIVISIONS // 2: "eighth",
        DIVISIONS // 4: "16th",
    }
    if ticks in base_map:
        return base_map[ticks], 0
    # dotted quarter/half
    if ticks == int(DIVISIONS * 1.5):
        return "quarter", 1
    if ticks == int(DIVISIONS * 3):
        return "half", 1
    return "quarter", 0


def add_note(parent, pitch: str, duration: int, finger: int, is_rest: bool = False):
    note = ET.SubElement(parent, "note")
    if is_rest:
        ET.SubElement(note, "rest")
    else:
        step, alter, octave = pitch_components(pitch)
        pitch_el = ET.SubElement(note, "pitch")
        ET.SubElement(pitch_el, "step").text = step
        if alter != 0:
            ET.SubElement(pitch_el, "alter").text = str(alter)
        ET.SubElement(pitch_el, "octave").text = str(octave)
    ET.SubElement(note, "duration").text = str(max(1, duration))
    ET.SubElement(note, "voice").text = "1"
    ntype, dots = note_type_and_dots(duration)
    ET.SubElement(note, "type").text = ntype
    for _ in range(dots):
        ET.SubElement(note, "dot")
    techn = ET.SubElement(note, "notations")
    tech = ET.SubElement(techn, "technical")
    ET.SubElement(tech, "fingering").text = str(finger)
    return note


def quantize_ticks(ticks: float) -> int:
    # snap to nearest 1/16 grid
    grid = DIVISIONS // 4
    q = max(grid, int(round(ticks / grid) * grid))
    return q


def chunk_event(duration_ticks: int, remaining_in_measure: int) -> Tuple[int, int]:
    """Return (use_now, left_over)."""
    use = min(duration_ticks, remaining_in_measure)
    left = duration_ticks - use
    return use, left


def make_attributes(parent, time_sig=DEFAULT_TIME):
    attributes = ET.SubElement(parent, "attributes")
    ET.SubElement(attributes, "divisions").text = str(DIVISIONS)
    time_el = ET.SubElement(attributes, "time")
    ET.SubElement(time_el, "beats").text = str(time_sig[0])
    ET.SubElement(time_el, "beat-type").text = str(time_sig[1])
    clef = ET.SubElement(attributes, "clef")
    ET.SubElement(clef, "sign").text = "G"
    ET.SubElement(clef, "line").text = "2"


def add_measures_for_hand(part_el, events: List[Tuple[NoteEvent, int]], sec_per_quarter: float):
    measure_idx = 1
    measure = ET.SubElement(part_el, "measure", number=str(measure_idx))
    make_attributes(measure)
    ticks_per_measure = DEFAULT_TIME[0] * DIVISIONS
    cursor_ticks = 0

    def ensure_measure():
        nonlocal measure_idx, measure, cursor_ticks
        if cursor_ticks >= ticks_per_measure - 1e-6:
            cursor_ticks = 0
            measure_idx += 1
            measure = ET.SubElement(part_el, "measure", number=str(measure_idx))

    for ev, f_idx in events:
        onset_ticks = int(round(ev.onset / sec_per_quarter * DIVISIONS))
        if onset_ticks < cursor_ticks:
            onset_ticks = cursor_ticks
        if onset_ticks > cursor_ticks:
            gap = onset_ticks - cursor_ticks
            while gap > 0:
                ensure_measure()
                rem = ticks_per_measure - cursor_ticks
                use, gap = chunk_event(gap, rem)
                add_note(measure, "C4", use, finger=0, is_rest=True)
                cursor_ticks += use
        duration_ticks = int(round((ev.offset - ev.onset) / sec_per_quarter * DIVISIONS))
        duration_ticks = quantize_ticks(duration_ticks)
        if duration_ticks <= 0:
            duration_ticks = quantize_ticks(DIVISIONS // 4)
        remaining = duration_ticks
        while remaining > 0:
            ensure_measure()
            rem = ticks_per_measure - cursor_ticks
            use, remaining = chunk_event(remaining, rem)
            add_note(measure, ev.pitch_str, use, finger=IDX_TO_FINGER.get(int(f_idx), 0), is_rest=False)
            cursor_ticks += use
    # fill final measure end with rest if needed
    if cursor_ticks > 0 and cursor_ticks < ticks_per_measure:
        add_note(measure, "C4", ticks_per_measure - cursor_ticks, finger=0, is_rest=True)


def split_by_hand(events: List[NoteEvent], fingers: List[int]) -> Dict[int, List[Tuple[NoteEvent, int]]]:
    hands = {0: [], 1: []}
    for ev, f in zip(events, fingers):
        if ev.channel == 0:
            hands[0].append((ev, f))
        elif ev.channel == 1:
            hands[1].append((ev, f))
    for k in hands:
        hands[k].sort(key=lambda x: x[0].onset)
    return hands


def predictions_to_musicxml(
    events: List[NoteEvent],
    finger_indices: List[int],
    out_path: str,
    title: str = "Generated Fingering",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    durations = [ev.offset - ev.onset for ev in events if ev.offset > ev.onset]
    median_dur = sorted(durations)[len(durations) // 2] if durations else 0.5
    sec_per_quarter = median_dur if median_dur > 0 else 0.5

    hands = split_by_hand(events, finger_indices)

    score = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title
    part_list = ET.SubElement(score, "part-list")
    rh_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(rh_part, "part-name").text = "Right Hand"
    lh_part = ET.SubElement(part_list, "score-part", id="P2")
    ET.SubElement(lh_part, "part-name").text = "Left Hand"

    part_r = ET.SubElement(score, "part", id="P1")
    part_l = ET.SubElement(score, "part", id="P2")

    add_measures_for_hand(part_r, hands[0], sec_per_quarter)
    add_measures_for_hand(part_l, hands[1], sec_per_quarter)

    tree = ET.ElementTree(score)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


