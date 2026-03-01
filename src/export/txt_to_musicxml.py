import math
import xml.etree.ElementTree as ET
from typing import List, Tuple


def convert_txt_to_musicxml(
    txt_content: str,
    bpm: float = 120.0,
    time_signature: str = "4/4",
    min_note: str = "1/16",
    divisions: int = 480,
) -> str:
    """
    Convert a PIG-format txt (onset/offset seconds + pitch + finger) to MusicXML.

    Args:
        txt_content: raw txt content.
        bpm: tempo in quarter-notes per minute.
        time_signature: e.g., "4/4" or "3/4".
        min_note: quantization grid, e.g., "1/16", "1/8".
        divisions: MusicXML divisions per quarter note.
    Returns:
        MusicXML string.
    """

    num, den = _parse_time_signature(time_signature)
    min_den = _parse_min_note(min_note)
    grid = _grid_from_min_note(divisions, min_den)
    sec_per_beat = 60.0 / bpm  # beat assumed as quarter note
    measure_div = int(num * divisions * 4 / den)

    events = _parse_txt(txt_content)
    # quantize onsets/durations to grid
    segments = []
    for ev in events:
        onset_beats = ev.onset / sec_per_beat
        offset_beats = ev.offset / sec_per_beat
        onset_div = _q(round(onset_beats * divisions), grid)
        dur_div = _q(round((offset_beats - onset_beats) * divisions), grid)
        dur_div = max(grid, dur_div)
        segments.extend(
                _split_across_measures(
                onset_div,
                dur_div,
                ev.pitch,
                    ev.midi,
                ev.finger,
                ev.channel,
                measure_div,
            )
        )

    measures = _group_by_measure(segments)
    xml_root = _build_xml(measures, divisions, num, den, bpm)
    return ET.tostring(xml_root, encoding="utf-8", xml_declaration=True).decode("utf-8")


# ---------------- helpers ----------------


class TxtEvent:
    __slots__ = ("idx", "onset", "offset", "pitch", "midi", "vel_on", "vel_off", "channel", "finger")

    def __init__(self, idx, onset, offset, pitch, midi, vel_on, vel_off, channel, finger):
        self.idx = idx
        self.onset = onset
        self.offset = offset
        self.pitch = pitch
        self.midi = midi
        self.vel_on = vel_on
        self.vel_off = vel_off
        self.channel = channel
        self.finger = finger


def _parse_txt(txt_content: str) -> List[TxtEvent]:
    events = []
    for line in txt_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            # allow missing finger/channel
            continue
        idx = int(parts[0])
        onset = float(parts[1])
        offset = float(parts[2])
        pitch = parts[3]
        vel_on = int(parts[4])
        vel_off = int(parts[5])
        channel = int(parts[6])
        finger_txt = parts[7]
        if finger_txt:
            # 兼容形如 "4_1" 或 "-3_-1" 的多指标记，取第一个数值
            first_token = finger_txt.split("_")[0]
            try:
                finger = int(first_token)
            except ValueError:
                finger = 0
        else:
            finger = 0
        midi = _pitch_to_midi(pitch)
        events.append(TxtEvent(idx, onset, offset, pitch, midi, vel_on, vel_off, channel, finger))
    return events


def _parse_time_signature(ts: str) -> Tuple[int, int]:
    if "/" in ts:
        a, b = ts.split("/", 1)
        return int(a), int(b)
    return 4, 4


def _parse_min_note(mn: str) -> int:
    if "/" in mn:
        _, b = mn.split("/", 1)
        return int(b)
    return 16


def _grid_from_min_note(divisions: int, min_den: int) -> int:
    return int(divisions * 4 / min_den)


def _q(val: int, grid: int) -> int:
    return int(round(val / grid) * grid)


def _split_across_measures(
    onset_div: int,
    dur_div: int,
    pitch: str,
    midi: int,
    finger: int,
    channel: int,
    measure_div: int,
):
    parts = []
    remaining = dur_div
    start = onset_div
    first = True
    while remaining > 0:
        measure_idx = start // measure_div
        pos_in_measure = start % measure_div
        room = measure_div - pos_in_measure
        take = min(remaining, room)
        parts.append(
            {
                "measure": measure_idx,
                "onset": pos_in_measure,
                "dur": take,
                "pitch": pitch,
                "midi": midi,
                "finger": finger,
                "channel": channel,
                "tie_start": not first,
                "tie_stop": remaining > room,
            }
        )
        remaining -= take
        start += take
        first = False
    return parts


def _group_by_measure(segments: List[dict]):
    measures = {}
    for seg in segments:
        m = seg["measure"]
        measures.setdefault(m, []).append(seg)
    return measures


def _pitch_to_step_alter_oct(pitch: str):
    # simple parser: e.g., C#4, Db4, F4
    if len(pitch) < 2:
        return "C", 0, 4
    step = pitch[0].upper()
    alter = 0
    rest = pitch[1:]
    if rest[0] in ("#", "b"):
        alter = 1 if rest[0] == "#" else -1
        rest = rest[1:]
    octave = int(rest) if rest else 4
    return step, alter, octave


def _note_type_from_duration(dur: int, divisions: int):
    # approximate to nearest standard type
    ratios = {
        "whole": 4,
        "half": 2,
        "quarter": 1,
        "eighth": 0.5,
        "16th": 0.25,
        "32nd": 0.125,
    }
    beat_ratio = dur / divisions
    best = min(ratios.items(), key=lambda kv: abs(kv[1] - beat_ratio))
    return best[0]


def _pitch_to_midi(pitch: str) -> int:
    step_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    step = pitch[0].upper() if pitch else "C"
    alter = 0
    rest = pitch[1:] if len(pitch) > 1 else ""
    if rest.startswith("#"):
        alter = 1
        rest = rest[1:]
    elif rest.startswith("b"):
        alter = -1
        rest = rest[1:]
    octave = int(rest) if rest.isdigit() else 4
    return (octave + 1) * 12 + step_map.get(step, 0) + alter


def _choose_staff(finger: int, midi: int | None, channel: int) -> str:
    # finger sign overrides: >0 -> right (staff 1), <0 -> left (staff 2)
    if finger > 0:
        return "1"
    if finger < 0:
        return "2"
    # fallback to pitch split
    if midi is not None:
        return "1" if midi >= 60 else "2"
    # fallback to channel
    return "1" if channel == 0 else "2"


def _build_xml(measures, divisions: int, num: int, den: int, bpm: float):
    root = ET.Element("score-partwise", version="3.1")
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Piano"
    part = ET.SubElement(root, "part", id="P1")

    max_measure = max(measures.keys()) if measures else 0
    for m in range(max_measure + 1):
        measure_el = ET.SubElement(part, "measure", number=str(m + 1))
        if m == 0:
            attrs = ET.SubElement(measure_el, "attributes")
            ET.SubElement(attrs, "divisions").text = str(divisions)
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(num)
            ET.SubElement(time_el, "beat-type").text = str(den)
            key_el = ET.SubElement(attrs, "key")
            ET.SubElement(key_el, "fifths").text = "0"
            ET.SubElement(attrs, "staves").text = "2"
            # clefs
            clef1 = ET.SubElement(attrs, "clef", number="1")
            ET.SubElement(clef1, "sign").text = "G"
            ET.SubElement(clef1, "line").text = "2"
            clef2 = ET.SubElement(attrs, "clef", number="2")
            ET.SubElement(clef2, "sign").text = "F"
            ET.SubElement(clef2, "line").text = "4"
            sound = ET.SubElement(measure_el, "sound", tempo=str(bpm))

        segs = measures.get(m, [])
        if not segs:
            rest_note = _rest_note(divisions * num * 4 // den, divisions, voice="1")
            measure_el.append(rest_note)
            continue

        # group by onset within measure
        segs.sort(key=lambda s: (s["onset"], s["pitch"]))
        groups = []
        cur_onset = None
        cur = []
        for seg in segs:
            if cur_onset is None or seg["onset"] != cur_onset:
                if cur:
                    groups.append((cur_onset, cur))
                cur_onset = seg["onset"]
                cur = [seg]
            else:
                cur.append(seg)
        if cur:
            groups.append((cur_onset, cur))

        pointer = 0
        measure_div = int(num * divisions * 4 / den)
        for onset, notes in groups:
            if onset > pointer:
                gap = onset - pointer
                # 默认休止填在高音谱，保证节奏对齐；如需更严谨可按 staff 分开填
                measure_el.append(_rest_note(gap, divisions, staff="1"))
                pointer += gap

            # create notes (chord if multiple)
            notes.sort(key=lambda s: s["pitch"])
            max_dur = 0
            for idx, seg in enumerate(notes):
                staff_num = _choose_staff(seg["finger"], seg.get("midi"), seg["channel"])
                note_el = _note_element(
                    seg["pitch"],
                    seg["dur"],
                    seg["finger"],
                    seg["tie_start"],
                    seg["tie_stop"],
                    seg["channel"],
                    seg.get("midi"),
                    divisions,
                    staff_num,
                    chord=(idx > 0),
                )
                measure_el.append(note_el)
                max_dur = max(max_dur, seg["dur"])
            pointer = onset + max_dur

        if pointer < measure_div:
            measure_el.append(_rest_note(measure_div - pointer, divisions, staff="1"))

    return root


def _rest_note(duration: int, divisions: int, staff: str = "1"):
    note = ET.Element("note")
    ET.SubElement(note, "rest")
    ET.SubElement(note, "duration").text = str(duration)
    ET.SubElement(note, "voice").text = "1" if staff == "1" else "2"
    ET.SubElement(note, "staff").text = staff
    ET.SubElement(note, "type").text = _note_type_from_duration(duration, divisions)
    return note


def _note_element(
    pitch: str,
    duration: int,
    finger: int,
    tie_start: bool,
    tie_stop: bool,
    channel: int,
    midi: int | None,
    divisions: int,
    staff: str,
    chord: bool = False,
):
    note = ET.Element("note")
    if chord:
        ET.SubElement(note, "chord")
    pitch_el = ET.SubElement(note, "pitch")
    step, alter, octave = _pitch_to_step_alter_oct(pitch)
    ET.SubElement(pitch_el, "step").text = step
    if alter != 0:
        ET.SubElement(pitch_el, "alter").text = str(alter)
    ET.SubElement(pitch_el, "octave").text = str(octave)
    ET.SubElement(note, "duration").text = str(duration)
    ET.SubElement(note, "voice").text = "1" if staff == "1" else "2"
    ET.SubElement(note, "type").text = _note_type_from_duration(duration, divisions)
    ET.SubElement(note, "staff").text = staff

    if tie_start:
        ET.SubElement(note, "tie", type="start")
    if tie_stop:
        ET.SubElement(note, "tie", type="stop")
    if tie_start or tie_stop:
        notations = ET.SubElement(note, "notations")
        if tie_start:
            ET.SubElement(notations, "tied", type="start")
        if tie_stop:
            ET.SubElement(notations, "tied", type="stop")
    if finger != 0:
        notations = note.find("notations") or ET.SubElement(note, "notations")
        technical = ET.SubElement(notations, "technical")
        ET.SubElement(technical, "fingering").text = str(finger)
    return note

