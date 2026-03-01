import pretty_midi
midi = pretty_midi.PrettyMIDI('/Users/ztz/Downloads/chopin-nocturne-op-9-no-2-e-flat-major.mid')

# 速度/拍号/调号
tempo_times, tempi = midi.get_tempo_changes()
time_sigs = midi.time_signature_changes          # 列表
key_sigs = midi.key_signature_changes            # 列表

# 歌词/文本事件
lyrics = midi.lyrics                              # 列表
text_events = midi.text_events                    # 列表
print(tempo_times, tempi)
print(time_sigs)
print(key_sigs)
print(lyrics)
print(text_events)
# 轨道/乐器信息
for inst in midi.instruments:
    name = inst.name
    program = inst.program
    is_drum = inst.is_drum
    print(name, program, is_drum)