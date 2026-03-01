import argparse

def parse_pig_line(line):
    parts = line.strip().split()
    if len(parts) < 8:
        parts = line.strip().split('\t')
    if len(parts) < 8:
        return None
    # idx, onset, offset, pitch, vel_on, vel_off, channel, finger
    return {
        'idx': int(parts[0]),
        'onset': float(parts[1]),
        'offset': float(parts[2]),
        'pitch': parts[3],
        'vel_on': int(float(parts[4])),
        'vel_off': int(float(parts[5])),
        'channel': int(parts[6]),
        # finger might be mixed
        'finger': parts[7]
    }

def compare_pig(file1, file2, tolerance=0.1):
    with open(file1) as f1, open(file2) as f2:
        lines1 = [l for l in f1 if not l.startswith('//') and l.strip()]
        lines2 = [l for l in f2 if not l.startswith('//') and l.strip()]
        
    print(f"File 1: {len(lines1)} lines")
    print(f"File 2: {len(lines2)} lines")
    
    # We allow slight length mismatch (grace notes?) but mostly should match
    min_len = min(len(lines1), len(lines2))
    
    mismatches = 0
    for i in range(min_len):
        ev1 = parse_pig_line(lines1[i])
        ev2 = parse_pig_line(lines2[i])
        
        if not ev1 or not ev2:
            continue
            
        # Compare onset (crucial)
        if abs(ev1['onset'] - ev2['onset']) > tolerance:
            print(f"Mismatch line {i}: Onset {ev1['onset']} != {ev2['onset']}")
            mismatches += 1
            if mismatches > 10: break
            continue
            
        # Compare pitch
        if ev1['pitch'] != ev2['pitch']:
            print(f"Mismatch line {i}: Pitch {ev1['pitch']} != {ev2['pitch']}")
            mismatches += 1
            continue

        # Compare channel
        if ev1['channel'] != ev2['channel']:
            print(f"Mismatch line {i}: Channel {ev1['channel']} != {ev2['channel']}")
            mismatches += 1
            continue
            
    if mismatches == 0:
        print("SUCCESS: Files match within tolerance.")
    else:
        print(f"FAILURE: Found {mismatches} mismatches.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file1")
    parser.add_argument("file2")
    args = parser.parse_args()
    compare_pig(args.file1, args.file2)
