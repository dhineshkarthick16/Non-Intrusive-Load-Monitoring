"""
collect_35min_dataset.py
------------------------
35-Minute Automated 4-Class Hardware Dataset Collection Protocol.

Protocol Schedule (7 cycles x 5 minutes = 35 minutes total):
  Cycle 1 (00:00 - 05:00): Expected -> 'NO LOAD' (Baseline)
  Cycle 2 (05:00 - 10:00): Expected -> 'Charger' (Charger ON)
  Cycle 3 (10:00 - 15:00): Expected -> 'NO LOAD' (Charger OFF & Removed)
  Cycle 4 (15:00 - 20:00): Expected -> 'Bulb' (Bulb ON)
  Cycle 5 (20:00 - 25:00): Expected -> 'NO LOAD' (Bulb OFF)
  Cycle 6 (25:00 - 30:00): Expected -> 'Both' (Charger ON + Bulb ON simultaneously)
  Cycle 7 (30:00 - 35:00): Expected -> 'NO LOAD' (Everything OFF & Removed)

Features:
  - 5-second transition alerts with audible countdown beeps.
  - Continuous streaming from COM3 @ 115200 baud.
  - Flushes raw (V1, V2, V3) and ground-truth labels directly to 'dataset_35min_4class.csv'.
  - Real-time terminal progress indicators and summary statistics per cycle.
"""

import argparse
import csv
import datetime
import os
import re
import sys
import time
import serial

try:
    import winsound

    def play_wav_or_beep(wav_name: str, fallback_freq: int = 1200, fallback_ms: int = 250):
        """Play loud Windows system sound file through laptop speakers."""
        wav_path = os.path.join(r"C:\Windows\Media", wav_name)
        played = False
        if os.path.isfile(wav_path):
            try:
                winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                played = True
            except Exception:
                pass
        if not played:
            try:
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                played = True
            except Exception:
                pass
        try:
            winsound.Beep(int(fallback_freq), int(fallback_ms))
        except Exception:
            sys.stdout.write("\a")
            sys.stdout.flush()

    def beep_countdown(sec: int):
        """Loud Windows 'ding' audio chime for each second of the transition countdown."""
        play_wav_or_beep("ding.wav", fallback_freq=1200, fallback_ms=250)

    def beep_transition_go():
        """Loud 'tada' chime confirming the new cycle has started."""
        play_wav_or_beep("tada.wav", fallback_freq=1800, fallback_ms=400)

    def beep_phase_complete():
        """Loud 'chimes' sound notifying that the 5-minute phase has completed."""
        play_wav_or_beep("chimes.wav", fallback_freq=1500, fallback_ms=350)

    def beep_test_all_done():
        """Final victory chime when the entire 35-minute test finishes."""
        play_wav_or_beep("tada.wav", fallback_freq=2000, fallback_ms=500)

except Exception:
    def fallback_beep():
        sys.stdout.write("\a")
        sys.stdout.flush()

    beep_countdown = lambda s: fallback_beep()
    beep_transition_go = lambda: fallback_beep()
    beep_phase_complete = lambda: fallback_beep()
    beep_test_all_done = lambda: fallback_beep()


PROTOCOL_CYCLES_35MIN = [
    {
        "cycle": 1,
        "name": "Cycle 1 (00:00 - 05:00)",
        "expected": "NO LOAD",
        "action": "BASELINE NO LOAD -> Ensure NO appliances are connected or turned ON.",
        "duration": 300,
    },
    {
        "cycle": 2,
        "name": "Cycle 2 (05:00 - 10:00)",
        "expected": "Charger",
        "action": "CHARGER ACTIVE -> Connect and switch ON the Phone Charger.",
        "duration": 300,
    },
    {
        "cycle": 3,
        "name": "Cycle 3 (10:00 - 15:00)",
        "expected": "NO LOAD",
        "action": "CHARGER REMOVED -> Turn OFF and disconnect the Phone Charger.",
        "duration": 300,
    },
    {
        "cycle": 4,
        "name": "Cycle 4 (15:00 - 20:00)",
        "expected": "Bulb",
        "action": "BULB ACTIVE -> Connect and switch ON the Bulb.",
        "duration": 300,
    },
    {
        "cycle": 5,
        "name": "Cycle 5 (20:00 - 25:00)",
        "expected": "NO LOAD",
        "action": "BULB OFF -> Turn OFF the Bulb (Observing post-bulb residual decay).",
        "duration": 300,
    },
    {
        "cycle": 6,
        "name": "Cycle 6 (25:00 - 30:00)",
        "expected": "Both",
        "action": "DUAL LOAD ACTIVE ('Both') -> Switch ON BOTH Phone Charger AND Bulb simultaneously!",
        "duration": 300,
    },
    {
        "cycle": 7,
        "name": "Cycle 7 (30:00 - 35:00)",
        "expected": "NO LOAD",
        "action": "FINAL CLEANUP NO LOAD -> Turn OFF and REMOVE all appliances completely.",
        "duration": 300,
    },
]


def countdown_transition_alert(cycle_info: dict, countdown_sec: int = 5):
    """Visual prompt banner and audible 5-second countdown at transition."""
    print("\n" + "=" * 80)
    print(f"  >>> TRANSITION ALERT: {cycle_info['name']} <<<")
    print(f"  TARGET CLASS : [ {cycle_info['expected']} ]")
    print(f"  ACTION REQ   : {cycle_info['action']}")
    print("=" * 80)

    for sec in range(countdown_sec, 0, -1):
        beep_countdown(sec)
        sys.stdout.write(f"\r  [*] Switch Hardware Now! Countdown: {sec}s... ")
        sys.stdout.flush()
        time.sleep(1.0)

    beep_transition_go()
    print("\r  [+] PHASE STARTED! Recording data for 300 seconds (5 minutes)...          \n")


def collect_dataset(port="COM3", baud=115200, output_csv="dataset_35min_4class.csv", cycle_duration=300):
    """
    Main collection loop running across the 7 protocol cycles.
    """
    abs_csv = os.path.abspath(output_csv)
    print("=" * 80)
    print("NILM 35-MINUTE 4-CLASS DATASET COLLECTION PROTOCOL")
    print("=" * 80)
    print(f"[*] Port:            {port} @ {baud} baud")
    print(f"[*] Output CSV:      {abs_csv}")
    print(f"[*] Cycle Duration:  {cycle_duration}s x 7 cycles = {cycle_duration*7/60:.1f} minutes")
    print(f"[*] Classes:         NO LOAD, Charger, Bulb, Both")
    print("=" * 80 + "\n")

    # Open serial connection
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
        time.sleep(2.0)  # Microcontroller reboot stabilization
        print(f"[+] Serial port {port} opened successfully.\n")
    except Exception as e:
        print(f"[!] Error opening serial port {port}: {e}")
        return False

    pattern = re.compile(r"([0-9]+\.?[0-9]*)\s*,\s*([0-9]+\.?[0-9]*)\s*,\s*([0-9]+\.?[0-9]*)")

    # Prepare CSV file
    csv_file = open(abs_csv, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["V1", "V2", "V3", "Label", "Cycle", "Timestamp", "ElapsedSec"])
    csv_file.flush()

    overall_start = time.time()
    total_samples_all = 0
    cycle_counts = {}

    print(f"{'TIME':<8} | {'V1':<6} {'V2':<6} {'V3':<6} | {'LABEL':<10} | {'CYCLE':<8} | {'PROGRESS'}")
    print("-" * 70)

    try:
        for cycle_info in PROTOCOL_CYCLES_35MIN:
            expected_label = cycle_info["expected"]
            cycle_idx = cycle_info["cycle"]
            duration = cycle_duration

            # Alert and 5-sec countdown beeps ONLY here at the transition!
            countdown_transition_alert(cycle_info, countdown_sec=5)

            cycle_start = time.time()
            samples_in_cycle = 0

            # Send expected state to ESP32 LCD
            ser.write(f"Collect:{expected_label}\n".encode("utf-8"))

            while (time.time() - cycle_start) < duration:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                match = pattern.search(line)
                if not match:
                    continue

                v1 = float(match.group(1))
                v2 = float(match.group(2))
                v3 = float(match.group(3))

                now_ts = time.time()
                elapsed_total = now_ts - overall_start
                cycle_elapsed = now_ts - cycle_start
                rem_sec = int(duration - cycle_elapsed)
                ts_str = datetime.datetime.now().strftime("%H:%M:%S")

                writer.writerow([v1, v2, v3, expected_label, cycle_idx, ts_str, round(elapsed_total, 2)])
                samples_in_cycle += 1
                total_samples_all += 1

                # Flush every 10 samples
                if samples_in_cycle % 10 == 0:
                    csv_file.flush()

                # Echo label to LCD
                ser.write(f"{expected_label}\n".encode("utf-8"))

                # Live progress line
                prog_str = f"{int(cycle_elapsed)}s/{duration}s ({rem_sec}s left)"
                print(f"{ts_str:<8} | {v1:<6.2f} {v2:<6.2f} {v3:<6.2f} | {expected_label:<10} | Cycle {cycle_idx} | {prog_str}")

            csv_file.flush()
            print(f"\n[+] Completed {cycle_info['name']} ({expected_label})! Playing completion beep...")
            beep_phase_complete()
            cycle_counts[cycle_idx] = {
                "name": cycle_info["name"],
                "label": expected_label,
                "samples": samples_in_cycle,
            }
            print("-" * 70)
            print(f"[+] Total samples in {cycle_info['name']}: {samples_in_cycle}")
            print("-" * 70 + "\n")

        # Test completely finished
        beep_test_all_done()

    except KeyboardInterrupt:
        print("\n[!] Collection interrupted by user.")
    finally:
        csv_file.close()
        ser.close()
        print(f"[*] Serial port closed. Total samples saved: {total_samples_all} -> {abs_csv}")

    # Summary table
    print("\n" + "=" * 65)
    print("35-MINUTE 4-CLASS COLLECTION SUMMARY")
    print("=" * 65)
    for c_id, info in cycle_counts.items():
        print(f"Cycle {c_id}: {info['label']:<10} | {info['samples']:<6} samples | {info['name']}")
    print("=" * 65)
    print(f"TOTAL SAMPLES COLLECTED: {total_samples_all}")
    print("=" * 65 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="35-Minute NILM 4-Class Dataset Collection Protocol")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port (default: COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--output", type=str, default="dataset_35min_4class.csv", help="Output CSV path")
    parser.add_argument("--duration", type=int, default=300, help="Duration per cycle in seconds (default: 300)")

    args = parser.parse_args()
    collect_dataset(port=args.port, baud=args.baud, output_csv=args.output, cycle_duration=args.duration)
