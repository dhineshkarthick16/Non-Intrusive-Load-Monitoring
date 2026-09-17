"""
automated_validation_suite.py
------------------------------
40-Minute Automated Hardware Validation Test Suite for Non-Intrusive Load Monitoring (NILM).

Protocol Schedule (8 sequential 5-minute cycles = 40 minutes total):
  Cycle 1 (00:00 - 05:00): Expected -> 'NO LOAD' (Baseline)
  Cycle 2 (05:00 - 10:00): Expected -> 'Charger' (ON)
  Cycle 3 (10:00 - 15:00): Expected -> 'NO LOAD' (Charger off & removed)
  Cycle 4 (15:00 - 20:00): Expected -> 'Bulb' (ON)
  Cycle 5 (20:00 - 25:00): Expected -> 'NO LOAD' (Bulb off)
  Cycle 6 (25:00 - 30:00): Expected -> 'NO LOAD' (Charger plugged in, but OFF)
  Cycle 7 (30:00 - 35:00): Expected -> 'Charger' (ON)
  Cycle 8 (35:00 - 40:00): Expected -> 'NO LOAD' (Everything removed)

Features:
  - Hardware Transition Alerts with 5-second countdowns & audio beeps.
  - Serial streaming on COM3 @ 115200 baud with feedback to ESP32 LCD.
  - Real-time evaluation through load_classifier.pkl and NILMStateFilter.
  - Transition latency, false positive rate, accuracy, and signal curve tracking.
  - Full output persisted to validation_report_40min.json and validation_curve_40min.png.
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure feature extractor & state filter are available
from feature_extractor import NILMFeatureExtractor, FEATURE_NAMES
from live_inference import NILMStateFilter, load_classifier

try:
    import winsound
    def play_alert(freq=1200, duration=250):
        winsound.Beep(freq, duration)
except Exception:
    def play_alert(freq=1200, duration=250):
        sys.stdout.write("\a")
        sys.stdout.flush()

try:
    import serial
except ImportError:
    serial = None

PROTOCOL_CYCLES = [
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
        "action": "CHARGER ACTIVE -> Plug in and switch ON the Phone Charger.",
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
        "expected": "NO LOAD",
        "action": "CHARGER PLUGGED BUT OFF -> Plug in Charger but KEEP IT SWITCHED OFF.",
        "duration": 300,
    },
    {
        "cycle": 7,
        "name": "Cycle 7 (30:00 - 35:00)",
        "expected": "Charger",
        "action": "CHARGER ACTIVE -> Switch ON the Phone Charger again.",
        "duration": 300,
    },
    {
        "cycle": 8,
        "name": "Cycle 8 (35:00 - 40:00)",
        "expected": "NO LOAD",
        "action": "CLEANUP NO LOAD -> Turn OFF and REMOVE all appliances completely.",
        "duration": 300,
    },
]


def wait_until_start_time(target_time_str: str):
    """Wait until a specific clock time HH:MM or HH:MM:SS if requested."""
    if not target_time_str:
        return
    now = datetime.datetime.now()
    parts = [int(p) for p in target_time_str.split(":")]
    if len(parts) == 2:
        target = now.replace(hour=parts[0], minute=parts[1], second=0, microsecond=0)
    elif len(parts) == 3:
        target = now.replace(hour=parts[0], minute=parts[1], second=parts[2], microsecond=0)
    else:
        return

    if now < target:
        diff_sec = (target - now).total_seconds()
        print(f"[*] Scheduled start time set to: {target.strftime('%H:%M:%S')}")
        print(f"[*] Waiting {diff_sec:.1f} seconds until start time...")
        while datetime.datetime.now() < target:
            remaining = (target - datetime.datetime.now()).total_seconds()
            if remaining > 5:
                time.sleep(1)
            else:
                time.sleep(0.1)
        print("[+] Start time reached! Commencing validation suite.")


def display_transition_alert(cycle_info: dict, countdown_sec: int = 5):
    """Visual banner, audio alert, and countdown to prompt manual hardware changes."""
    print("\n" + "#" * 80)
    print(f"  >>> HARDWARE TRANSITION ALERT: {cycle_info['name']} <<<")
    print(f"  EXPECTED STATE: [ {cycle_info['expected']} ]")
    print(f"  REQUIRED ACTION: {cycle_info['action']}")
    print("#" * 80)

    for sec in range(countdown_sec, 0, -1):
        play_alert(freq=1000 + (countdown_sec - sec) * 200, duration=150)
        sys.stdout.write(f"\r  [*] Transition Countdown: {sec}s remaining... (Please switch hardware now) ")
        sys.stdout.flush()
        time.sleep(1.0)

    play_alert(freq=1800, duration=400)
    print("\r  [+] PHASE ACTIVE! Recording data for 300 seconds (5 min)...             \n")


def generate_validation_plots(timeseries_data: list, cycle_summaries: list, output_image="validation_curve_40min.png"):
    """
    Generate comprehensive 40-minute diagnostic curve plots:
      Subplot 1: V1 RMS curve with cycle boundaries and expected state bands.
      Subplot 2: V2 Peak & V3 Crest factor trajectory.
      Subplot 3: Confirmed State vs Expected State timeline.
      Subplot 4: Transition latencies & Phase accuracies.
    """
    if not timeseries_data:
        return

    abs_path = os.path.abspath(output_image)
    print(f"[*] Generating 40-minute validation diagnostic curves to: {abs_path}")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)
    plt.suptitle("40-Minute Automated Hardware Validation Curve & State Analysis", fontsize=16, fontweight="bold", y=0.99)

    times = [d["elapsed_min"] for d in timeseries_data]
    v1_vals = [d["v1"] for d in timeseries_data]
    v2_vals = [d["v2"] for d in timeseries_data]
    v3_vals = [d["v3"] for d in timeseries_data]
    
    state_map = {"NO LOAD": 0, "Charger": 1, "Bulb": 2}
    confirmed_num = [state_map.get(d["confirmed"], -1) for d in timeseries_data]
    expected_num = [state_map.get(d["expected"], -1) for d in timeseries_data]

    # Cycle boundaries in minutes: 0, 5, 10, 15, 20, 25, 30, 35, 40
    cycle_bounds = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    cycle_colors = ["#E8F8F5", "#EBF5FB", "#E8F8F5", "#FEF9E7", "#E8F8F5", "#E8F8F5", "#EBF5FB", "#E8F8F5"]

    # 1. V1 RMS Signal Curve
    ax1 = axes[0]
    ax1.plot(times, v1_vals, color="#1B4F72", linewidth=1.2, label="V1 (RMS Voltage/Current Proxy)")
    ax1.set_ylabel("V1 (RMS)", fontsize=11, fontweight="bold")
    ax1.set_title("1. Tri-Axial V1 (RMS) Dynamic Response & Hysteresis Discharge Curve", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    for i in range(len(cycle_colors)):
        ax1.axvspan(cycle_bounds[i], cycle_bounds[i+1], color=cycle_colors[i], alpha=0.5)

    # 2. V2 Peak & V3 Crest Factor
    ax2 = axes[1]
    ax2.plot(times, v2_vals, color="#E67E22", linewidth=1.1, label="V2 (Peak Amplitude)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(times, v3_vals, color="#27AE60", linewidth=1.1, linestyle="--", label="V3 (Crest Factor)")
    ax2.set_ylabel("V2 (Peak)", fontsize=11, fontweight="bold", color="#E67E22")
    ax2_twin.set_ylabel("V3 (Crest Factor)", fontsize=11, fontweight="bold", color="#27AE60")
    ax2.set_title("2. V2 (Peak) and V3 (Crest Factor) Trajectories", fontsize=12, fontweight="bold")
    for i in range(len(cycle_colors)):
        ax2.axvspan(cycle_bounds[i], cycle_bounds[i+1], color=cycle_colors[i], alpha=0.5)

    # 3. State Tracking Timeline
    ax3 = axes[2]
    ax3.step(times, expected_num, where="post", color="#7F8C8D", linewidth=2.0, linestyle="--", label="Expected State")
    ax3.step(times, confirmed_num, where="post", color="#8E44AD", linewidth=1.8, label="Confirmed State (Debounced)")
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["NO LOAD", "Charger", "Bulb"], fontsize=10, fontweight="bold")
    ax3.set_ylabel("Load State", fontsize=11, fontweight="bold")
    ax3.set_title("3. Debounced Classification vs. Protocol Schedule", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right")
    for i in range(len(cycle_colors)):
        ax3.axvspan(cycle_bounds[i], cycle_bounds[i+1], color=cycle_colors[i], alpha=0.5)

    # 4. Phase Metrics Bar
    ax4 = axes[3]
    phases = [c["name"].split(" ")[0] + f" ({c['expected']})" for c in PROTOCOL_CYCLES]
    accuracies = [c["accuracy"] * 100 for c in cycle_summaries]
    bars = ax4.bar(range(len(phases)), accuracies, color="#16A085", edgecolor="black", width=0.55)
    ax4.set_xticks(range(len(phases)))
    ax4.set_xticklabels(phases, fontsize=9, fontweight="bold")
    ax4.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax4.set_ylim(0, 115)
    ax4.set_title("4. Cycle-by-Cycle Classification Accuracy", fontsize=12, fontweight="bold")
    for b in bars:
        h = b.get_height()
        ax4.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(abs_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Diagnostic curves saved successfully to: {abs_path}")


def run_automated_validation(port="COM3", baud=115200, model_path="load_classifier.pkl",
                             report_json="validation_report_40min.json", start_at="14:20:00",
                             cycle_duration=300):
    """
    Main 40-minute automated hardware validation suite loop.
    """
    print("=" * 80)
    print("NILM 40-MINUTE HARDWARE AUTOMATED VALIDATION SUITE")
    print("=" * 80)
    print(f"[*] Serial Port:       {port} @ {baud} baud")
    print(f"[*] Classifier Model:  {model_path}")
    print(f"[*] Report Output:     {report_json}")
    print(f"[*] Cycle Duration:    {cycle_duration} seconds (5 min) x 8 cycles = {cycle_duration*8/60:.0f} mins")

    # 1. Load trained classifier pipeline
    try:
        pipeline = load_classifier(model_path)
        print("[+] Classifier pipeline loaded successfully.")
    except Exception as e:
        print(f"[!] Error loading classifier: {e}")
        return False

    # 2. Open serial connection
    if serial is None:
        print("[!] PySerial module is not installed.")
        return False

    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)  # Microcontroller stabilization
        print(f"[+] Serial port {port} opened successfully.")
    except Exception as e:
        print(f"[!] Could not open serial port {port}: {e}")
        return False

    # 3. Handle scheduled start time
    if start_at:
        wait_until_start_time(start_at)

    pattern = re.compile(r"([0-9]+\.?[0-9]*)\s*,\s*([0-9]+\.?[0-9]*)\s*,\s*([0-9]+\.?[0-9]*)")
    state_filter = NILMStateFilter(window_size=8, transition_threshold=0.70)

    overall_start_time = time.time()
    timeseries_data = []
    cycle_summaries = []

    print("\n" + "=" * 95)
    print(f"{'TIME':<8} | {'V1':<6} {'V2':<6} {'V3':<6} | {'RAW PREDICT':<12} | {'CONFIRMED':<12} {'EXP':<10} | {'STATUS'}")
    print("=" * 95)

    try:
        for cycle_idx, cycle in enumerate(PROTOCOL_CYCLES, start=1):
            expected = cycle["expected"]
            duration = cycle_duration

            # Hardware transition alert & countdown
            display_transition_alert(cycle, countdown_sec=5)

            cycle_start = time.time()
            total_samples = 0
            correct_samples = 0
            false_positives = 0
            latency = None
            v1_list = []
            v2_list = []
            v3_list = []

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

                v1_list.append(v1)
                v2_list.append(v2)
                v3_list.append(v3)

                now_ts = time.time()
                raw_prediction = pipeline.predict([[v1, v2, v3]])[0]
                probabilities = pipeline.predict_proba([[v1, v2, v3]])[0]
                confidence = float(np.max(probabilities) * 100)

                confirmed_state, consensus_ratio, dv1, dv1_dt, event = state_filter.process(
                    v1, v2, v3, raw_prediction, confidence, current_time=now_ts
                )

                # Send debounced prediction back to ESP32 LCD
                ser.write(f"{confirmed_state}\n".encode("utf-8"))

                total_samples += 1
                is_correct = (confirmed_state == expected)
                if is_correct:
                    correct_samples += 1
                    if latency is None:
                        latency = now_ts - cycle_start
                else:
                    false_positives += 1

                elapsed_total = now_ts - overall_start_time
                elapsed_min = elapsed_total / 60.0

                timeseries_data.append({
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "elapsed_sec": round(elapsed_total, 2),
                    "elapsed_min": round(elapsed_min, 3),
                    "cycle": cycle_idx,
                    "v1": v1,
                    "v2": v2,
                    "v3": v3,
                    "raw": raw_prediction,
                    "conf": round(confidence, 1),
                    "confirmed": confirmed_state,
                    "expected": expected,
                    "ratio": round(consensus_ratio, 2),
                    "event": event,
                })

                t_str = datetime.datetime.now().strftime("%H:%M:%S")
                print(
                    f"{t_str:<8} | {v1:<6.2f} {v2:<6.2f} {v3:<6.2f} | {raw_prediction:<12} | "
                    f"{confirmed_state:<12} {expected:<10} | {event}"
                )

            # End of cycle statistics
            acc = correct_samples / total_samples if total_samples > 0 else 0.0
            summary = {
                "cycle": cycle_idx,
                "name": cycle["name"],
                "expected": expected,
                "total_samples": total_samples,
                "correct_samples": correct_samples,
                "false_positives": false_positives,
                "accuracy": round(acc, 4),
                "latency_sec": round(latency, 2) if latency is not None else -1.0,
                "mean_v1": round(float(np.mean(v1_list)), 2) if v1_list else 0.0,
                "std_v1": round(float(np.std(v1_list)), 2) if v1_list else 0.0,
                "mean_v2": round(float(np.mean(v2_list)), 2) if v2_list else 0.0,
                "mean_v3": round(float(np.mean(v3_list)), 2) if v3_list else 0.0,
            }
            cycle_summaries.append(summary)

            print("\n" + "-" * 75)
            print(f"[+] Finished {cycle['name']} -> Accuracy: {acc*100:.1f}% | Latency: {summary['latency_sec']}s | Samples: {total_samples}")
            print("-" * 75 + "\n")

    except KeyboardInterrupt:
        print("\n[!] Validation interrupted by user.")
    finally:
        ser.close()
        print("[*] Serial port closed safely.")

    # 4. Print Phase-by-Phase Accuracy Breakdown Matrix
    print("\n" + "=" * 90)
    print("40-MINUTE VALIDATION BREAKDOWN MATRIX")
    print("=" * 90)
    print(f"{'Cycle':<8} {'Expected':<10} {'Samples':<8} {'Correct':<8} {'Accuracy':<10} {'Latency':<10} {'FP':<6} {'V1 (mean±std)':<16} {'V2':<6} {'V3':<6}")
    print("-" * 90)

    total_all = sum(s["total_samples"] for s in cycle_summaries)
    correct_all = sum(s["correct_samples"] for s in cycle_summaries)
    overall_accuracy = correct_all / total_all if total_all > 0 else 0.0

    for s in cycle_summaries:
        lat_str = f"{s['latency_sec']}s" if s['latency_sec'] >= 0 else "N/A"
        v1_stat = f"{s['mean_v1']:.1f}±{s['std_v1']:.1f}"
        print(f"Cycle {s['cycle']:<2} {s['expected']:<10} {s['total_samples']:<8} {s['correct_samples']:<8} "
              f"{s['accuracy']*100:>6.1f}%    {lat_str:<10} {s['false_positives']:<6} {v1_stat:<16} {s['mean_v2']:<6.1f} {s['mean_v3']:<6.2f}")

    print("=" * 90)
    print(f"OVERALL 40-MINUTE SUITE ACCURACY: {overall_accuracy*100:.2f}% ({correct_all}/{total_all} samples)")
    print("=" * 90 + "\n")

    # 5. Save report JSON
    final_report = {
        "metadata": {
            "test_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_minutes": round((time.time() - overall_start_time) / 60.0, 2),
            "port": port,
            "baudrate": baud,
            "model_path": model_path,
            "overall_accuracy": round(overall_accuracy, 4),
            "total_samples": total_all,
            "correct_samples": correct_all,
        },
        "cycle_summaries": cycle_summaries,
        "timeseries_data": timeseries_data,
    }

    report_path = os.path.abspath(report_json)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"[+] Full performance report saved to: {report_path}")

    # 6. Generate diagnostic visual curves
    generate_validation_plots(timeseries_data, cycle_summaries, output_image="validation_curve_40min.png")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="40-Minute Automated Hardware Validation Suite for NILM")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port (default: COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--model", type=str, default="load_classifier.pkl", help="Classifier model path")
    parser.add_argument("--output", type=str, default="validation_report_40min.json", help="Report JSON path")
    parser.add_argument("--start-at", type=str, default="", help="Optional start time HH:MM or HH:MM:SS (e.g. 14:20:00)")
    parser.add_argument("--duration", type=int, default=300, help="Duration per cycle in seconds (default: 300)")

    args = parser.parse_args()
    run_automated_validation(
        port=args.port,
        baud=args.baud,
        model_path=args.model,
        report_json=args.output,
        start_at=args.start_at,
        cycle_duration=args.duration,
    )
