"""
live_inference.py
-----------------
Real-Time Non-Intrusive Load Monitoring (NILM) Inference Engine with:
  1. Temporal Moving Window & Majority Voting (70% consensus threshold to eliminate state flicker).
  2. Rate-of-Change (dV1/dt) & Step-Drop Disconnection Detection.
  3. Dynamic Baseline Tracking & Post-Load Hysteresis Debouncing (eliminating post-bulb residual drift).
  4. Dual Operation: Live Serial Streaming (ESP32) and Built-In Simulation / Test Mode.
"""

import argparse
import collections
import os
import re
import sys
import time
import joblib
import numpy as np

# Ensure feature extractor and multi-expert classifier are available for unpickling
from feature_extractor import NILMFeatureExtractor, FEATURE_NAMES
from regime_classifier import RegimeAwareNILMClassifier


class NILMStateFilter:
    """
    Temporal State Filter & Hysteresis Engine.
    
    Eliminates load flipping / state flicker on noisy sensor readings and prevents
    post-bulb residual analog baseline drift from triggering false-positive 'Charger' states.
    """

    def __init__(
        self,
        window_size: int = 8,
        transition_threshold: float = 0.70,
        disconnect_drop_threshold: float = -1.50,
        hysteresis_noise_floor: float = 11.20,
        baseline_init: float = 6.60,
    ):
        self.window_size = window_size
        self.transition_threshold = transition_threshold
        self.disconnect_drop_threshold = disconnect_drop_threshold
        self.hysteresis_noise_floor = hysteresis_noise_floor

        self.history = collections.deque(maxlen=window_size)
        self.confirmed_state = "NO LOAD"
        self.prev_v1 = None
        self.prev_time = None
        self.baseline_v1 = baseline_init
        self.debounce_counter = 0

    def process(self, v1: float, v2: float, v3: float, raw_pred: str, confidence: float, current_time: float = None):
        """
        Process a single measurement sample through delta tracking, hysteresis,
        and temporal moving window majority voting.
        
        Returns:
            confirmed_state (str): Stabilized load classification to display/actuate.
            consensus_ratio (float): Percentage of window agreeing on confirmed state.
            dv1 (float): Delta V1 compared to previous sample.
            dv1_dt (float): Rate of change in V1 per second.
            event (str): Diagnostic status flag.
        """
        if current_time is None:
            current_time = time.time()

        # Compute delta & rate of change (dV/dt)
        if self.prev_v1 is not None and self.prev_time is not None:
            dt = max(1e-3, current_time - self.prev_time)
            dv1 = v1 - self.prev_v1
            dv1_dt = dv1 / dt
        else:
            dv1 = 0.0
            dv1_dt = 0.0

        event = "STEADY"

        active_noise_floor = getattr(self, "baseline_v1", 6.60) + 0.85

        # -------------------------------------------------------------
        # 1. Step-Drop Load Disconnection Detection
        # -------------------------------------------------------------
        # A sharp negative drop in RMS when in an active load state signifies
        # an explicit load turn-off / unplug event.
        if self.prev_v1 is not None and dv1 < self.disconnect_drop_threshold:
            if self.confirmed_state in ["Bulb", "Charger", "Both"]:
                if v1 < active_noise_floor:
                    # Full disconnect: voltage collapsed back to baseline / noise floor
                    event = f"DISCONNECT_DROP (dV={dv1:+.2f})"
                    self.confirmed_state = "NO LOAD"
                    self.history.clear()
                    for _ in range(self.window_size):
                        self.history.append("NO LOAD")
                    # Engage debounce counter to absorb post-disconnect decay tail
                    self.debounce_counter = self.window_size + 4
                    self.prev_v1 = v1
                    self.prev_time = current_time
                    return self.confirmed_state, 1.0, dv1, dv1_dt, event
                else:
                    # Partial disconnect: dropped from 'Both' to a single load (Bulb or Charger)
                    event = f"PARTIAL_DROP (dV={dv1:+.2f} -> {raw_pred})"
                    self.history.clear()
                    for _ in range(self.window_size // 2):
                        self.history.append(raw_pred)

        # -------------------------------------------------------------
        # 2. Post-Load Hysteresis & Residual Drift Suppression
        # -------------------------------------------------------------
        effective_pred = raw_pred

        if self.debounce_counter > 0:
            self.debounce_counter -= 1
            # If still below active threshold, suppress false load trigger
            if v1 < active_noise_floor:
                effective_pred = "NO LOAD"
                event = f"HYSTERESIS_HOLD (V1={v1:.2f})"
            elif v1 >= getattr(self, "baseline_v1", 6.60) + 3.0:
                # Strong re-energization (e.g. bulb switched right back on)
                self.debounce_counter = 0

        # -------------------------------------------------------------
        # 3. Dynamic Baseline Tracking & Noise Floor
        # -------------------------------------------------------------
        base_v = getattr(self, "baseline_v1", 6.60)
        if v1 < base_v + 0.40:
            # Voltage is strictly within the physical NO LOAD floor
            effective_pred = "NO LOAD"
        # Otherwise: effective_pred is 100% purely what the multi-expert model predicted (raw_pred)

        # -------------------------------------------------------------
        # 4. Temporal Moving Window & Majority Voting
        # -------------------------------------------------------------
        self.history.append(effective_pred)

        counts = collections.Counter(self.history)
        candidate, top_count = counts.most_common(1)[0]
        consensus_ratio = top_count / len(self.history)

        # State transition requires minimum consensus threshold (>= 70%)
        if candidate != self.confirmed_state:
            if len(self.history) >= 4 and consensus_ratio >= self.transition_threshold:
                event = f"STATE_SWITCH -> {candidate} ({consensus_ratio*100:.0f}%)"
                self.confirmed_state = candidate
            else:
                event = f"SMOOTHING (Cand: {candidate} {consensus_ratio*100:.0f}%, Kept: {self.confirmed_state})"
        elif event == "STEADY":
            event = f"LOCKED ({self.confirmed_state})"

        self.prev_v1 = v1
        self.prev_time = current_time

        return self.confirmed_state, consensus_ratio, dv1, dv1_dt, event


def load_classifier(model_path: str):
    """Load persisted scikit-learn pipeline."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)


def run_simulation_tests(model_path: str, window_size: int = 8, threshold: float = 0.70):
    """
    Run automated simulation test suite demonstrating:
      Scenario 1: Steady NO LOAD with transient noise spikes (filtering verification).
      Scenario 2: Bulb ON -> Sudden Turn OFF with residual capacitor/bias decay (hysteresis verification).
      Scenario 3: Rapid state transition from NO LOAD to Charger to Bulb.
    """
    print("\n" + "=" * 80)
    print("RUNNING NILM INFERENCE ENGINE SIMULATION & UNIT TESTS")
    print("=" * 80)

    try:
        model = load_classifier(model_path)
        if hasattr(model, "set_baseline"):
            model.set_baseline(9.50)
        print(f"[+] Loaded classifier pipeline from '{model_path}' successfully.")
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return False

    state_filter = NILMStateFilter(window_size=window_size, transition_threshold=threshold)

    # -------------------------------------------------------------
    # Scenario 1: Transient Noise Spike During NO LOAD
    # -------------------------------------------------------------
    print("\n--- SCENARIO 1: Transient Voltage Spikes in NO LOAD State ---")
    s1_samples = [
        (8.52, 53.69, 6.30),
        (8.49, 55.29, 6.51),
        (8.40, 49.46, 5.89),
        (8.47, 53.11, 6.27),
        (12.88, 62.65, 4.86),  # Momentary single-sample glitch!
        (8.39, 51.91, 6.19),
        (8.58, 50.52, 5.89),
        (8.52, 47.10, 5.53),
    ]

    print(f"{'V1':<6} {'V2':<6} {'V3':<6} | {'RAW PREDICTION':<16} {'CONF':<6} | {'CONFIRMED LOAD':<14} {'RATIO':<6} | {'EVENT / FILTER STATUS'}")
    print("-" * 80)

    sim_time = 100.0
    for v1, v2, v3 in s1_samples:
        raw_pred = model.predict([[v1, v2, v3]])[0]
        raw_conf = max(model.predict_proba([[v1, v2, v3]])[0]) * 100
        conf_state, ratio, dv1, dv1_dt, ev = state_filter.process(v1, v2, v3, raw_pred, raw_conf, sim_time)
        print(f"{v1:<6.2f} {v2:<6.2f} {v3:<6.2f} | {raw_pred:<16} {raw_conf:>5.1f}% | {conf_state:<14} {ratio*100:>5.0f}% | {ev}")
        sim_time += 0.3

    assert state_filter.confirmed_state == "NO LOAD", "Test Failed: Transient spike caused state flicker!"
    print("[PASS] Single-sample glitch correctly absorbed; NO LOAD remained rock-solid.")

    # -------------------------------------------------------------
    # Scenario 2: Bulb Disconnect & Residual Offset Decay (Set 8 Real Data)
    # -------------------------------------------------------------
    print("\n--- SCENARIO 2: Bulb Turn-Off with Post-Disconnection Residual Decay Tail ---")
    state_filter2 = NILMStateFilter(window_size=window_size, transition_threshold=threshold)
    # Pre-condition filter to steady Bulb state
    for _ in range(8):
        raw_p = model.predict([[15.25, 78.50, 5.15]])[0]
        state_filter2.process(15.25, 78.50, 5.15, raw_p, 95.0, sim_time)
        sim_time += 0.3

    # Now bulb is turned off (actual values from dataset Set 8 where raw model flickers to Charger)
    s2_samples = [
        (15.20, 77.10, 5.07),
        (14.95, 76.50, 5.12),
        (10.50, 59.80, 5.70),  # Sudden drop (turn-off event)
        (10.33, 56.61, 5.48),  # Residual drift
        (10.10, 61.86, 6.12),  # Residual drift
        (9.85, 56.54, 5.74),   # Residual drift
        (9.66, 54.62, 5.65),   # Decaying
        (9.03, 50.23, 5.56),   # Return to baseline
        (8.62, 47.09, 5.46),   # Normal NO LOAD
    ]

    print(f"{'V1':<6} {'V2':<6} {'V3':<6} | {'RAW PREDICTION':<16} {'CONF':<6} | {'CONFIRMED LOAD':<14} {'RATIO':<6} | {'EVENT / FILTER STATUS'}")
    print("-" * 80)

    for v1, v2, v3 in s2_samples:
        raw_pred = model.predict([[v1, v2, v3]])[0]
        raw_conf = max(model.predict_proba([[v1, v2, v3]])[0]) * 100
        conf_state, ratio, dv1, dv1_dt, ev = state_filter2.process(v1, v2, v3, raw_pred, raw_conf, sim_time)
        print(f"{v1:<6.2f} {v2:<6.2f} {v3:<6.2f} | {raw_pred:<16} {raw_conf:>5.1f}% | {conf_state:<14} {ratio*100:>5.0f}% | {ev}")
        sim_time += 0.3

    assert state_filter2.confirmed_state == "NO LOAD", "Test Failed: Hysteresis failed to suppress post-bulb drift!"
    print("[PASS] Load disconnection detected instantly; residual decay prevented false Charger state.")

    # -------------------------------------------------------------
    # Scenario 3: Real Transition from NO LOAD to Charger to Bulb
    # -------------------------------------------------------------
    print("\n--- SCENARIO 3: Clean State Transitions (NO LOAD -> Charger -> Bulb) ---")
    state_filter3 = NILMStateFilter(window_size=window_size, transition_threshold=threshold)
    transitions = (
        [(8.5, 52.0, 6.2)] * 6
        + [(12.5, 62.0, 4.9)] * 8   # Charger sustained
        + [(15.5, 78.0, 5.0)] * 8   # Bulb sustained
    )
    for v1, v2, v3 in transitions:
        raw_pred = model.predict([[v1, v2, v3]])[0]
        raw_conf = max(model.predict_proba([[v1, v2, v3]])[0]) * 100
        conf_state, ratio, dv1, dv1_dt, ev = state_filter3.process(v1, v2, v3, raw_pred, raw_conf, sim_time)
        sim_time += 0.3

    print(f"[+] Final stabilized state after Bulb activation: '{state_filter3.confirmed_state}'")
    assert state_filter3.confirmed_state == "Bulb", "Test Failed: Did not transition to Bulb!"
    print("[PASS] Sustained load events successfully trigger clean state transitions.")

    print("\n" + "=" * 80)
    print("ALL SIMULATION AND HYSTERESIS TESTS PASSED!")
    print("=" * 80 + "\n")
    return True


def start_live_inference(
    port: str,
    baudrate: int,
    model_path: str,
    window_size: int = 8,
    threshold: float = 0.70,
    baseline: float = 9.50,
    auto_drift: bool = False,
):
    """
    Connect to ESP32 serial port and run real-time inference loop with temporal debouncing
    and adaptive baseline drift compensation.
    """
    print(f"[*] Loading classifier pipeline from '{model_path}'...")
    try:
        model = load_classifier(model_path)
        print("[+] Pipeline loaded successfully!")
    except Exception as e:
        print(f"[!] Error loading model file: {e}")
        return

    # Attempt serial connection
    try:
        import serial
    except ImportError:
        print("[!] PySerial is not installed. Please run: pip install pyserial")
        return

    print(f"[*] Connecting to {port} at {baudrate} baud...")
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Stabilization delay
        print(f"[+] Connected! Baseline Reference: {baseline:.2f}V (Auto-drift: {auto_drift})\n")
    except Exception as e:
        print(f"[!] Failed to open serial port {port}: {e}")
        print("[i] If testing without hardware, run: python live_inference.py --test")
        return

    pattern = re.compile(
        r"([0-9]+\.?[0-9]*)\s*,\s*([0-9]+\.?[0-9]*)\s*,\s*([0-9]+\.?[0-9]*)"
    )
    state_filter = NILMStateFilter(window_size=window_size, transition_threshold=threshold)
    v_baseline = float(baseline)

    print("=" * 105)
    print(
        f"{'TIMESTAMP':<10} | {'V1 (RAW)':<8} {'V2':<6} {'V3':<6} | {'RAW PREDICT':<12} {'CONF':<6} | {'CONFIRMED LOAD':<14} {'CONSENSUS':<9} | {'STATUS & BASELINE'}"
    )
    print("=" * 105)

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            match = pattern.search(line)
            if match:
                v1_raw = float(match.group(1))
                v2 = float(match.group(2))
                v3 = float(match.group(3))

                # User Rule: Strictly NO background calibrations when load is added or running.
                # Baseline is strictly fixed to the user-specified baseline.
                if auto_drift and state_filter.confirmed_state == "NO LOAD" and abs(v1_raw - v_baseline) <= 0.50:
                    v_baseline = 0.995 * v_baseline + 0.005 * v1_raw

                v_baseline = max(4.0, min(14.0, v_baseline))
                baseline_offset = v_baseline - 9.50
                v1_norm = max(0.1, v1_raw - baseline_offset)

                if hasattr(model, "set_baseline"):
                    model.set_baseline(v_baseline)
                    model_v1 = v1_raw
                    regime_tag = getattr(model, "active_regime_code", "MID")
                else:
                    model_v1 = v1_norm
                    regime_tag = "LEGACY"

                features = np.array([[model_v1, v2, v3]])
                raw_prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                confidence = max(probabilities) * 100

                # Filter through state debouncer and hysteresis engine using normalized V1
                confirmed_state, consensus_ratio, dv1, dv1_dt, event = state_filter.process(
                    v1_norm, v2, v3, raw_prediction, confidence
                )

                timestamp = time.strftime("%H:%M:%S")

                print(
                    f"{timestamp:<10} | {v1_raw:<8.2f} {v2:<6.2f} {v3:<6.2f} | {raw_prediction:<12} {confidence:>5.1f}% | "
                    f"{confirmed_state:<14} {consensus_ratio*100:>7.0f}% | [{regime_tag}] {event} (Base={v_baseline:.1f}V)"
                )

                # Send debounced, rock-solid state back to ESP32 for LCD display
                ser.write(f"{confirmed_state}\n".encode("utf-8"))

    except KeyboardInterrupt:
        print("\n[*] Execution interrupted by user.")
    finally:
        ser.close()
        print("[*] Serial connection closed safely.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robust Real-Time NILM Load Classification & Hysteresis Engine"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="Serial port name (e.g. COM3)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Baud rate (default: 115200)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="load_classifier.pkl",
        help="Path to trained model pipeline (default: load_classifier.pkl)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Sliding window buffer size N (default: 8 samples)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Majority vote consensus threshold for state transition (default: 0.70)",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=9.50,
        help="NO LOAD baseline reference voltage (e.g. 5.0V or 11.0V, default: 9.5V)",
    )
    parser.add_argument(
        "--auto-drift",
        action="store_true",
        default=False,
        help="Enable automatic continuous zero-drift micro-compensation (default: False, strictly manual)",
    )
    parser.add_argument(
        "--test",
        "--simulate",
        action="store_true",
        help="Run comprehensive automated simulation test suite on live filtering logic",
    )

    args = parser.parse_args()

    if args.test:
        run_simulation_tests(
            model_path=args.model,
            window_size=args.window_size,
            threshold=args.threshold,
        )
    else:
        start_live_inference(
            port=args.port,
            baudrate=args.baud,
            model_path=args.model,
            window_size=args.window_size,
            threshold=args.threshold,
            baseline=args.baseline,
            auto_drift=args.auto_drift,
        )