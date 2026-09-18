"""
dashboard.py
------------
Real-Time NILM Power Monitoring & Classification Web Dashboard.
Built with Streamlit, Plotly, PySerial, and Scikit-Learn.

Integrates with:
  - 'load_classifier.pkl' (Champion ML pipeline)
  - 'NILMStateFilter' (Temporal moving window & post-bulb hysteresis filter)
"""

import collections
import datetime
import os
import re
import sys
import threading
import time
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import serial
import serial.tools.list_ports
import streamlit as st

# Ensure feature extractor and multi-expert classifier are importable
from feature_extractor import NILMFeatureExtractor, FEATURE_NAMES
from regime_classifier import RegimeAwareNILMClassifier
from live_inference import NILMStateFilter, load_classifier

# Page Configuration
st.set_page_config(
    page_title="EdgeWatt - Real-Time NILM Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for card badges and layout polish
st.markdown(
    """
    <style>
    .metric-card {
        background: #fdfdfd;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .badge-no-load {
        background-color: #7f8c8d;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
    .badge-charger {
        background-color: #2980b9;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
    .badge-bulb {
        background-color: #d35400;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
    .badge-both {
        background-color: #8e44ad;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


class SerialPipelineManager:
    """
    Thread-safe background reader and accounting engine.
    Continuously streams from hardware COM port (or simulated generator)
    and maintains real-time metrics without blocking Streamlit UI.
    """

    def __init__(self, model_path="load_classifier.pkl"):
        self.lock = threading.Lock()
        self.model_path = model_path
        self.pipeline = None
        self._load_model()

        # Serial settings
        self.port = "COM3"
        self.baud = 115200
        self.simulation_mode = False
        self.is_connected = False
        self.ser = None
        self.last_serial_error = None

        # State filter & debouncer (window=5, threshold=0.55 for responsive debouncing without lockup)
        self.state_filter = NILMStateFilter(window_size=5, transition_threshold=0.55)

        # Calibration parameters (defaults)
        self.v_mains = 230.0
        self.pf = 0.95
        self.k_cal = 1.0
        self.cost_per_kwh = 8.0

        # Accounting trackers
        self.current_state = "NO LOAD"
        self.current_v1 = 0.0
        self.current_v1_raw = 0.0
        self.current_v1_norm = 0.0
        self.current_v2 = 0.0
        self.current_v3 = 0.0
        self.current_power = 0.0
        self.current_confidence = 0.0
        self.consensus_ratio = 1.0
        self.last_event = "INITIALIZING"
        self.current_power_health = "NORMAL"
        self.current_power_health_msg = "Standby"



        # Baseline calibration (strictly manual per user rule)
        self.nominal_baseline = 6.60
        self.v_baseline = 6.60
        self.baseline_offset = 0.0
        self.auto_baseline_tracking = False
        self.telemetry_history_5s = collections.deque(maxlen=300)
        self.last_calibration_time = None
        self.last_calibration_count = 0

        # Appliance stopwatches and energy accumulators
        self.session_start_time = time.time()
        self.last_update_time = time.time()
        self.device_runtime = {"NO LOAD": 0.0, "Charger": 0.0, "Bulb": 0.0, "Both": 0.0}
        self.device_energy_kwh = {"NO LOAD": 0.0, "Charger": 0.0, "Bulb": 0.0, "Both": 0.0}
        self.device_power_samples = {"Charger": [], "Bulb": [], "Both": []}

        # Rolling time-series buffer for 2-minute rolling chart (max 300 points @ ~2.5 Hz)
        self.history_samples = collections.deque(maxlen=300)
        # Event log stream (last 10 events)
        self.event_logs = collections.deque(maxlen=10)

        # Background worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _load_model(self):
        try:
            abs_model = os.path.abspath(self.model_path)
            self.pipeline = joblib.load(abs_model)
            print(f"[+] Dashboard Pipeline Manager loaded model: {abs_model}")
        except Exception as e:
            print(f"[!] Error loading model in Dashboard Manager: {e}")
            self.pipeline = None

    def reload_model(self):
        """Reload the model pipeline from disk and update active baseline."""
        with self.lock:
            self._load_model()
            if self.pipeline and hasattr(self.pipeline, "set_baseline"):
                self.pipeline.set_baseline(self.v_baseline)
            print(f"[+] Reloaded model pipeline from disk (Baseline={self.v_baseline:.2f}V)")

    def __getattr__(self, item):
        """Fallback attribute resolver for backward-compatibility with cached instances."""
        defaults = {
            "nominal_baseline": 6.60,
            "v_baseline": 6.60,
            "baseline_offset": 0.0,
            "auto_baseline_tracking": False,
            "current_v1_raw": 0.0,
            "current_v1_norm": 0.0,
            "current_power_health": "NORMAL",
            "current_power_health_msg": "Standby",
        }
        if item in defaults:
            setattr(self, item, defaults[item])
            return defaults[item]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def update_calibration(self, v_mains: float, pf: float, k_cal: float, cost_per_kwh: float):
        """Update calibration scalar and line voltage from sidebar controls."""
        with self.lock:
            self.v_mains = float(v_mains)
            self.pf = float(pf)
            self.k_cal = float(k_cal)
            self.cost_per_kwh = float(cost_per_kwh)

    def reset_accounting(self):
        """Reset stopwatches and cumulative energy counters."""
        with self.lock:
            self.session_start_time = time.time()
            self.last_update_time = time.time()
            self.device_runtime = {"NO LOAD": 0.0, "Charger": 0.0, "Bulb": 0.0, "Both": 0.0}
            self.device_energy_kwh = {"NO LOAD": 0.0, "Charger": 0.0, "Bulb": 0.0, "Both": 0.0}
            self.device_power_samples = {"Charger": [], "Bulb": [], "Both": []}
            self.history_samples.clear()
            self.event_logs.clear()
            self.event_logs.append({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "event": "ACCOUNTING_RESET",
                "state": self.current_state,
                "confidence": 100.0,
            })

    def set_as_no_load(self, manual_val: float = None) -> tuple[float, int]:
        """
        Calibrate the NO LOAD baseline using the overall average of the last 5 seconds of readings.
        Strictly manual: no calibrations occur when loads are added or removed.
        """
        with self.lock:
            now = time.time()
            if manual_val is not None:
                calibrated_val = float(manual_val)
                sample_count = 1
            else:
                recent_5s = [v for (t, v) in self.telemetry_history_5s if (now - t) <= 5.0]
                if recent_5s:
                    calibrated_val = float(np.mean(recent_5s))
                    sample_count = len(recent_5s)
                elif self.current_v1_raw > 0:
                    calibrated_val = float(self.current_v1_raw)
                    sample_count = 1
                else:
                    calibrated_val = self.nominal_baseline
                    sample_count = 0

            self.v_baseline = max(4.0, min(14.0, calibrated_val))
            self.baseline_offset = self.v_baseline - self.nominal_baseline
            self.last_calibration_time = datetime.datetime.now()
            self.last_calibration_count = sample_count

            # Immediately update active baseline on the multi-expert pipeline
            if self.pipeline and hasattr(self.pipeline, "set_baseline"):
                self.pipeline.set_baseline(self.v_baseline)

            # Immediately reset filter and load state to NO LOAD
            self.current_state = "NO LOAD"
            self.state_filter.confirmed_state = "NO LOAD"
            self.state_filter.history.clear()
            for _ in range(self.state_filter.window_size):
                self.state_filter.history.append("NO LOAD")

            self.event_logs.appendleft({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "event": f"SET_AS_NO_LOAD (5s Avg = {self.v_baseline:.2f}V, N={sample_count})",
                "from_state": self.current_state,
                "to_state": "NO LOAD",
                "confidence": 100.0,
                "v1": round(self.v_baseline, 2),
            })
            return self.v_baseline, sample_count

    def tare_baseline(self, manual_val: float = None):
        """Backward compatibility alias for set_as_no_load."""
        return self.set_as_no_load(manual_val)

    def reset_baseline_default(self):
        """Reset baseline reference to nominal 6.6V."""
        return self.set_as_no_load(self.nominal_baseline)

    def set_auto_tracking(self, enabled: bool):
        """No-op stub kept for compatibility. Baseline is fixed per user rule."""
        with self.lock:
            self.auto_baseline_tracking = False

    def stop(self):
        """Cleanly terminate worker loop and release serial handle."""
        self.running = False
        with self.lock:
            if self.ser:
                try:
                    if self.ser.is_open:
                        self.ser.close()
                except Exception:
                    pass
                self.ser = None
            self.is_connected = False
        try:
            if hasattr(self, "worker_thread") and self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=1.2)
        except Exception:
            pass

    def reconnect(self):
        """Force reset and reopen serial interface."""
        with self.lock:
            if self.ser:
                try:
                    if self.ser.is_open:
                        self.ser.close()
                except Exception:
                    pass
                self.ser = None
            self.is_connected = False
            self.last_serial_error = None

    def configure_serial(self, port: str, baud: int = 115200, simulation: bool = False):
        """Reconfigure serial connection or switch to simulation mode."""
        with self.lock:
            if self.port != port or self.baud != baud or self.simulation_mode != simulation:
                self.port = port
                self.baud = baud
                self.simulation_mode = simulation
                if self.ser and self.ser.is_open:
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                self.ser = None
                self.is_connected = False
                self.last_serial_error = None

    def _worker_loop(self):
        """Continuous background sampling and prediction loop."""
        pattern = re.compile(r"([0-9]+\.?[0-9]*)\s*,\s*([0-9]+\.?[0-9]*)\s*,\s*([0-9]+\.?[0-9]*)")
        sim_step = 0

        while self.running:
            raw_data = None
            loop_now = time.time()

            # 1. Acquire raw data (Hardware serial or Simulation)
            if self.simulation_mode:
                self.is_connected = True
                self.last_serial_error = None
                sim_step += 1
                # Cycle between NO LOAD (0-40), Charger (40-80), Bulb (80-120), Both (120-160)
                phase = (sim_step // 40) % 4
                if phase == 0:
                    v1 = 6.8 + np.random.normal(0, 0.15)
                    v2 = 33.0 + np.random.normal(0, 1.5)
                    v3 = 4.9 + np.random.normal(0, 0.15)
                elif phase == 1:
                    v1 = 9.5 + np.random.normal(0, 0.2)
                    v2 = 54.0 + np.random.normal(0, 2.0)
                    v3 = 5.0 + np.random.normal(0, 0.15)
                elif phase == 2:
                    v1 = 10.6 + np.random.normal(0, 0.2)
                    v2 = 72.0 + np.random.normal(0, 2.0)
                    v3 = 5.0 + np.random.normal(0, 0.15)
                else:
                    v1 = 15.8 + np.random.normal(0, 0.3)
                    v2 = 85.0 + np.random.normal(0, 2.5)
                    v3 = 5.0 + np.random.normal(0, 0.15)
                raw_data = (v1, v2, v3)
                time.sleep(0.35)
            else:
                # Hardware Serial Read
                if self.ser is None or not self.ser.is_open:
                    try:
                        self.ser = serial.Serial(self.port, self.baud, timeout=1.0)
                        self.is_connected = True
                        self.last_serial_error = None
                        print(f"[+] Serial port {self.port} connected successfully.")
                    except Exception as err:
                        self.is_connected = False
                        self.last_serial_error = str(err)
                        time.sleep(1.0)
                        continue

                try:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        m = pattern.search(line)
                        if m:
                            v1 = float(m.group(1))
                            v2 = float(m.group(2))
                            v3 = float(m.group(3))
                            raw_data = (v1, v2, v3)
                except Exception as err:
                    self.is_connected = False
                    self.last_serial_error = str(err)
                    if self.ser:
                        try:
                            self.ser.close()
                        except Exception:
                            pass
                    self.ser = None
                    time.sleep(1.0)
                    continue

            if raw_data is None:
                continue

            v1_raw, v2, v3 = raw_data

            with self.lock:
                # Buffer reading with timestamp for 5-second average calculation on user click
                self.telemetry_history_5s.append((loop_now, v1_raw))
                while self.telemetry_history_5s and (loop_now - self.telemetry_history_5s[0][0]) > 6.5:
                    self.telemetry_history_5s.popleft()

                # User Rule: Strictly NO background calibrations when load is added or running.
                # NO LOAD calibration is set solely when the user clicks "SET AS NO LOAD".
                self.baseline_offset = self.v_baseline - self.nominal_baseline
                v1_norm = max(0.1, v1_raw - self.baseline_offset)

                self.current_v1_raw = v1_raw
                self.current_v1_norm = v1_norm

            # 2. ML Inference through Pipeline (Regime-Aware Multi-Expert Routing)
            if self.pipeline:
                try:
                    if hasattr(self.pipeline, "set_baseline"):
                        self.pipeline.set_baseline(self.v_baseline)
                        model_v1 = v1_raw
                    else:
                        model_v1 = v1_norm

                    raw_pred = self.pipeline.predict([[model_v1, v2, v3]])[0]
                    probs = self.pipeline.predict_proba([[model_v1, v2, v3]])[0]
                    confidence = float(np.max(probs) * 100)
                except Exception:
                    raw_pred = "NO LOAD"
                    confidence = 50.0
            else:
                raw_pred = "NO LOAD"
                confidence = 50.0

            # 3. Apply State Filter & Hysteresis with active baseline reference
            prev_state = self.current_state
            self.state_filter.baseline_v1 = self.v_baseline
            confirmed_state, consensus_ratio, dv1, dv1_dt, event = self.state_filter.process(
                v1_raw, v2, v3, raw_pred, confidence, current_time=loop_now
            )

            # Echo confirmed prediction back over serial if open
            if self.ser and self.ser.is_open and not self.simulation_mode:
                try:
                    self.ser.write(f"{confirmed_state}\n".encode("utf-8"))
                except Exception:
                    pass

            # 4. Power & Energy Accounting
            with self.lock:
                dt = max(0.01, loop_now - self.last_update_time)
                self.last_update_time = loop_now

                # Instantaneous power measured from physical mains telemetry (0W if NO LOAD)
                # Watts = Delta_V1 * V_mains * PF * K_cal
                if confirmed_state != "NO LOAD":
                    v_delta = max(0.0, v1_raw - self.v_baseline)
                    p_inst = round(v_delta * self.v_mains * self.pf * self.k_cal, 1)
                else:
                    p_inst = 0.0

                self.current_state = confirmed_state
                self.current_v1 = v1_raw
                self.current_v2 = v2
                self.current_v3 = v3
                self.current_power = p_inst
                self.current_confidence = confidence
                self.consensus_ratio = consensus_ratio
                self.last_event = event
                self.current_power_health = "NORMAL"
                self.current_power_health_msg = "Normal"

                # Accumulate device runtimes & energy
                self.device_runtime[confirmed_state] += dt
                if confirmed_state != "NO LOAD":
                    # kWh += (Watts * dt) / (1000 * 3600)
                    added_kwh = (p_inst * dt) / 3600000.0
                    self.device_energy_kwh[confirmed_state] += added_kwh
                    self.device_power_samples[confirmed_state].append(p_inst)

                # Append to time-series chart buffer
                ts_str = datetime.datetime.now().strftime("%H:%M:%S")
                reg_code = getattr(self.pipeline, "active_regime_code", "MID") if self.pipeline else "MID"
                self.history_samples.append({
                    "timestamp": ts_str,
                    "datetime": datetime.datetime.now(),
                    "v1": round(v1_raw, 2),
                    "v2": round(v2, 2),
                    "v3": round(v3, 2),
                    "v_base": round(self.v_baseline, 2),
                    "v1_norm": round(v1_norm, 2),
                    "raw_pred": raw_pred,
                    "power_watts": round(p_inst, 1),
                    "state": confirmed_state,
                    "confidence": round(confidence, 1),
                    "consensus": round(consensus_ratio * 100, 0),
                    "status": f"[{reg_code}] {event} (Base={self.v_baseline:.1f}V)",
                })

                # Log state transition events
                if confirmed_state != prev_state or "DISCONNECT" in event or "STATE_CHANGE" in event:
                    self.event_logs.appendleft({
                        "timestamp": ts_str,
                        "event": event,
                        "from_state": prev_state,
                        "to_state": confirmed_state,
                        "confidence": round(confidence, 1),
                        "v1": round(v1_raw, 2),
                    })

        with self.lock:
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.ser = None
            self.is_connected = False

    def get_snapshot(self):
        """Retrieve thread-safe immutable snapshot for rendering."""
        with self.lock:
            total_energy_kwh = sum(self.device_energy_kwh.values())
            total_cost = total_energy_kwh * self.cost_per_kwh

            avg_power = {}
            for dev in ["Charger", "Bulb", "Both"]:
                samples = self.device_power_samples[dev]
                avg_power[dev] = float(np.mean(samples[-50:])) if samples else 0.0

            return {
                "port": self.port,
                "baud": self.baud,
                "is_connected": self.is_connected,
                "last_serial_error": self.last_serial_error,
                "simulation_mode": self.simulation_mode,
                "current_state": self.current_state,
                "current_v1": self.current_v1,
                "current_v1_raw": self.current_v1_raw,
                "current_v1_norm": self.current_v1_norm,
                "nominal_baseline": self.nominal_baseline,
                "v_baseline": self.v_baseline,
                "baseline_offset": self.baseline_offset,
                "auto_baseline_tracking": self.auto_baseline_tracking,
                "active_regime": getattr(self.pipeline, "active_regime", "MID (8-9V) [40-min Model]"),
                "active_regime_code": getattr(self.pipeline, "active_regime_code", "MID"),
                "current_v2": self.current_v2,
                "current_v3": self.current_v3,
                "current_power": self.current_power,
                "current_confidence": self.current_confidence,
                "consensus_ratio": self.consensus_ratio,
                "last_event": self.last_event,
                "total_energy_kwh": total_energy_kwh,
                "total_cost": total_cost,
                "cost_per_kwh": self.cost_per_kwh,
                "session_runtime_sec": time.time() - self.session_start_time,
                "device_runtime": dict(self.device_runtime),
                "device_energy_kwh": dict(self.device_energy_kwh),
                "avg_power": avg_power,
                "current_power_health": getattr(self, "current_power_health", "NORMAL"),
                "current_power_health_msg": getattr(self, "current_power_health_msg", "Normal power consumption"),
                "last_calibration_time": self.last_calibration_time.strftime("%H:%M:%S") if self.last_calibration_time else None,
                "last_calibration_count": self.last_calibration_count,
                "history": list(self.history_samples),
                "events": list(self.event_logs),
            }


_ACTIVE_PIPELINE_MANAGER = None


@st.cache_resource
def get_pipeline_manager(version: str = "v8_fresh_pipeline_reload"):
    """Singleton pipeline manager instance with clean teardown."""
    global _ACTIVE_PIPELINE_MANAGER
    if _ACTIVE_PIPELINE_MANAGER is not None:
        try:
            _ACTIVE_PIPELINE_MANAGER.stop()
            time.sleep(0.5)
        except Exception:
            pass
    _ACTIVE_PIPELINE_MANAGER = SerialPipelineManager()
    return _ACTIVE_PIPELINE_MANAGER


def format_hms(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    sec = int(seconds)
    hours = sec // 3600
    minutes = (sec % 3600) // 60
    rem_sec = sec % 60
    return f"{hours:02d}:{minutes:02d}:{rem_sec:02d}"


# ==============================================================================
# MAIN STREAMLIT APP
# ==============================================================================
manager = get_pipeline_manager(version="v8_fresh_pipeline_reload")

# Self-healing: if an older instance was cached before methods/attributes were added, refresh it
if not hasattr(manager, "set_as_no_load") or not hasattr(manager, "reconnect") or not hasattr(manager, "telemetry_history_5s") or not hasattr(manager, "current_power_health"):
    st.cache_resource.clear()
    manager = get_pipeline_manager("v7_center_low_660_calibrated_reloaded")

# Defensive attribute assignment for complete hot-reload resilience
_safe_defaults = {
    "nominal_baseline": 6.60,
    "v_baseline": 6.60,
    "baseline_offset": 0.0,
    "auto_baseline_tracking": False,
    "current_v1_raw": 0.0,
    "current_v1_norm": 0.0,
    "current_power_health": "NORMAL",
    "current_power_health_msg": "Standby",
    "last_serial_error": None,
    "last_calibration_time": None,
    "last_calibration_count": 0,
}
for _attr, _val in _safe_defaults.items():
    if not hasattr(manager, _attr):
        setattr(manager, _attr, _val)

if hasattr(manager, "device_runtime") and "Both" not in manager.device_runtime:
    manager.device_runtime["Both"] = 0.0
if hasattr(manager, "device_energy_kwh") and "Both" not in manager.device_energy_kwh:
    manager.device_energy_kwh["Both"] = 0.0
if hasattr(manager, "device_power_samples") and "Both" not in manager.device_power_samples:
    manager.device_power_samples["Both"] = []

# ------------------------------------------------------------------------------
# Sidebar - Configuration & Calibration Controls
# ------------------------------------------------------------------------------
snapshot = manager.get_snapshot()

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/electricity.png", width=64)
    st.title("⚡ EdgeWatt NILM")
    st.caption("Real-Time Load Disaggregation & Energy Intelligence")

    st.markdown("---")
    st.subheader("🔌 Hardware Serial Interface")

    # Available serial ports detection
    available_ports = [p.device for p in serial.tools.list_ports.comports()]
    if not available_ports:
        available_ports = ["COM3"]
    if "COM3" not in available_ports:
        available_ports.insert(0, "COM3")

    col_port1, col_port2 = st.columns([1.6, 1.4])
    with col_port1:
        selected_port = st.selectbox("Serial Port", options=available_ports, index=0)
    with col_port2:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔌 Reconnect", use_container_width=True, help="Force reconnect serial interface"):
            if hasattr(manager, "reconnect"):
                manager.reconnect()
            st.toast("Reconnecting serial port...", icon="🔌")
            st.rerun()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Reload Model", use_container_width=True, help="Force reload model pipeline from disk"):
            if hasattr(manager, "reload_model"):
                manager.reload_model()
            st.toast("Model pipeline reloaded from disk!", icon="🔄")
            st.rerun()
    with col_btn2:
        if st.button("🧹 Clear Cache", use_container_width=True, help="Clear Streamlit cache and restart pipeline"):
            st.cache_resource.clear()
            st.rerun()

    selected_baud = st.selectbox("Baud Rate", options=[9600, 57600, 115200, 230400], index=2)
    use_simulation = st.checkbox("Demo / Simulation Mode", value=False, help="Simulate live sensor telemetry without physical hardware")

    manager.configure_serial(selected_port, selected_baud, use_simulation)

    st.markdown("---")
    st.subheader("⚙️ Mains Calibration")

    v_mains = st.number_input("Line Voltage (V_mains)", min_value=100.0, max_value=250.0, value=230.0, step=10.0)
    pf = st.slider("Power Factor (PF)", min_value=0.50, max_value=1.00, value=0.95, step=0.01)
    k_cal = st.number_input("Calibration Scalar (K_cal)", min_value=0.001, max_value=10.0, value=0.04, step=0.005, format="%.4f",
                            help="Current transformer scaling factor (Watts = V1 * V_mains * PF * K_cal)")
    tariff = st.number_input("Electricity Tariff (per kWh)", min_value=0.0, max_value=100.0, value=8.0, step=0.5)

    manager.update_calibration(v_mains, pf, k_cal, tariff)

    st.markdown("---")
    st.subheader("🎯 Zero-Load Calibration")
    st.caption("Click when no appliances are connected. Averages last 5 seconds of readings as the fixed NO LOAD reference. No background calibrations occur when loads are added.")

    if st.button("SET AS NO LOAD", key="sb_set_no_load", use_container_width=True, type="primary"):
        avg_v, count = manager.set_as_no_load()
        st.toast(f"✅ NO LOAD set to {avg_v:.2f} V ({count} samples)", icon="🎯")
        st.rerun()

    regime_tag = snapshot.get("active_regime", "MID (8-9V) [40-min Model]")
    regime_code = snapshot.get("active_regime_code", "MID")
    reg_colors = {"LOW": "#27ae60", "MID": "#2980b9", "HIGH": "#8e44ad", "INTERP": "#e67e22"}
    sb_reg_color = reg_colors.get(regime_code, "#2980b9")

    st.markdown(
        f"""
        <div style="background: #f1f2f6; border-radius: 6px; padding: 8px 10px; font-size: 12px; margin-top: 6px; border: 1px solid #dcdde1;">
            <b>Active Zero Reference:</b> {getattr(manager, 'v_baseline', 6.6):.2f} V<br>
            <b>Drift Offset (ΔV):</b> {getattr(manager, 'baseline_offset', 0.0):+.2f} V<br>
            <b>Active Model:</b> <span style="color: {sb_reg_color}; font-weight: bold;">{regime_tag}</span><br>
            <b>Raw V1:</b> {getattr(manager, 'current_v1_raw', 0.0):.2f} V → <b>Norm V1:</b> {getattr(manager, 'current_v1_norm', 0.0):.2f} V
        </div>
        """,
        unsafe_allow_html=True,
    )

    hide_no_load = st.toggle("👁️ Hide NO LOAD from Views", value=False, help="Hide NO LOAD from appliance table and show only active appliances")

    st.markdown("---")
    refresh_rate = st.select_slider("Dashboard Refresh Rate", options=[0.5, 1.0, 2.0, 3.0], value=1.0)
    auto_refresh = st.toggle("Enable Live Auto-Refresh", value=True)

    if st.button("🗑️ Reset Accounting Data", use_container_width=True):
        manager.reset_accounting()
        st.toast("Accounting and energy counters reset successfully!", icon="✅")


# ------------------------------------------------------------------------------
# Top Header Section
# ------------------------------------------------------------------------------
snapshot = manager.get_snapshot()

col_header_left, col_header_right = st.columns([3, 1])
with col_header_left:
    st.title("⚡ Non-Intrusive Load Monitoring Dashboard")
    st.markdown("Real-time appliance disaggregation, instantaneous power telemetry, and energy accounting.")

with col_header_right:
    status_color = "green" if snapshot["is_connected"] else "red"
    conn_text = "SIMULATED" if snapshot["simulation_mode"] else ("CONNECTED" if snapshot["is_connected"] else "DISCONNECTED")
    err_hint = f"<br><span style='font-size: 11px; color: #e74c3c;'>{snapshot['last_serial_error'][:40]}...</span>" if not snapshot["is_connected"] and not snapshot["simulation_mode"] and snapshot.get("last_serial_error") else ""
    st.markdown(
        f"""
        <div style="text-align: right; padding-top: 10px;">
            <span style="font-size: 13px; color: #555;">PORT: <b>{snapshot['port']}</b> @ {snapshot['baud']}</span><br>
            <span style="font-size: 14px; color: {status_color}; font-weight: bold;">● {conn_text}</span>
            {err_hint}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# Top Status Bar (Front view - clean info readout, calibration button in sidebar)
# ------------------------------------------------------------------------------
offset_color = "#27ae60" if abs(snapshot['baseline_offset']) < 0.8 else ("#e67e22" if abs(snapshot['baseline_offset']) < 2.5 else "#e74c3c")
reg_tag = snapshot.get("active_regime", "MID (8-9V) [40-min Model]")
reg_code = snapshot.get("active_regime_code", "MID")
reg_colors = {"LOW": "#27ae60", "MID": "#2980b9", "HIGH": "#8e44ad", "INTERP": "#e67e22"}
reg_icons = {"LOW": "🟢", "MID": "🔵", "HIGH": "🟣", "INTERP": "🟠"}
reg_bg = reg_colors.get(reg_code, "#2980b9")
reg_ico = reg_icons.get(reg_code, "🔵")
cal_info = f" • Calibrated: <b>{snapshot['last_calibration_time']}</b> ({snapshot['last_calibration_count']} pts)" if snapshot.get("last_calibration_time") else " • <i>To set zero reference, click 'SET AS NO LOAD' in left sidebar</i>"

st.markdown(
    f"""
    <div style="background: #ffffff; border: 1px solid #dcdde1; border-radius: 8px; padding: 10px 18px; font-size: 13px; display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
        <div>
            <span>Locked NO LOAD Baseline: <b>{snapshot['v_baseline']:.2f} V</b> (Offset: <b style="color: {offset_color};">{snapshot['baseline_offset']:+.2f}V</b>)</span>
            <span style="color: #7f8c8d; font-size: 12px;">{cal_info}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 12px; color: #555;">Active Multi-Expert Model:</span>
            <span style="background: {reg_bg}; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; font-size: 11px; white-space: nowrap;">
                {reg_ico} {reg_tag}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# ------------------------------------------------------------------------------
# Top Row - Live Metric Cards (Showing Locked Device & Power Health)
# ------------------------------------------------------------------------------
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

cur_dev = snapshot["current_state"]
badge_class = "badge-no-load"
dev_icon_label = "🟢 NO LOAD"

if cur_dev == "Charger":
    badge_class = "badge-charger"
    dev_icon_label = "🔌 Charger"
elif cur_dev == "Bulb":
    badge_class = "badge-bulb"
    dev_icon_label = "💡 Bulb"
elif cur_dev == "Both":
    badge_class = "badge-both"
    dev_icon_label = "⚡ Bulb, Charger"

if hide_no_load and cur_dev == "NO LOAD":
    dev_icon_label = "⚪ IDLE (Ready)"

cur_display = "Bulb, Charger" if cur_dev == "Both" else cur_dev
power_subtext = f'<span style="color: #27ae60; font-size: 12px;">Active Load: {cur_display}</span>' if cur_dev != "NO LOAD" else '<span style="color: #888; font-size: 12px;">No active load</span>'

with col_m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 12px; color: #7f8c8d; text-transform: uppercase; font-weight: 600;">ACTIVE DEVICE</span>
            </div>
            <div style="margin-top: 6px;">
                <span class="{badge_class}" style="font-size: 13px;">{dev_icon_label}</span>
            </div>
            <span style="font-size: 12px; color: #888; margin-top: 4px; display: inline-block;">
                Consensus: {snapshot['consensus_ratio']*100:.0f}% | Conf: {snapshot['current_confidence']:.1f}%
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <span style="font-size: 12px; color: #7f8c8d; text-transform: uppercase; font-weight: 600;">INSTANTANEOUS POWER</span><br>
            <span style="font-size: 26px; font-weight: 700; color: #111;">{snapshot['current_power']:.1f} W</span><br>
            {power_subtext}
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <span style="font-size: 12px; color: #7f8c8d; text-transform: uppercase; font-weight: 600;">ENERGY CONSUMED</span><br>
            <span style="font-size: 26px; font-weight: 700; color: #16a085;">{snapshot['total_energy_kwh']:.5f} kWh</span><br>
            <span style="font-size: 12px; color: #888;">
                Estimated Cost: ₹{snapshot['total_cost']:.2f} (₹{snapshot['cost_per_kwh']:.1f}/kWh)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <span style="font-size: 12px; color: #7f8c8d; text-transform: uppercase; font-weight: 600;">SESSION RUNTIME</span><br>
            <span style="font-size: 26px; font-weight: 700; color: #2c3e50;">{format_hms(snapshot['session_runtime_sec'])}</span><br>
            <span style="font-size: 12px; color: #888;">
                Filter: {snapshot['last_event']}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Middle Row - Real-Time Charts (2 Graphs: Power & Voltage + 1 Energy Pie Chart)
# ------------------------------------------------------------------------------
col_c1, col_c2 = st.columns([1.8, 1.2])

with col_c1:
    history = snapshot["history"]
    if history:
        df_hist = pd.DataFrame(history)

        # -------------------------------------------------------------
        # GRAPH 1: Active Power Telemetry (Watts)
        # -------------------------------------------------------------
        st.subheader("⚡ Real-Time Active Power (Watts)")
        fig_power = go.Figure()
        fig_power.add_trace(
            go.Scatter(
                x=df_hist["timestamp"],
                y=df_hist["power_watts"],
                name="Power (W)",
                mode="lines+markers",
                line=dict(color="#2980b9", width=2.5),
                marker=dict(size=4),
                fill="tozeroy",
                fillcolor="rgba(41, 128, 185, 0.15)",
            )
        )


        fig_power.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=25, b=20),
            hovermode="x unified",
            template="plotly_white",
            yaxis=dict(title="Active Power (Watts)", showgrid=True),
            xaxis=dict(showgrid=True, nticks=8),
        )
        st.plotly_chart(fig_power, use_container_width=True)

        # -------------------------------------------------------------
        # GRAPH 2: Sensor Voltage Stream (Physical V1 vs Locked Floor)
        # -------------------------------------------------------------
        st.subheader("📈 Sensor Voltage Stream (Physical V1 vs Locked Baseline)")
        fig_voltage = go.Figure()
        fig_voltage.add_trace(
            go.Scatter(
                x=df_hist["timestamp"],
                y=df_hist["v1"],
                name="V1 Raw (Physical Sensor)",
                mode="lines",
                line=dict(color="#d35400", width=2.2),
            )
        )
        if "v_base" in df_hist.columns:
            fig_voltage.add_trace(
                go.Scatter(
                    x=df_hist["timestamp"],
                    y=df_hist["v_base"],
                    name="Locked NO LOAD Baseline",
                    mode="lines",
                    line=dict(color="#27ae60", width=2.0, dash="dash"),
                )
            )
        fig_voltage.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=25, b=20),
            hovermode="x unified",
            template="plotly_white",
            yaxis=dict(title="RMS Proxy Voltage (V1)", showgrid=True),
            xaxis=dict(title="Timestamp", showgrid=True, nticks=8),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_voltage, use_container_width=True)

    else:
        st.info("Awaiting sensor data stream...")

with col_c2:
    st.subheader("🍩 Energy Distribution (kWh)")
    kwh_data = snapshot["device_energy_kwh"]
    active_kwh = {k: v for k, v in kwh_data.items() if k != "NO LOAD" and v > 0}

    if active_kwh and sum(active_kwh.values()) > 0:
        color_map = {"Charger": "#2980b9", "Bulb": "#d35400", "Bulb, Charger": "#8e44ad", "Both": "#8e44ad"}
        donut_labels = ["Bulb, Charger" if k == "Both" else k for k in active_kwh.keys()]
        donut_colors = [color_map.get(lbl, "#8e44ad") for lbl in donut_labels]

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=donut_labels,
                    values=list(active_kwh.values()),
                    hole=0.58,
                    marker=dict(colors=donut_colors),
                    textinfo="label+percent",
                    insidetextorientation="radial",
                )
            ]
        )
        fig_donut.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=25, b=20),
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
            annotations=[
                dict(
                    text=f"<b>{sum(active_kwh.values()):.5f}</b><br><span style='font-size:12px;color:#777;'>Total kWh</span>",
                    x=0.5,
                    y=0.5,
                    font_size=16,
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.markdown(
            """
            <div style="height: 520px; display: flex; align-items: center; justify-content: center; border: 1px dashed #ccc; border-radius: 8px; color: #888; text-align: center; padding: 20px;">
                Awaiting active load energy accumulation (Charger, Bulb, or Bulb, Charger)...
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------------------
# Bottom Section - Appliance Energy Accounting & Terminal Log Stream
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Appliance Runtime & Energy Accounting")

runtimes = snapshot["device_runtime"]
energies = snapshot["device_energy_kwh"]
avg_powers = snapshot["avg_power"]

table_records = []
display_devices = ["NO LOAD", "Charger", "Bulb", "Both"]
if hide_no_load:
    display_devices = ["Charger", "Bulb", "Both"]

for dev in display_devices:
    is_active = (snapshot["current_state"] == dev)
    status_tag = "🟢 ACTIVE" if is_active else "⚪ IDLE"
    kwh = energies.get(dev, 0.0)
    cost = kwh * snapshot["cost_per_kwh"]

    if is_active:
        live_w = f"{snapshot['current_power']:.1f} W"
    else:
        live_w = "0.0 W"

    display_dev = "Bulb, Charger" if dev == "Both" else dev

    table_records.append({
        "Appliance": display_dev,
        "Status": status_tag,
        "Live Power": live_w,
        "Total Active Time": format_hms(runtimes.get(dev, 0.0)),
        "Energy (kWh)": f"{kwh:.5f}",
        "Cost (₹)": f"₹{cost:.2f}",
    })

df_table = pd.DataFrame(table_records)
st.dataframe(df_table, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# Terminal-Style Live Log Stream (Matching Terminal Output)
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🖥️ Live Terminal Log Stream (ESP32 Telemetry & Inference)")
st.caption("Real-time chronological telemetry records formatted identically to terminal live inference.")

history_data = snapshot.get("history", [])
if history_data:
    recent_history = list(history_data)[-20:][::-1]
    term_records = []
    for rec in recent_history:
        raw_pred_val = rec.get("raw_pred", rec.get("state", ""))
        raw_disp = "Bulb, Charger" if raw_pred_val == "Both" else raw_pred_val
        conf_state_val = rec.get("state", "")
        conf_disp = "Bulb, Charger" if conf_state_val == "Both" else conf_state_val

        term_records.append({
            "TIMESTAMP": rec.get("timestamp", ""),
            "V1 (RAW)": f"{rec.get('v1', 0.0):.2f}",
            "V2": f"{rec.get('v2', 0.0):.2f}",
            "V3": f"{rec.get('v3', 0.0):.2f}",
            "RAW PREDICT": raw_disp,
            "CONF": f"{rec.get('confidence', 0.0):.1f}%",
            "CONFIRMED LOAD": conf_disp,
            "CONSENSUS": f"{rec.get('consensus', 100):.0f}%",
            "POWER": f"{rec.get('power_watts', 0.0):.1f} W",
            "STATUS & BASELINE": rec.get("status", ""),
        })
    df_terminal = pd.DataFrame(term_records)
    st.dataframe(df_terminal, use_container_width=True, hide_index=True)
else:
    st.info("Awaiting telemetry stream from hardware...")


# ------------------------------------------------------------------------------
# Footer: Attribution & Portfolio Link
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; padding: 30px 10px 15px 10px; margin-top: 40px; border-top: 1px solid #dcdde1; font-size: 14px; color: #576574;">
        <span>© DK 2026 • </span>
        <span>Developed by <a href="https://dkportfolio-livid.vercel.app/" target="_blank" style="color: #2980b9; font-weight: 700; text-decoration: none;">Dhinesh Karthick D</a></span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# Auto-Refresh Trigger
# ------------------------------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
