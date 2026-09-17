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

# Ensure feature extractor is importable for model pipeline
from feature_extractor import NILMFeatureExtractor, FEATURE_NAMES
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

        # State filter & debouncer
        self.state_filter = NILMStateFilter(window_size=8, transition_threshold=0.70)

        # Calibration parameters (defaults)
        self.v_mains = 230.0
        self.pf = 0.95
        self.k_cal = 1.0
        self.cost_per_kwh = 8.0

        # Accounting trackers
        self.current_state = "NO LOAD"
        self.current_v1 = 0.0
        self.current_v2 = 0.0
        self.current_v3 = 0.0
        self.current_power = 0.0
        self.current_confidence = 0.0
        self.consensus_ratio = 1.0
        self.last_event = "INITIALIZING"

        # Appliance stopwatches and energy accumulators
        self.session_start_time = time.time()
        self.last_update_time = time.time()
        self.device_runtime = {"NO LOAD": 0.0, "Charger": 0.0, "Bulb": 0.0}
        self.device_energy_kwh = {"NO LOAD": 0.0, "Charger": 0.0, "Bulb": 0.0}
        self.device_power_samples = {"Charger": [], "Bulb": []}

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
            self.device_runtime = {"NO LOAD": 0.0, "Charger": 0.0, "Bulb": 0.0}
            self.device_energy_kwh = {"NO LOAD": 0.0, "Charger": 0.0, "Bulb": 0.0}
            self.device_power_samples = {"Charger": [], "Bulb": []}
            self.history_samples.clear()
            self.event_logs.clear()
            self.event_logs.append({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "event": "ACCOUNTING_RESET",
                "state": self.current_state,
                "confidence": 100.0,
            })

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
                sim_step += 1
                # Cycle between NO LOAD (steps 0-40), Charger (40-80), Bulb (80-120)
                phase = (sim_step // 40) % 3
                if phase == 0:
                    v1 = 8.8 + np.random.normal(0, 0.3)
                    v2 = 52.0 + np.random.normal(0, 2.0)
                    v3 = 5.9 + np.random.normal(0, 0.2)
                elif phase == 1:
                    v1 = 12.2 + np.random.normal(0, 0.4)
                    v2 = 62.0 + np.random.normal(0, 2.5)
                    v3 = 5.1 + np.random.normal(0, 0.2)
                else:
                    v1 = 14.8 + np.random.normal(0, 0.5)
                    v2 = 74.0 + np.random.normal(0, 3.0)
                    v3 = 5.0 + np.random.normal(0, 0.2)
                raw_data = (v1, v2, v3)
                time.sleep(0.35)
            else:
                # Hardware Serial Read
                if self.ser is None or not self.ser.is_open:
                    try:
                        self.ser = serial.Serial(self.port, self.baud, timeout=1.0)
                        self.is_connected = True
                        print(f"[+] Serial port {self.port} connected successfully.")
                    except Exception:
                        self.is_connected = False
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
                except Exception:
                    self.is_connected = False
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

            v1, v2, v3 = raw_data

            # 2. ML Inference through Pipeline
            if self.pipeline:
                try:
                    raw_pred = self.pipeline.predict([[v1, v2, v3]])[0]
                    probs = self.pipeline.predict_proba([[v1, v2, v3]])[0]
                    confidence = float(np.max(probs) * 100)
                except Exception:
                    raw_pred = "NO LOAD"
                    confidence = 50.0
            else:
                raw_pred = "NO LOAD"
                confidence = 50.0

            # 3. Apply State Filter & Hysteresis
            prev_state = self.current_state
            confirmed_state, consensus_ratio, dv1, dv1_dt, event = self.state_filter.process(
                v1, v2, v3, raw_pred, confidence, current_time=loop_now
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

                # Compute Instantaneous Power:
                # P_inst = V1 * V_mains * PF * K_cal if active load; forced to 0W if NO LOAD
                if confirmed_state != "NO LOAD":
                    p_inst = v1 * self.v_mains * self.pf * self.k_cal
                else:
                    p_inst = 0.0

                self.current_state = confirmed_state
                self.current_v1 = v1
                self.current_v2 = v2
                self.current_v3 = v3
                self.current_power = p_inst
                self.current_confidence = confidence
                self.consensus_ratio = consensus_ratio
                self.last_event = event

                # Accumulate device runtimes & energy
                self.device_runtime[confirmed_state] += dt
                if confirmed_state != "NO LOAD":
                    # kWh += (Watts * dt) / (1000 * 3600)
                    added_kwh = (p_inst * dt) / 3600000.0
                    self.device_energy_kwh[confirmed_state] += added_kwh
                    self.device_power_samples[confirmed_state].append(p_inst)

                # Append to time-series chart buffer
                ts_str = datetime.datetime.now().strftime("%H:%M:%S")
                self.history_samples.append({
                    "timestamp": ts_str,
                    "datetime": datetime.datetime.now(),
                    "v1": round(v1, 2),
                    "power_watts": round(p_inst, 1),
                    "state": confirmed_state,
                    "confidence": round(confidence, 1),
                })

                # Log state transition events
                if confirmed_state != prev_state or "DISCONNECT" in event or "STATE_CHANGE" in event:
                    self.event_logs.appendleft({
                        "timestamp": ts_str,
                        "event": event,
                        "from_state": prev_state,
                        "to_state": confirmed_state,
                        "confidence": round(confidence, 1),
                        "v1": round(v1, 2),
                    })

    def get_snapshot(self):
        """Retrieve thread-safe immutable snapshot for rendering."""
        with self.lock:
            total_energy_kwh = sum(self.device_energy_kwh.values())
            total_cost = total_energy_kwh * self.cost_per_kwh

            avg_power = {}
            for dev in ["Charger", "Bulb"]:
                samples = self.device_power_samples[dev]
                avg_power[dev] = float(np.mean(samples[-50:])) if samples else 0.0

            return {
                "port": self.port,
                "baud": self.baud,
                "is_connected": self.is_connected,
                "simulation_mode": self.simulation_mode,
                "current_state": self.current_state,
                "current_v1": self.current_v1,
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
                "history": list(self.history_samples),
                "events": list(self.event_logs),
            }


@st.cache_resource
def get_pipeline_manager():
    """Singleton pipeline manager instance."""
    return SerialPipelineManager()


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
manager = get_pipeline_manager()

# ------------------------------------------------------------------------------
# Sidebar - Configuration & Calibration Controls
# ------------------------------------------------------------------------------
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

    selected_port = st.selectbox("Serial Port", options=available_ports, index=0)
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
    st.markdown(
        f"""
        <div style="text-align: right; padding-top: 10px;">
            <span style="font-size: 13px; color: #555;">PORT: <b>{snapshot['port']}</b> @ {snapshot['baud']}</span><br>
            <span style="font-size: 14px; color: {status_color}; font-weight: bold;">● {conn_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# ------------------------------------------------------------------------------
# Top Row - Live Metric Cards
# ------------------------------------------------------------------------------
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

badge_class = "badge-no-load"
if snapshot["current_state"] == "Charger":
    badge_class = "badge-charger"
elif snapshot["current_state"] == "Bulb":
    badge_class = "badge-bulb"

with col_m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <span style="font-size: 12px; color: #7f8c8d; text-transform: uppercase; font-weight: 600;">ACTIVE LOAD STATE</span><br>
            <div style="margin-top: 6px;">
                <span class="{badge_class}">{snapshot['current_state']}</span>
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
            <span style="font-size: 12px; color: #888;">
                V1: {snapshot['current_v1']:.2f} | V2: {snapshot['current_v2']:.1f} | V3: {snapshot['current_v3']:.2f}
            </span>
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
                Estimated Cost: {snapshot['total_cost']:.2f} (Tariff: {snapshot['cost_per_kwh']:.1f})
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
                Filter Status: {snapshot['last_event']}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Middle Row - Real-Time Charts (Plotly)
# ------------------------------------------------------------------------------
col_c1, col_c2 = st.columns([2, 1])

with col_c1:
    st.subheader("📈 Live Power Timeline (Rolling 2 Minutes)")
    history = snapshot["history"]

    if history:
        df_hist = pd.DataFrame(history)
        fig_timeline = make_subplots(specs=[[{"secondary_y": True}]])

        # Primary Axis: Instantaneous Power (Watts)
        fig_timeline.add_trace(
            go.Scatter(
                x=df_hist["timestamp"],
                y=df_hist["power_watts"],
                name="Power (Watts)",
                mode="lines",
                line=dict(color="#2980b9", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(41, 128, 185, 0.15)",
            ),
            secondary_y=False,
        )

        # Secondary Axis: V1 (RMS)
        fig_timeline.add_trace(
            go.Scatter(
                x=df_hist["timestamp"],
                y=df_hist["v1"],
                name="V1 (RMS Proxy)",
                mode="lines",
                line=dict(color="#e67e22", width=1.5, dash="dot"),
            ),
            secondary_y=True,
        )

        fig_timeline.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=25, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            template="plotly_white",
        )
        fig_timeline.update_xaxes(title_text="Timestamp", showgrid=True, nticks=8)
        fig_timeline.update_yaxes(title_text="Active Power (Watts)", secondary_y=False, showgrid=True)
        fig_timeline.update_yaxes(title_text="V1 (RMS)", secondary_y=True, showgrid=False)

        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("Awaiting sensor data stream...")

with col_c2:
    st.subheader("🍩 Energy Distribution (kWh)")
    kwh_data = snapshot["device_energy_kwh"]
    active_kwh = {k: v for k, v in kwh_data.items() if k != "NO LOAD" and v > 0}

    if active_kwh and sum(active_kwh.values()) > 0:
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=list(active_kwh.keys()),
                    values=list(active_kwh.values()),
                    hole=0.55,
                    marker=dict(colors=["#2980b9", "#d35400"]),
                    textinfo="label+percent",
                    insidetextorientation="radial",
                )
            ]
        )
        fig_donut.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=25, b=20),
            template="plotly_white",
            showlegend=True,
            annotations=[
                dict(
                    text=f"<b>{sum(active_kwh.values()):.4f}</b><br>kWh",
                    x=0.5,
                    y=0.5,
                    font_size=15,
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.markdown(
            """
            <div style="height: 340px; display: flex; align-items: center; justify-content: center; border: 1px dashed #ccc; border-radius: 8px; color: #888;">
                Awaiting active load energy accumulation (Charger or Bulb)...
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------------------
# Bottom Row - Device Statistics & Logs
# ------------------------------------------------------------------------------
col_b1, col_b2 = st.columns([3, 2])

with col_b1:
    st.subheader("📊 Appliance Runtime & Energy Accounting")

    runtimes = snapshot["device_runtime"]
    energies = snapshot["device_energy_kwh"]
    avg_powers = snapshot["avg_power"]

    table_records = []
    for dev in ["NO LOAD", "Charger", "Bulb"]:
        is_active = (snapshot["current_state"] == dev)
        status_tag = "🟢 ACTIVE" if is_active else "⚪ IDLE"
        kwh = energies.get(dev, 0.0)
        cost = kwh * snapshot["cost_per_kwh"]
        table_records.append({
            "Appliance": dev,
            "State Status": status_tag,
            "Total Active Time": format_hms(runtimes.get(dev, 0.0)),
            "Avg Power (W)": f"{avg_powers.get(dev, 0.0):.1f} W" if dev != "NO LOAD" else "0.0 W",
            "Energy (kWh)": f"{kwh:.5f}",
            "Cost": f"{cost:.2f}",
        })

    df_table = pd.DataFrame(table_records)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

with col_b2:
    st.subheader("📜 Event Log Stream (Last 10 Events)")
    events = snapshot["events"]

    if events:
        event_items = []
        for ev in events:
            time_str = ev.get("timestamp", "")
            event_name = ev.get("event", "")
            to_st = ev.get("to_state", ev.get("state", ""))
            conf = ev.get("confidence", 0.0)
            event_items.append({
                "Time": time_str,
                "Event Tag": event_name,
                "Load State": to_st,
                "Conf": f"{conf:.0f}%",
            })
        st.dataframe(pd.DataFrame(event_items), use_container_width=True, hide_index=True)
    else:
        st.info("No state transitions recorded yet.")


# ------------------------------------------------------------------------------
# Auto-Refresh Trigger
# ------------------------------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
