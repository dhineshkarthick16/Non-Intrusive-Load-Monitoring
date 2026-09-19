# EdgeWatt: ESP32-Based Non-Intrusive Load Monitoring (NILM) System

> **A low-cost, edge-resident energy disaggregation system leveraging 13-dimensional steady-state feature engineering, real-time thermal drift calibration, and sub-millisecond ensemble classification.**

---

## 📌 Project Overview & Brief Explanation

**EdgeWatt** is a cloud-free Non-Intrusive Load Monitoring (NILM) solution designed to disaggregate electrical appliance usage across an entire facility using a single split-core current transformer (CT clamp) attached at the main panel[cite: 2]. 

By eliminating the need for individual smart plugs per appliance or heavy cloud GPUs, EdgeWatt processes raw current waveforms on a dual-core ESP32 microcontroller using FreeRTOS[cite: 2]. It combines a 13-dimensional steady-state feature extraction pipeline with dynamic thermal drift compensation ("SET AS NO LOAD") to achieve sub-millisecond local inference while maintaining complete user data privacy[cite: 2].

---

## ⚡ Key Features & Novelty

* **Edge-Native Inference:** Executes a Scikit-Learn soft-voting ensemble model directly on an ESP32 in **<0.8 ms** per window[cite: 2].
* **Dynamic Thermal Drift Offset:** Features real-time baseline subtraction ("SET AS NO LOAD") to eliminate zero-point sensor drift caused by ambient temperature changes[cite: 2].
* **Low-Power Load Disambiguation:** Leverages higher-order statistical moments and harmonic variance to separate overlapping low-wattage SMPS and resistive loads (e.g., ~8.5W phone charger vs. ~17W incandescent bulb)[cite: 2].
* **Sub-$5 Hardware Overhead:** Built entirely on open-source, commodity components rather than expensive proprietary ARM Cortex platforms[cite: 2].
* **Privacy-Preserving Telemetry:** Operates 100% locally with dual outputs: real-time appliance status on an I2C 16x2 LCD and remote monitoring via a Streamlit dashboard[cite: 2].

---

## 🚦 Current Progress

- [x] **Analog Signal Chain Design:** Hardware front-end validated using SCT-013 CT clamp, LM358 op-amp gain stage, and 4N35 zero-cross detection[cite: 2].
- [x] **Data Acquisition & Extraction:** 13-dimensional feature extraction pipeline implemented in C/C++ firmware using circular ADC buffers[cite: 2].
- [x] **Ensemble Machine Learning:** Soft-voting ensemble (GBDT + ExtraTrees + HistGB) trained and validated in Python[cite: 2].
- [x] **On-Device Firmware Conversion:** Python model decision rules exported to optimized C-header logic for direct ESP32 compilation[cite: 2].
- [x] **Benchmarking & Dashboard:** Achieved 99.1% steady-state accuracy and integrated real-time Streamlit supervisory dashboard[cite: 2].
- [ ] **Next Step:** Multi-node mesh network integration for multi-distribution board monitoring.

---

## 📊 Dataset Specifications

The project relies on **2 core hardware-validated datasets** collected directly from the physical sensing setup, totaling **10,200+ recorded samples**[cite: 2]:

1. **Multi-Regime Steady-State Dataset (8,000+ Samples):**
   * Contains raw 12-bit ADC waveform logs, 13-D feature vectors, and power estimates across isolated and combined appliance states[cite: 2].
   * Captures resistive, inductive, capacitive, and SMPS non-linear load signatures[cite: 2].

2. **40-Minute Dynamic Switching Stress Test Dataset (2,200+ Samples):**
   * Continuous multi-appliance switching sequence used to evaluate temporal debouncing filters, dynamic thermal drift tracking, and transition state latencies[cite: 2].
   * Yields a verified **91.8% dynamic tracking accuracy** under continuous real-world electrical noise[cite: 2].

---

## 🛠️ System Architecture

[ Mains AC Line ] ──► [ SCT-013 CT Clamp ]
│
▼
[ LM358 Op-Amp ] ──► (Signal Conditioning)
│
▼
[ ESP32 (GPIO34) ] ◄── [ 4N35 Optocoupler ] (Zero-Cross Sync)
│
├─► 13-D Feature Pipeline & Baseline Calibration
├─► Sub-ms Soft-Voting Ensemble Inference
│
├─► I2C 16x2 LCD Display (Local Telemetry)
└─► Streamlit Dashboard (Remote Visuals)