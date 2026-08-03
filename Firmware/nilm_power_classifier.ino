// ==============================================================
// NILM Classifier + Power/Energy Estimation - ESP32
// Uses steady-state features (RMS, crest factor) instead of
// switch-on transients, since Phase 2 (zero-cross) was skipped.
//
// IMPORTANT: the fingerprints and calibration constants below
// are PLACEHOLDERS. Replace them with your real Phase 3 data
// before trusting any Watt/kWh number this prints.
// ==============================================================

#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

const int ADC_PIN = 34;
const int CYCLES_TO_AVERAGE = 10;              // 10 cycles @ 50Hz = 200ms
const unsigned long WINDOW_MS = CYCLES_TO_AVERAGE * 20;
const unsigned long SAMPLE_INTERVAL_US = 250;   // ~4 kHz
const int MAX_SAMPLES = 1200;

// ---- Calibration constants: REPLACE with your own measured values ----
// Measured no-load RMS noise floor (counts), from your baseline test.
const double NOISE_FLOOR_RMS = 9.5;

// From your 12W bulb test: measured 0.06A actual, ~8.1 counts of
// signal-only RMS (noise removed). K = actual_amps / signal_counts.
const double AMPS_PER_COUNT = 0.06 / 8.1;   // ~0.0074 A per signal-count

const double MAINS_VOLTAGE = 230.0;

// ---- Appliance fingerprints: REPLACE with real Phase 3 measurements ----
// rms/crest = typical values measured for that appliance.
// spread = how much they vary across your trials (use to flag anomalies).
// pf = assumed power factor (1.0 resistive, ~0.6 typical SMPS charger).
struct Fingerprint {
  const char* name;
  double rms;
  double crest;
  double rmsSpread;
  double crestSpread;
  double pf;
};

Fingerprint knownAppliances[] = {
  {"Small Bulb (12W)", 13.0, 1.55, 2.0, 0.15, 1.0},
  {"Big Bulb (60W)",   40.0, 1.50, 4.0, 0.15, 1.0},
  {"Charger",          20.0, 2.80, 3.0, 0.30, 0.6},
};
const int numAppliances = 3;

// Scale factors so RMS and crest factor contribute comparably to distance
// (they're on very different numeric scales otherwise).
const double RMS_SCALE = 15.0;
const double CREST_SCALE = 0.8;

const double REJECT_DISTANCE = 1.6;      // beyond this -> "Unknown"
const double ANOMALY_MULTIPLIER = 2.0;   // how many "spreads" before flagging anomaly

// ---- Running energy/time totals per appliance ----
double energyWh[numAppliances] = {0};
unsigned long onTimeSeconds[numAppliances] = {0};

double distance(Fingerprint fp, double rms, double crest) {
  double dRms = (rms - fp.rms) / RMS_SCALE;
  double dCrest = (crest - fp.crest) / CREST_SCALE;
  return sqrt(dRms * dRms + dCrest * dCrest);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  analogReadResolution(12);
  analogSetPinAttenuation(ADC_PIN, ADC_11db);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("NILM Power Meter");
  delay(1500);

  Serial.println("appliance, watts, energy_wh, anomaly");
}

void loop() {
  static int samples[MAX_SAMPLES];
  unsigned long windowStart = millis();
  long sampleCount = 0;

  while (millis() - windowStart < WINDOW_MS && sampleCount < MAX_SAMPLES) {
    samples[sampleCount] = analogRead(ADC_PIN);
    sampleCount++;
    delayMicroseconds(SAMPLE_INTERVAL_US);
  }
  if (sampleCount < 2) return;

  // --- Compute RMS and peak (same method as Phase 3 test) ---
  double mean = 0;
  for (long i = 0; i < sampleCount; i++) mean += samples[i];
  mean /= sampleCount;

  double sumOfSquares = 0;
  double peakDeviation = 0;
  for (long i = 0; i < sampleCount; i++) {
    double diff = samples[i] - mean;
    sumOfSquares += diff * diff;
    double a = abs(diff);
    if (a > peakDeviation) peakDeviation = a;
  }
  double rmsCounts = sqrt(sumOfSquares / sampleCount);
  double crestFactor = (rmsCounts > 0.01) ? (peakDeviation / rmsCounts) : 0;

  // --- Remove noise floor (root-sum-square) to get signal-only RMS ---
  double signalRms = 0;
  if (rmsCounts > NOISE_FLOOR_RMS) {
    signalRms = sqrt(rmsCounts * rmsCounts - NOISE_FLOOR_RMS * NOISE_FLOOR_RMS);
  }

  // --- Classify ---
  int bestIdx = -1;
  double bestDist = 999999;
  for (int i = 0; i < numAppliances; i++) {
    double d = distance(knownAppliances[i], rmsCounts, crestFactor);
    if (d < bestDist) {
      bestDist = d;
      bestIdx = i;
    }
  }

  bool isKnown = (bestDist <= REJECT_DISTANCE) && (signalRms > 0.5);
  const char* label = isKnown ? knownAppliances[bestIdx].name : "Unknown / Off";

  // --- Estimate power (only meaningful if classified as known) ---
  double watts = 0;
  bool anomaly = false;
  if (isKnown) {
    double amps = signalRms * AMPS_PER_COUNT;
    watts = MAINS_VOLTAGE * amps * knownAppliances[bestIdx].pf;

    // Anomaly check: is this reading far outside the appliance's normal spread?
    Fingerprint fp = knownAppliances[bestIdx];
    bool rmsOff = abs(rmsCounts - fp.rms) > (ANOMALY_MULTIPLIER * fp.rmsSpread);
    bool crestOff = abs(crestFactor - fp.crest) > (ANOMALY_MULTIPLIER * fp.crestSpread);
    anomaly = rmsOff || crestOff;

    // Accumulate energy and on-time for this appliance
    energyWh[bestIdx] += watts * (WINDOW_MS / 3600000.0);
    onTimeSeconds[bestIdx] += WINDOW_MS / 1000;
  }

  // --- Output: Serial (for logging / dashboard bridging) ---
  Serial.print(label);
  Serial.print(", ");
  Serial.print(watts, 2);
  Serial.print(", ");
  Serial.print(isKnown ? energyWh[bestIdx] : 0, 3);
  Serial.print(", ");
  Serial.println(anomaly ? "YES" : "no");

  // --- Output: LCD ---
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(label);
  lcd.setCursor(0, 1);
  if (anomaly) {
    lcd.print("ANOMALY DETECTED");
  } else if (isKnown) {
    char buf[17];
    snprintf(buf, 17, "%.1fW  %.2fWh", watts, energyWh[bestIdx]);
    lcd.print(buf);
  } else {
    lcd.print("No load detected");
  }
}
