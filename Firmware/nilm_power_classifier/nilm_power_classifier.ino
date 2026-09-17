// ==============================================================
// EdgeWatt Phase 3 - NILM Classifier & Energy Meter
// Updated with real Phase 3 data, true Z-Score distance,
// multi-modal cluster support, and temporal hysteresis.
// ==============================================================

#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

const int ADC_PIN = 34;

// 50 cycles @ 50Hz = 1000ms (1 second). Smooths out SMPS burst-mode pulses.
const int CYCLES_TO_AVERAGE = 50; 
const unsigned long WINDOW_MS = CYCLES_TO_AVERAGE * 20;
const unsigned long SAMPLE_INTERVAL_US = 250;   // ~4 kHz sampling
const int MAX_SAMPLES = 4500;

// ---- Measured Calibration Constants ----
const double NOISE_FLOOR_RMS = 7.62;             // Baseline no-load RMS
const double AMPS_PER_COUNT = 0.06 / (12.14 - 7.62); // ~0.01327 A/count (Anchored to 12W)
const double MAINS_VOLTAGE = 230.0;

// ---- Fingerprint Definition ----
struct Fingerprint {
  const char* name;
  uint8_t parentId;   // Maps multiple clusters to the same physical load
  double rmsMean;
  double rmsStd;
  double crestMean;
  double crestStd;
  double pf;          // Power Factor
};

// Registered Physical Loads
const char* APPLIANCE_NAMES[] = {"No Load / <1W", "9W Bulb", "12W Bulb", "Phone Charger"};
const int NUM_PHYSICAL_APPLIANCES = 4;

// Calibration Table (Includes Charger Idle & Active Modes)
const Fingerprint knownClusters[] = {
  // name,             parentId, rmsMean, rmsStd, crestMean, crestStd, pf
  {"No Load / <1W",    0,        7.62,    0.70,   6.39,      0.73,     1.0},
  {"9W Bulb",          1,        9.17,    0.52,   5.41,      0.59,     1.0},
  {"12W Bulb",         2,        12.14,   0.44,   4.68,      0.48,     1.0},
  {"Phone Charger",    3,        24.08,   1.20,   3.18,      0.40,     0.6}, // Charger Idle
  {"Phone Charger",    3,        42.48,   1.00,   2.51,      0.30,     0.6}  // Charger Active
};
const int NUM_CLUSTERS = sizeof(knownClusters) / sizeof(knownClusters[0]);

const double REJECT_Z_SCORE = 3.5;    // Distance threshold before flagging "Unknown"
const double ANOMALY_Z_SCORE = 2.5;   // Z-score offset to trigger anomaly banner

// ---- Energy & Time Accumulators (Indexed by parentId) ----
double energyWh[NUM_PHYSICAL_APPLIANCES] = {0};
unsigned long onTimeSeconds[NUM_PHYSICAL_APPLIANCES] = {0};

// ---- Standardized Mahalanobis / Z-Score Distance ----
double calculateZDistance(Fingerprint fp, double rms, double crest) {
  double zRms = (rms - fp.rmsMean) / fp.rmsStd;
  double zCrest = (crest - fp.crestMean) / fp.crestStd;
  return sqrt(zRms * zRms + zCrest * zCrest);
}

// ---- Temporal Hysteresis Filter (3-Sample Majority Vote) ----
int applyHysteresis(int currentBestClusterIdx) {
  static int history[3] = {0, 0, 0};
  static int histIdx = 0;

  history[histIdx] = currentBestClusterIdx;
  histIdx = (histIdx + 1) % 3;

  if (history[0] == history[1] || history[0] == history[2]) return history[0];
  if (history[1] == history[2]) return history[1];

  return currentBestClusterIdx;
}

void setup() {
  Serial.begin(115200);
  delay(500);

  analogReadResolution(12);
  analogSetPinAttenuation(ADC_PIN, ADC_11db);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("EdgeWatt NILM");
  lcd.setCursor(0, 1);
  lcd.print("System Ready");
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

  // --- Feature Extraction ---
  double mean = 0;
  for (long i = 0; i < sampleCount; i++) mean += samples[i];
  mean /= sampleCount;

  double sumOfSquares = 0;
  double peakDeviation = 0;
  for (long i = 0; i < sampleCount; i++) {
    double diff = samples[i] - mean;
    sumOfSquares += diff * diff;
    double absDiff = abs(diff);
    if (absDiff > peakDeviation) peakDeviation = absDiff;
  }
  double rmsCounts = sqrt(sumOfSquares / sampleCount);
  double crestFactor = (rmsCounts > 0.01) ? (peakDeviation / rmsCounts) : 0;

  // --- Z-Score Distance Nearest Neighbor ---
  int bestClusterIdx = 0;
  double minDistance = 999999;

  for (int i = 0; i < NUM_CLUSTERS; i++) {
    double dist = calculateZDistance(knownClusters[i], rmsCounts, crestFactor);
    if (dist < minDistance) {
      minDistance = dist;
      bestClusterIdx = i;
    }
  }

  // Smooth out transitional flicker
  int stableClusterIdx = applyHysteresis(bestClusterIdx);
  Fingerprint matchedFp = knownClusters[stableClusterIdx];
  uint8_t parentId = matchedFp.parentId;

  bool isKnown = (minDistance <= REJECT_Z_SCORE);
  const char* label = isKnown ? APPLIANCE_NAMES[parentId] : "Unknown Load";

  // --- Power & Energy Calculation ---
  double watts = 0;
  bool anomaly = false;

  if (isKnown && parentId != 0) { // If known and not "No Load"
    double signalRms = 0;
    if (rmsCounts > NOISE_FLOOR_RMS) {
      signalRms = sqrt(rmsCounts * rmsCounts - NOISE_FLOOR_RMS * NOISE_FLOOR_RMS);
    }

    double amps = signalRms * AMPS_PER_COUNT;
    watts = MAINS_VOLTAGE * amps * matchedFp.pf;

    // Anomaly Detection based on standard deviation distance
    anomaly = (minDistance > ANOMALY_Z_SCORE);

    // Accumulate metrics to the correct physical appliance
    energyWh[parentId] += watts * (WINDOW_MS / 3600000.0);
    onTimeSeconds[parentId] += WINDOW_MS / 1000;
  }

  // --- Serial Logging Output ---
  Serial.print(label);
  Serial.print(", ");
  Serial.print(watts, 2);
  Serial.print(", ");
  Serial.print(energyWh[parentId], 3);
  Serial.print(", ");
  Serial.println(anomaly ? "YES" : "no");

  // --- Display Output ---
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(label);
  lcd.setCursor(0, 1);

  if (anomaly) {
    lcd.print("ANOMALY DETECTED");
  } else if (isKnown && parentId != 0) {
    char buf[17];
    snprintf(buf, 17, "%.1fW  %.2fWh", watts, energyWh[parentId]);
    lcd.print(buf);
  } else {
    lcd.print("No load detected");
  }
}