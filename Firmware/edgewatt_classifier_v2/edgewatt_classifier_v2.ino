// ============================================================
// EdgeWatt - AI NILM Classifier with FFT Harmonics
// ESP32 + SCT013 + I2C 16x2 LCD
// ============================================================

#include <Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <math.h>
#include "arduinoFFT.h" 

#define LCD_ADDRESS 0x27
#define LCD_COLUMNS 16
#define LCD_ROWS 2
LiquidCrystal_I2C lcd(LCD_ADDRESS, LCD_COLUMNS, LCD_ROWS);

const int ADC_PIN = 34;

// ============================================================
// FFT & SAMPLING PARAMETERS
// ============================================================
const uint16_t SAMPLES = 512;             // Must be a power of 2
const double SAMPLING_FREQUENCY = 2560.0; // 5 Hz resolution per bin
const unsigned int SAMPLE_INTERVAL_US = 1000000 / SAMPLING_FREQUENCY;

double vReal[SAMPLES];
double vImag[SAMPLES];
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, SAMPLES, SAMPLING_FREQUENCY);

const int CONFIRM_WINDOWS = 3;
const double REJECT_ZSCORE = 6.0; 
const double ANOMALY_ZSCORE = 4.0;

// ============================================================
// FINGERPRINTS (Now includes Harmonic Ratio)
// ============================================================
struct Fingerprint {
  const char* name;
  double rmsMean;
  double rmsStd;
  double crestMean;
  double crestStd;
  double harmonicMean; // 3rd Harmonic Ratio (150Hz / 50Hz)
  double harmonicStd;
  bool isLoad;
  double nominalWatts;
};

// IMPORTANT: The harmonic values are placeholders. 
// You MUST look at the serial monitor and update these based on what it prints!
Fingerprint knownStates[] = {
  {
    "No Load",
    5.26, 0.42,
    6.91, 1.04,
    1.00, 1.00, // Replace with actual serial output
    false, 0.0
  },
  {
    "Bulb",
    13.52, 0.31,
    3.38, 0.61,
    0.15, 0.10, // Replace with actual serial output
    true, 12.0
  },
  {
    "Charger",
    8.85, 0.80,
    4.15, 1.20,
    0.60, 0.15, // Replace with actual serial output
    true, 10.0
  }
};

const int NUM_STATES = sizeof(knownStates) / sizeof(knownStates[0]);

enum LoadState { STATE_UNKNOWN = -1, STATE_NO_LOAD = 0, STATE_BULB = 1, STATE_CHARGER = 2 };

LoadState confirmedState = STATE_UNKNOWN;
LoadState candidateState = STATE_UNKNOWN;
int candidateCount = 0;
unsigned long stateStartMillis = 0;
unsigned long lastMeasurementMillis = 0;
double energyWh[NUM_STATES] = { 0.0, 0.0, 0.0 };
unsigned long onTimeSeconds[NUM_STATES] = { 0, 0, 0 };

struct Features {
  double rms;
  double peak;
  double crest;
  double harmonicRatio;
};

const char* getStateName(LoadState state) {
  switch (state) {
    case STATE_NO_LOAD: return "No Load";
    case STATE_BULB:    return "Bulb";
    case STATE_CHARGER: return "Charger";
    default:            return "Unknown";
  }
}

int stateToIndex(LoadState state) {
  if (state >= STATE_NO_LOAD && state <= STATE_CHARGER) return (int)state;
  return -1;
}

String formatHMS(unsigned long totalSeconds) {
  unsigned long hours = totalSeconds / 3600;
  unsigned long minutes = (totalSeconds % 3600) / 60;
  unsigned long seconds = totalSeconds % 60;
  char buf[17];
  if (hours > 99) hours = 99;
  snprintf(buf, sizeof(buf), "%02lu:%02lu:%02lu", hours, minutes, seconds);
  return String(buf);
}

double distanceTo(const Fingerprint& fp, double rms, double crest, double harmonicRatio) {
  double rmsZ = (rms - fp.rmsMean) / fp.rmsStd;
  double crestZ = (crest - fp.crestMean) / fp.crestStd;
  double harmonicZ = (harmonicRatio - fp.harmonicMean) / fp.harmonicStd;
  return sqrt((rmsZ * rmsZ) + (crestZ * crestZ) + (harmonicZ * harmonicZ));
}

Features readFeatures() {
  unsigned long nextSample = micros();
  
  // 1. Collect exact samples for FFT
  for (int i = 0; i < SAMPLES; i++) {
    while (micros() - nextSample < SAMPLE_INTERVAL_US) { /* Wait */ }
    nextSample += SAMPLE_INTERVAL_US;
    vReal[i] = analogRead(ADC_PIN);
    vImag[i] = 0.0;
  }

  // 2. Calculate Mean (DC Bias)
  double mean = 0.0;
  for (int i = 0; i < SAMPLES; i++) mean += vReal[i];
  mean /= SAMPLES;

  // 3. Time Domain Features
  double sumSquares = 0.0;
  double peakDeviation = 0.0;
  for (int i = 0; i < SAMPLES; i++) {
    vReal[i] = vReal[i] - mean; // Remove DC bias for FFT
    sumSquares += (vReal[i] * vReal[i]);
    if (fabs(vReal[i]) > peakDeviation) peakDeviation = fabs(vReal[i]);
  }

  double rms = sqrt(sumSquares / SAMPLES);
  double crest = (rms > 0.01) ? (peakDeviation / rms) : 0.0;

  // 4. Frequency Domain Features (FFT)
  FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
  FFT.compute(FFTDirection::Forward);
  FFT.complexToMagnitude();

  // Bin Resolution = 2560Hz / 512 = 5Hz per bin
  // Bin 10 = 50Hz (Fundamental)
  // Bin 30 = 150Hz (3rd Harmonic)
  double fundamental = vReal[10];
  double harmonic3rd = vReal[30];
  
  double harmonicRatio = 0.0;
  if (fundamental > 1.0) {
    harmonicRatio = harmonic3rd / fundamental;
  }

  return {rms, peakDeviation, crest, harmonicRatio};
}

LoadState classify(const Features& f, double& bestDistance, bool& anomaly, bool& recognized) {
  int bestIndex = -1;
  bestDistance = 999999.0;

  for (int i = 0; i < NUM_STATES; i++) {
    double distance = distanceTo(knownStates[i], f.rms, f.crest, f.harmonicRatio);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }

  recognized = (bestDistance <= REJECT_ZSCORE);
  anomaly = recognized && (bestDistance > ANOMALY_ZSCORE);
  if (!recognized) return STATE_UNKNOWN;
  return (LoadState)bestIndex;
}

bool updateState(LoadState detected) {
  if (detected == STATE_UNKNOWN || detected == confirmedState) {
    candidateState = STATE_UNKNOWN;
    candidateCount = 0;
    return false;
  }
  if (detected != candidateState) {
    candidateState = detected;
    candidateCount = 1;
  } else {
    candidateCount++;
  }

  if (candidateCount >= CONFIRM_WINDOWS) {
    confirmedState = candidateState;
    candidateState = STATE_UNKNOWN;
    candidateCount = 0;
    stateStartMillis = millis();
    return true;
  }
  return false;
}

void updateEnergy(unsigned long elapsedMs) {
  int idx = stateToIndex(confirmedState);
  if (idx < 0) return;
  double watts = knownStates[idx].nominalWatts;
  if (watts <= 0.0) return;
  double elapsedHours = elapsedMs / 3600000.0;
  energyWh[idx] += watts * elapsedHours;
  onTimeSeconds[idx] += elapsedMs / 1000;
}

void clearLCDLine(int row) {
  lcd.setCursor(0, row);
  lcd.print("                ");
  lcd.setCursor(0, row);
}

void updateLCD(double watts, bool anomaly, double distance) {
  if (confirmedState == STATE_UNKNOWN) {
    clearLCDLine(0); clearLCDLine(1);
    lcd.setCursor(0, 0); lcd.print("Fluctuation...");
    return;
  }
  int idx = stateToIndex(confirmedState);
  clearLCDLine(0);
  lcd.setCursor(0, 0);
  if (anomaly) lcd.print("ANOMALY ");
  lcd.print(getStateName(confirmedState));

  clearLCDLine(1);
  lcd.setCursor(0, 1);
  if (watts > 0) {
    lcd.print(watts, 1); lcd.print("W ");
  }
  lcd.print(formatHMS(onTimeSeconds[idx]));
}

void printStatus(const Features& f, LoadState detected, double distance) {
  Serial.println("\n==============================");
  Serial.print("RMS: "); Serial.println(f.rms, 3);
  Serial.print("Crest: "); Serial.println(f.crest, 3);
  Serial.print("Harmonic Ratio (150Hz/50Hz): "); Serial.println(f.harmonicRatio, 4);
  Serial.print("Detected: "); Serial.println(getStateName(detected));
  Serial.print("Distance (Z): "); Serial.println(distance, 3);
  Serial.println("==============================");
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(ADC_PIN, ADC_11db);
  pinMode(ADC_PIN, INPUT);

  Wire.begin(21, 22);
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.print("EdgeWatt AI");
  delay(1500);

  lastMeasurementMillis = millis();
}

void loop() {
  Features f = readFeatures();
  double bestDistance;
  bool anomaly, recognized;
  LoadState detected = classify(f, bestDistance, anomaly, recognized);
  
  unsigned long now = millis();
  updateEnergy(now - lastMeasurementMillis);
  lastMeasurementMillis = now;

  updateState(detected);
  printStatus(f, detected, bestDistance);

  int idx = stateToIndex(confirmedState);
  double watts = (idx >= 0) ? knownStates[idx].nominalWatts : 0.0;
  updateLCD(watts, anomaly, bestDistance);
  delay(20);
}