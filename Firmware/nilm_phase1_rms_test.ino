#define MODE_PLOT 0
#define MODE_RMS  1
#define MODE MODE_RMS   // <-- change this to switch modes

const int ADC_PIN = 34;

// Mains is 50Hz -> one cycle = 20ms. Average over 10 cycles = 200ms,
// long enough to smooth out noise, short enough to stay responsive.
const int CYCLES_TO_AVERAGE = 10;
const unsigned long WINDOW_MS = CYCLES_TO_AVERAGE * 20;

// Sample as fast as the ADC comfortably allows on the ESP32.
// analogRead() alone (no special config) gives ~2-4 kHz typically -
// enough for this test since we only need RMS, not fine transient shape.
const unsigned long SAMPLE_INTERVAL_US = 250; // ~4 kHz

void setup() {
  Serial.begin(115200);
  delay(500);

  // ESP32 ADC setup - 12-bit resolution, 0-3.3V range (default attenuation
  // may not reach full 3.3V; ADC_11db gives closest to full-scale).
  analogReadResolution(12);
  analogSetPinAttenuation(ADC_PIN, ADC_11db);

  if (MODE == MODE_RMS) {
    Serial.println("Mode: RMS over cycles. Format: rms_counts, rms_millivolts");
  } else {
    Serial.println("Mode: Raw waveform plot. Open Tools > Serial Plotter now.");
  }
}

void loop() {
  if (MODE == MODE_PLOT) {
    runPlotMode();
  } else {
    runRmsMode();
  }
}

// ---- Mode 1: stream raw samples for visual waveform check ----
void runPlotMode() {
  int raw = analogRead(ADC_PIN);
  Serial.println(raw);
  delayMicroseconds(SAMPLE_INTERVAL_US);
}

// ---- Mode 2: RMS over a fixed time window ----
void runRmsMode() {
  unsigned long windowStart = millis();
  double sumOfSquares = 0;
  long sampleCount = 0;

  // First pass: find the DC bias (mean) of this window, so RMS
  // reflects only the AC component riding on top of VBIAS.
  // We buffer into a small array; adjust size if you increase WINDOW_MS.
  const int MAX_SAMPLES = 1200;
  static int samples[MAX_SAMPLES];

  while (millis() - windowStart < WINDOW_MS && sampleCount < MAX_SAMPLES) {
    samples[sampleCount] = analogRead(ADC_PIN);
    sampleCount++;
    delayMicroseconds(SAMPLE_INTERVAL_US);
  }

  if (sampleCount < 2) return;

  double mean = 0;
  for (long i = 0; i < sampleCount; i++) {
    mean += samples[i];
  }
  mean /= sampleCount;

  for (long i = 0; i < sampleCount; i++) {
    double diff = samples[i] - mean;
    sumOfSquares += diff * diff;
  }
  double rmsCounts = sqrt(sumOfSquares / sampleCount);

  // Convert counts to millivolts at the ADC pin (12-bit, ~3.3V full scale)
  double rmsMillivolts = rmsCounts * (3300.0 / 4095.0);

  Serial.print(rmsCounts, 2);
  Serial.print(", ");
  Serial.println(rmsMillivolts, 2);
}
