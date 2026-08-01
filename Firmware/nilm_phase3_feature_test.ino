// ==============================================================
// Phase 3 - Feature Extraction Test Sketch
// Builds on the Phase 1 RMS test: now also computes peak
// deviation and crest factor (peak / RMS), per measurement
// window. These two numbers are your calibration fingerprint
// for each appliance - write them down per appliance as you test.
// ==============================================================

const int ADC_PIN = 34;

// Mains is 50Hz -> one cycle = 20ms. Average over 10 cycles = 200ms.
const int CYCLES_TO_AVERAGE = 10;
const unsigned long WINDOW_MS = CYCLES_TO_AVERAGE * 20;

const unsigned long SAMPLE_INTERVAL_US = 250; // ~4 kHz
const int MAX_SAMPLES = 1200;

void setup() {
  Serial.begin(115200);
  delay(500);

  analogReadResolution(12);
  analogSetPinAttenuation(ADC_PIN, ADC_11db);

  Serial.println("Format: rms_counts, peak_counts, crest_factor");
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

  // Step 1: find the window's mean (DC bias / VBIAS point for this window)
  double mean = 0;
  for (long i = 0; i < sampleCount; i++) {
    mean += samples[i];
  }
  mean /= sampleCount;

  // Step 2: compute RMS of the AC component, and track peak deviation
  double sumOfSquares = 0;
  double peakDeviation = 0;

  for (long i = 0; i < sampleCount; i++) {
    double diff = samples[i] - mean;
    sumOfSquares += diff * diff;

    double absDiff = abs(diff);
    if (absDiff > peakDeviation) {
      peakDeviation = absDiff;
    }
  }

  double rmsCounts = sqrt(sumOfSquares / sampleCount);

  // Crest factor = peak / RMS. A clean sine wave (resistive load like a
  // bulb) sits close to 1.41. A switch-mode charger draws sharp current
  // pulses, so its crest factor will read noticeably higher (often 2.5+).
  double crestFactor = (rmsCounts > 0.01) ? (peakDeviation / rmsCounts) : 0;

  Serial.print(rmsCounts, 2);
  Serial.print(", ");
  Serial.print(peakDeviation, 2);
  Serial.print(", ");
  Serial.println(crestFactor, 2);
}
