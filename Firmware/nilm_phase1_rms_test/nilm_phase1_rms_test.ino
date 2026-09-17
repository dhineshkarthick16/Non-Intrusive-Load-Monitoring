/*
   NILM - 3 State Classifier
   --------------------------------
   States:
     0 = NO LOAD / <1W
     1 = 9W BULB
     2 = 12W BULB

   First:
     - Collect 30 feature readings
     - Average RMS + crest factor
     - Classify initial state

   Then:
     - Continue monitoring
     - Re-classify periodically
     - Require multiple consecutive classifications
       before changing state

   Serial Monitor only.
   LCD will be added after this works.
*/

#include <Arduino.h>
#include <math.h>

// ============================================================
// ADC / SAMPLING
// ============================================================

const int CT_PIN = 34;

const int SAMPLES_PER_WINDOW = 1000;

// Sampling delay in microseconds.
// Keep this consistent with your Phase 3 test.
const unsigned long SAMPLE_INTERVAL_US = 200;

// ============================================================
// CLASSIFICATION SETTINGS
// ============================================================

// Number of feature windows collected at startup
const int INITIAL_READINGS = 30;

// Number of readings used to confirm a state change
const int CONFIRM_READINGS = 3;

// Re-classify after this many milliseconds
const unsigned long CLASSIFY_INTERVAL_MS = 2000;

// ============================================================
// FINGERPRINTS
// ============================================================

// Based on your measured datasets.
//
// NO LOAD:
// mean RMS   ≈ 7.11
// mean crest ≈ 6.54
//
// 9W BULB:
// mean RMS   ≈ 9.17
// mean crest ≈ 5.41
//
// 12W BULB:
// mean RMS   ≈ 12.14
// mean crest ≈ 4.68

struct Fingerprint {
  const char* name;

  float rmsMean;
  float crestMean;

  // Expected variation/tolerance
  float rmsTolerance;
  float crestTolerance;

  float nominalPower;
};

Fingerprint fingerprints[] = {

  {
    "NO LOAD / <1W",
    7.11,
    6.54,
    2.0,
    1.5,
    0.0
  },

  {
    "9W BULB",
    9.17,
    5.41,
    1.2,
    1.2,
    9.0
  },

  {
    "12W BULB",
    12.14,
    4.68,
    1.0,
    1.0,
    12.0
  }
};

const int NUM_STATES =
  sizeof(fingerprints) / sizeof(fingerprints[0]);

// ============================================================
// STATE VARIABLES
// ============================================================

int currentState = -1;

int candidateState = -1;
int candidateCount = 0;

unsigned long lastClassificationTime = 0;

// ============================================================
// ENERGY / TIME TRACKING
// ============================================================

unsigned long stateStartTime = 0;

float totalEnergyWh = 0.0;

unsigned long lastEnergyUpdate = 0;

// ============================================================
// FEATURE STRUCTURE
// ============================================================

struct Features {
  float rms;
  float peak;
  float crest;
};

// ============================================================
// READ ONE FEATURE WINDOW
// ============================================================

Features readFeatures() {

  float sum = 0.0;
  float sumSquares = 0.0;

  float minValue = 4095.0;
  float maxValue = 0.0;

  unsigned long nextSample = micros();

  for (int i = 0; i < SAMPLES_PER_WINDOW; i++) {

    while ((long)(micros() - nextSample) < 0) {
      // wait
    }

    nextSample += SAMPLE_INTERVAL_US;

    int raw = analogRead(CT_PIN);

    float value = (float)raw;

    sum += value;
    sumSquares += value * value;

    if (value < minValue)
      minValue = value;

    if (value > maxValue)
      maxValue = value;
  }

  float mean = sum / SAMPLES_PER_WINDOW;

  float variance =
    (sumSquares / SAMPLES_PER_WINDOW)
    - (mean * mean);

  if (variance < 0)
    variance = 0;

  float rms = sqrt(variance);

  float peak =
    max(
      fabs(maxValue - mean),
      fabs(minValue - mean)
    );

  float crest = 0.0;

  if (rms > 0.001)
    crest = peak / rms;

  Features f;

  f.rms = rms;
  f.peak = peak;
  f.crest = crest;

  return f;
}

// ============================================================
// CLASSIFY FEATURES
// ============================================================

int classify(float rms, float crest) {

  float bestScore = 999999.0;

  int bestState = -1;

  for (int i = 0; i < NUM_STATES; i++) {

    float rmsError =
      fabs(rms - fingerprints[i].rmsMean)
      / fingerprints[i].rmsTolerance;

    float crestError =
      fabs(crest - fingerprints[i].crestMean)
      / fingerprints[i].crestTolerance;

    /*
       Equal weighting of RMS and crest factor.
    */

    float score =
      rmsError + crestError;

    if (score < bestScore) {

      bestScore = score;
      bestState = i;
    }
  }

  /*
     Reject clearly abnormal readings.

     If the best fingerprint is still very far away,
     report UNKNOWN.
  */

  if (bestScore > 3.5) {
    return -1;
  }

  return bestState;
}

// ============================================================
// PRINT CLASSIFICATION
// ============================================================

void printClassification(
  int state,
  float rms,
  float crest
) {

  Serial.println();
  Serial.println("--------------------------------");

  Serial.print("RMS: ");
  Serial.println(rms, 2);

  Serial.print("Crest factor: ");
  Serial.println(crest, 2);

  if (state == -1) {

    Serial.println("STATE: UNKNOWN");

  } else {

    Serial.print("STATE: ");
    Serial.println(fingerprints[state].name);

    Serial.print("Nominal Power: ");
    Serial.print(fingerprints[state].nominalPower, 1);
    Serial.println(" W");
  }

  Serial.println("--------------------------------");
}

// ============================================================
// CONFIRM STATE CHANGE
// ============================================================

void updateState(int detectedState) {

  // Unknown does not immediately change the state
  if (detectedState == -1) {

    candidateState = -1;
    candidateCount = 0;

    return;
  }

  // Already in this state
  if (detectedState == currentState) {

    candidateState = -1;
    candidateCount = 0;

    return;
  }

  // New candidate
  if (candidateState != detectedState) {

    candidateState = detectedState;
    candidateCount = 1;

  } else {

    candidateCount++;
  }

  Serial.print("Candidate state: ");
  Serial.print(fingerprints[candidateState].name);

  Serial.print("  confirmation ");
  Serial.print(candidateCount);
  Serial.print("/");
  Serial.println(CONFIRM_READINGS);

  // Confirm only after multiple consecutive detections
  if (candidateCount >= CONFIRM_READINGS) {

    currentState = candidateState;

    candidateState = -1;
    candidateCount = 0;

    stateStartTime = millis();

    Serial.println();
    Serial.println("******** STATE CHANGED ********");

    Serial.print("NEW STATE: ");
    Serial.println(fingerprints[currentState].name);

    Serial.println("*******************************");
  }
}

// ============================================================
// STARTUP CLASSIFICATION
// ============================================================

void initialClassification() {

  Serial.println();
  Serial.println("================================");
  Serial.println("     NILM INITIALIZATION");
  Serial.println("================================");

  Serial.println();

  Serial.print("Collecting ");
  Serial.print(INITIAL_READINGS);
  Serial.println(" readings...");

  Serial.println();

  float rmsSum = 0.0;
  float crestSum = 0.0;

  for (int i = 0; i < INITIAL_READINGS; i++) {

    Features f = readFeatures();

    rmsSum += f.rms;
    crestSum += f.crest;

    Serial.print("Reading ");
    Serial.print(i + 1);
    Serial.print("/");
    Serial.print(INITIAL_READINGS);

    Serial.print("  RMS=");
    Serial.print(f.rms, 2);

    Serial.print("  Crest=");
    Serial.println(f.crest, 2);
  }

  float meanRms =
    rmsSum / INITIAL_READINGS;

  float meanCrest =
    crestSum / INITIAL_READINGS;

  Serial.println();
  Serial.println("================================");
  Serial.println("INITIAL AVERAGE");
  Serial.println("================================");

  Serial.print("Mean RMS: ");
  Serial.println(meanRms, 3);

  Serial.print("Mean Crest: ");
  Serial.println(meanCrest, 3);

  int detectedState =
    classify(meanRms, meanCrest);

  Serial.println();

  if (detectedState == -1) {

    Serial.println("INITIAL STATE: UNKNOWN");

    currentState = -1;

  } else {

    currentState = detectedState;

    Serial.print("INITIAL STATE: ");
    Serial.println(
      fingerprints[currentState].name
    );

    Serial.print("Nominal Power: ");
    Serial.print(
      fingerprints[currentState].nominalPower,
      1
    );

    Serial.println(" W");
  }

  stateStartTime = millis();

  Serial.println();
  Serial.println("Starting continuous monitoring...");
  Serial.println();
}

// ============================================================
// UPDATE ENERGY
// ============================================================

void updateEnergy() {

  unsigned long now = millis();

  if (lastEnergyUpdate == 0) {
    lastEnergyUpdate = now;
    return;
  }

  unsigned long elapsed =
    now - lastEnergyUpdate;

  lastEnergyUpdate = now;

  if (currentState < 0)
    return;

  float power =
    fingerprints[currentState].nominalPower;

  // Do not accumulate energy for no-load
  if (power <= 0.5)
    return;

  /*
     Energy:

     Wh = W × hours

     elapsed milliseconds converted to hours
  */

  float hours =
    elapsed / 3600000.0;

  totalEnergyWh +=
    power * hours;
}

// ============================================================
// PRINT RUNNING STATUS
// ============================================================

void printStatus(
  int state,
  float rms,
  float crest
) {

  Serial.println();

  Serial.println("========== STATUS ==========");

  Serial.print("RMS: ");
  Serial.println(rms, 2);

  Serial.print("Crest: ");
  Serial.println(crest, 2);

  Serial.print("Detected: ");

  if (state == -1) {

    Serial.println("UNKNOWN");

  } else {

    Serial.println(
      fingerprints[state].name
    );

    Serial.print("Nominal Power: ");
    Serial.print(
      fingerprints[state].nominalPower,
      1
    );

    Serial.println(" W");

    if (state != 0) {

      unsigned long onTime =
        millis() - stateStartTime;

      unsigned long seconds =
        onTime / 1000;

      unsigned long hours =
        seconds / 3600;

      seconds %= 3600;

      unsigned long minutes =
        seconds / 60;

      seconds %= 60;

      Serial.print("ON TIME: ");

      if (hours < 10) Serial.print("0");
      Serial.print(hours);
      Serial.print(":");

      if (minutes < 10) Serial.print("0");
      Serial.print(minutes);
      Serial.print(":");

      if (seconds < 10) Serial.print("0");
      Serial.println(seconds);
    }
  }

  Serial.print("Energy: ");
  Serial.print(totalEnergyWh, 4);
  Serial.println(" Wh");

  Serial.print("Energy: ");
  Serial.print(totalEnergyWh / 1000.0, 6);
  Serial.println(" kWh");

  Serial.println("============================");
}

// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  delay(1000);

  analogReadResolution(12);

  pinMode(CT_PIN, INPUT);

  Serial.println();
  Serial.println();
  Serial.println("================================");
  Serial.println("       NILM 3-STATE TEST");
  Serial.println("================================");

  Serial.println();

  Serial.println("States:");
  Serial.println("0 = NO LOAD / <1W");
  Serial.println("1 = 9W BULB");
  Serial.println("2 = 12W BULB");

  Serial.println();

  /*
     Give the ADC a moment to settle.
  */

  delay(1000);

  initialClassification();

  lastClassificationTime = millis();
  lastEnergyUpdate = millis();
}

// ============================================================
// LOOP
// ============================================================

void loop() {

  unsigned long now = millis();

  if (
    now - lastClassificationTime
    >= CLASSIFY_INTERVAL_MS
  ) {

    lastClassificationTime = now;

    Features f = readFeatures();

    int detectedState =
      classify(f.rms, f.crest);

    printClassification(
      detectedState,
      f.rms,
      f.crest
    );

    updateState(detectedState);

    updateEnergy();

    printStatus(
      currentState,
      f.rms,
      f.crest
    );
  }
}