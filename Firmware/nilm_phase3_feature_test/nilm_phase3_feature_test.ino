#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define ADC_PIN 34          // GPIO pin for CT Sensor
#define SAMPLING_WINDOW 200 // Sample for 200ms

LiquidCrystal_I2C lcd(0x27, 16, 2);
String currentPrediction = "Initializing";

void setup() {
  Serial.begin(115200);
  
  Wire.begin(21, 22); // SDA = GPIO 21, SCL = GPIO 22
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("NILM Monitor");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");

  analogReadResolution(12);                   
  analogSetPinAttenuation(ADC_PIN, ADC_11db); 
  delay(1000);
  lcd.clear();
}

void loop() {
  double sumSq = 0;
  double peakVal = 0;
  int sampleCount = 0;

  // 1. Estimate DC Offset
  long sumRaw = 0;
  for (int i = 0; i < 200; i++) {
    sumRaw += analogRead(ADC_PIN);
    delayMicroseconds(100);
  }
  double dcOffset = sumRaw / 200.0;

  // 2. High-speed sampling loop
  unsigned long startTime = millis();
  while (millis() - startTime < SAMPLING_WINDOW) {
    double raw = analogRead(ADC_PIN);
    double centered = raw - dcOffset; 

    sumSq += centered * centered; 

    double absVal = abs(centered);
    if (absVal > peakVal) {
      peakVal = absVal; 
    }

    sampleCount++;
    delayMicroseconds(200); 
  }

  // 3. Feature Calculations
  double rmsVal = sqrt(sumSq / sampleCount);
  double crestFactor = (rmsVal > 0.5) ? (peakVal / rmsVal) : 1.0;

  // 4. Output to Serial Monitor for Python Model
  Serial.printf("%.2f, %.2f, %.2f\n", rmsVal, peakVal, crestFactor);

  // 5. Read incoming prediction back from Python if available
  if (Serial.available() > 0) {
    String rx = Serial.readStringUntil('\n');
    rx.trim();
    if (rx.length() > 0) {
      currentPrediction = rx;
    }
  }

  // 6. Display features on Row 0 and Prediction on Row 1
  lcd.setCursor(0, 0);
  lcd.printf("R:%.1f P:%.0f     ", rmsVal, peakVal);
  
  lcd.setCursor(0, 1);
  lcd.printf("Load: %-10s", currentPrediction.c_str());

  delay(300); 
}