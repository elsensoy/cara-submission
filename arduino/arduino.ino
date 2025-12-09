// const int motorPin = 3;       // Motor control pin
// const int pawButton = 2;      // Button input pin

// bool motorState = false;      // Keeps track of motor ON/OFF
// bool lastButtonState = HIGH;  // Used for button state change detection

// unsigned long lastDebounceTime = 0;
// const unsigned long debounceDelay = 200; // milliseconds

// void setup() {
//   pinMode(motorPin, OUTPUT);
//   pinMode(pawButton, INPUT_PULLUP);  // Internal pull-up enabled
//   Serial.begin(9600);
//   delay(500); // Give serial monitor time to open
//   Serial.println("🧸 Cara Diagnostic Mode Initialized");
//   Serial.println("--------------------------------------------------");
//   Serial.println("Setup complete. Awaiting paw button interaction...");
// }

// void loop() {
//   // Read button state
//   bool currentButtonState = digitalRead(pawButton);

//   // Detect button press (transition from HIGH to LOW)
//   if ((lastButtonState == HIGH) && (currentButtonState == LOW)) {
//     unsigned long currentTime = millis();
//     if (currentTime - lastDebounceTime > debounceDelay) {
//       Serial.println("\n Paw button was just pressed!");

//       // Toggle motor state
//       motorState = !motorState;

//       if (motorState) {
//         Serial.println("🔄 Motor turning ON (PWM = 200)");
//         Serial.println(" If motor does NOT move, check:");
//         Serial.println("   - Power source voltage (Is it strong enough?)");
//         Serial.println("   - Connections (resistor, diode, transistor)");
//         Serial.println("   - Is motor overloaded or blocked?");
//         analogWrite(motorPin, 200); // PWM value between 0–255
//       } else {
//         Serial.println("⛔ Motor turning OFF by paw press.");
//         digitalWrite(motorPin, LOW);
//       }

//       lastDebounceTime = currentTime;
//     }
//   }

//   // Continuously report motor status every few seconds (optional)
//   static unsigned long lastStatusPrint = 0;
//   if (millis() - lastStatusPrint > 3000) {
//     Serial.print("Status: Motor is ");
//     Serial.println(motorState ? "ON" : "OFF");
//     lastStatusPrint = millis();
//   }

//   // Save button state for edge detection
//   lastButtonState = currentButtonState;
// }
const int MOTOR_PIN = 3;
const int BLINK_PIN = 4;

void setup() {
  pinMode(MOTOR_PIN, OUTPUT);
  pinMode(BLINK_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);
  digitalWrite(BLINK_PIN, LOW);

  Serial.begin(9600);
  Serial.println("Arduino ready");
}

void loop() {
  if (Serial.available() > 0) {
    char emotion = Serial.read();

    if (emotion == 'H') {
      Serial.println("Happy received -> move head");
      digitalWrite(MOTOR_PIN, HIGH);
      delay(500);
      digitalWrite(MOTOR_PIN, LOW);
    } 
    else if (emotion == 'B') {
      Serial.println("Sad received -> blink");
      digitalWrite(BLINK_PIN, HIGH);
      delay(300);
      digitalWrite(BLINK_PIN, LOW);
    } 
    else {
      Serial.println("Neutral or unknown -> do nothing");
    }
  }
}

