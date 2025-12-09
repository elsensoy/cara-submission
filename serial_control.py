 
# Set this one variable to 'True' to use Jetson's pins.
# Set it to 'False' to send commands to the Arduino.
USE_JETSON_GPIO = False

import atexit
import time 
# We only import the libraries we actually need based on the toggle
if USE_JETSON_GPIO:
    import RPi.GPIO as GPIO
else:
    # You may need to run: pip3 install pyserial
    import serial

# Global Placeholders 
# (These will be filled by the correct init function)
pwm_motor = None
arduino_serial = None

# -----------------------------------------------------------------
# --- SECTION 1: JETSON NANO (GPIO) LOGIC ---
# -----------------------------------------------------------------
# (This logic runs if USE_JETSON_GPIO is True)

# --- Jetson Config ---
# Pin 33 (PWM) for Motor
# Pin 12 (Digital) for Eyes
JETSON_MOTOR_PIN = 33 
JETSON_EYES_PIN = 12

def _jetson_initialize():
    """
    Initializes the Jetson's GPIO pins.
    """
    global pwm_motor
    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        
        # Setup Eyes Pin (Digital)
        GPIO.setup(JETSON_EYES_PIN, GPIO.OUT, initial=GPIO.LOW)
        
        # Setup Motor Pin (PWM)
        GPIO.setup(JETSON_MOTOR_PIN, GPIO.OUT, initial=GPIO.LOW)
        pwm_motor = GPIO.PWM(JETSON_MOTOR_PIN, 100) # 100Hz
        pwm_motor.start(0) # Start with motor off
        
        print(f"--- SUCCESS: Using JETSON GPIO ---")
        print(f"DEBUG: Board model detected as: {GPIO.model}")
        print(f"  Motor (PWM) on Pin {JETSON_MOTOR_PIN}")
        print(f"  Eyes (Digital) on Pin {JETSON_EYES_PIN}")

    except Exception as e:
        print(f"--- ERROR: Could not initialize Jetson GPIO! ---")
        print(f"Error: {e}")
        print("Did you remember to run with 'sudo'?")
        print("Is your hardware patch in '/usr/lib/...' correct?")

def _jetson_send_emotion(emotion_string):
    """
    Directly controls the Jetson's pins based on the emotion.
    """
    global pwm_motor
    
    try:
        if emotion_string == "happy":
            # Uses 100% power as determined in our tests
            print("[GPIO]: Happy received -> move head (PWM 100%)")
            pwm_motor.ChangeDutyCycle(100)
            time.sleep(0.5) 
            pwm_motor.ChangeDutyCycle(0)
        
        elif emotion_string == "sad":
            # This is your "blink" logic from cara.py
            print(f"[GPIO]: Sad received -> blink eyes (Pin {JETSON_EYES_PIN})")
            GPIO.output(JETSON_EYES_PIN, GPIO.HIGH)
            time.sleep(0.3)
            GPIO.output(JETSON_EYES_PIN, GPIO.LOW)
        
        else:
            # Neutral: Make sure everything is off
            print("[GPIO]: Neutral or unknown -> do nothing")
            pwm_motor.ChangeDutyCycle(0)
            GPIO.output(JETSON_EYES_PIN, GPIO.LOW)
            
    except Exception as e:
        print(f"ERROR: Failed to write to Jetson GPIO pins: {e}")

def _jetson_cleanup():
    """
    Safely cleans up the Jetson GPIO pins on exit.
    Includes fixes for known shutdown bugs.
    """
    print("Shutting down and cleaning up Jetson GPIO pins...")
    try:
        if pwm_motor:
            pwm_motor.stop()
        GPIO.cleanup()
        print("Jetson GPIO cleanup complete.")
    except (OSError, NameError) as e:
        # Ignore known, harmless shutdown errors
        print(f"Harmless shutdown warning ignored: {e}")
    except Exception as e:
        print(f"An error occurred during Jetson GPIO cleanup: {e}")


# -----------------------------------------------------------------
# --- SECTION 2: ARDUINO NANO (SERIAL) LOGIC ---
# -----------------------------------------------------------------
# (This logic runs if USE_JETSON_GPIO is False)

# --- Arduino Config ---
# Make sure this port is correct! Run 'ls /dev/ttyUSB*'
#ARDUINO_PORT = '/dev/ttyUSB0' 
ARDUINO_PORT = '/dev/ttyAMA0'
ARDUINO_BAUD = 9600

def _arduino_initialize():
    """
    Initializes the serial connection to the Arduino.
    """
    global arduino_serial
    try:
        arduino_serial = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        # Wait for the Arduino to reset
        time.sleep(2) 
        print(f"--- SUCCESS: Using ARDUINO on {ARDUINO_PORT} ---")
    except serial.SerialException as e:
        print(f"--- ERROR: Could not connect to Arduino! ---")
        print(f"Error: {e}")
        print("1. Is the Arduino plugged in?")
        print(f"2. Is the port correct? (Is it {ARDUINO_PORT}?)")
        print("3. Did you run 'sudo modprobe ch341'?")
    except Exception as e:
        print(f"An unknown error occurred during Arduino init: {e}")


def _arduino_send_emotion(emotion_string):
    """
    Sends the correct command byte to the Arduino.
    """
    global arduino_serial
    
    if not arduino_serial or not arduino_serial.is_open:
        print("Cannot send command: Arduino is not connected.")
        return
        
    try:
        if emotion_string == "happy":
            # 'H' matches your 'arduino.ino' sketch
            print(f"[Serial]: Happy received -> sending 'H'")
            arduino_serial.write(b'H') 
        
        elif emotion_string == "sad":
            # 'B' (for Blink) matches your 'arduino.ino' sketch
            print(f"[Serial]: Sad received -> sending 'B'")
            arduino_serial.write(b'B')
            
        else:
            print("[Serial]: Neutral or unknown -> sending nothing")
            
    except Exception as e:
        print(f"ERROR: Failed to write to Arduino serial port: {e}")

def _arduino_cleanup():
    """
    Safely closes the serial port on exit.
    """
    global arduino_serial
    if arduino_serial and arduino_serial.is_open:
        print(f"Shutting down and closing Arduino port {ARDUINO_PORT}...")
        arduino_serial.close()
        print("Arduino port closed.")

# -----------------------------------------------------------------
# --- SECTION 3: PUBLIC WRAPPER FUNCTIONS ---
# -----------------------------------------------------------------
#
#  cara.py script will call THESE functions.
# They will automatically call the correct Jetson or Arduino
# function based on the 'USE_JETSON_GPIO' toggle at the top.
#

def initialize_arduino():
    """
    PUBLIC: Initializes the chosen controller.
    """
    if USE_JETSON_GPIO:
        _jetson_initialize()
    else:
        _arduino_initialize()

def send_emotion_to_arduino(emotion_string):
    """
    PUBLIC: Sends an emotion command to the chosen controller.
    """
    if USE_JETSON_GPIO:
        _jetson_send_emotion(emotion_string)
    else:
        _arduino_send_emotion(emotion_string)

def close_arduino():
    """
    PUBLIC: Cleans up the chosen controller.
    (This function is registered with atexit).
    """
    if USE_JETSON_GPIO:
        _jetson_cleanup()
    else:
        _arduino_cleanup()

# Automatic Cleanup 
# This makes sure that 'close_arduino()' is called no matter
# how your script exits, preventing pins from getting stuck.
atexit.register(close_arduino)

