import os
import json
import glob
import atexit
import time
import threading  # NEW: For background ROS
from datetime import datetime
from dotenv import load_dotenv

# ROS 2 Imports (NEW) 
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

#AI & Audio Imports 
import google.generativeai as genai
from cara_audio_io import recognize_from_microphone, speak_text, get_recognition_stats

# Hardware & Emotion Imports 
from sentiment_detection import detect_emotion, react_to_emotion
from serial_control import initialize_arduino, send_emotion_to_arduino
# Assuming HeadController is in a file named head_controller.py
from head_controller import HeadController 

# Memory Imports 
from cara_memory.cara_summarizer import summarize_conversation_for_memory
import random 

# ============================================================================
# 0. ROS 2 BACKGROUND LISTENER (The Eyes)
# ============================================================================

# Global variable to hold the latest emotion from the camera
# The ROS thread writes to this, the Main thread reads from it.
current_visual_emotion = "neutral (0.0)"

class VisualEmotionListener(Node):
    def __init__(self):
        super().__init__('cara_brain_listener')
        # Subscribe to the node we built earlier
        self.sub = self.create_subscription(
            String, 
            '/cara/emotion', 
            self.emotion_callback, 
            10
        )
        # Publisher to trigger training (optional)
        self.pub_train = self.create_publisher(String, '/cara/feedback', 10)

    def emotion_callback(self, msg):
        global current_visual_emotion
        # Msg format is usually "happy (0.85)"
        current_visual_emotion = msg.data

def start_ros_background_thread():
    """Starts ROS 2 in a separate thread so it doesn't block audio/AI"""
    rclpy.init(args=None)
    ros_node = VisualEmotionListener()
    
    # Create a thread that keeps ROS spinning
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(ros_node)
    
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    
    return ros_node, executor

# ============================================================================
# MAIN SETUP
# ============================================================================

# 1. SETUP ENVIRONMENT
load_dotenv()
print("SUCCESS: Libraries loaded.")

# 2. START ROS LISTENER
print("Connecting to Visual System...")
ros_node, ros_executor = start_ros_background_thread()

# 3. INITIALIZE HARDWARE
initialize_arduino()
head_controller = HeadController()
if hasattr(head_controller, 'start'):
    head_controller.start()

# 4. CONFIGURE GEMINI
gemini_api_key = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

# 5. LOAD BASE PROMPT
with open("cara_prompt.json", "r") as f:
    prompt_config = json.load(f)
base_system_instruction = prompt_config["system_instruction"]

# 6. MEMORY FUNCTIONS
def get_memory_context():
    """Reads Profile, Recent Sessions, and 'Good Things'."""
    context_str = "\n\n[MEMORY SYSTEM ACTIVATED]\n"
    
    # A. Profile
    try:
        with open("cara_memory/profile.json", "r") as f:
            profile = json.load(f)
            context_str += f"USER PROFILE:\n{json.dumps(profile, indent=2)}\n\n"
    except FileNotFoundError:
        context_str += "USER PROFILE: Not found.\n\n"

    # B. Recent Sessions
    session_files = glob.glob("cara_memory/sessions/*.json")
    session_files.sort(reverse=True)
    recent_files = session_files[:3] 
    
    context_str += "RECENT CONVERSATIONS:\n"
    if not recent_files:
        context_str += "No previous session notes found.\n"
    
    for file_path in recent_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                summary_snippet = {
                    "date": data.get("timestamp"),
                    "summary": data.get("summary"),
                    "user_emotion": data.get("emotion")
                }
                context_str += f"- {json.dumps(summary_snippet)}\n"
        except Exception as e:
            print(f"Error reading session file: {e}")

    # C. Good Things
    try:
        with open("cara_memory/good_things.json", "r") as f:
            good_things = json.load(f)
            if good_things:
                selected_joy = random.sample(good_things, min(len(good_things), 3))
                context_str += "\nHAPPY MEMORIES BANK:\n"
                for item in selected_joy:
                    context_str += f"- On {item['date']}: {item['memory']}\n"
    except FileNotFoundError:
        print("No good_things.json found.")

    return context_str

def save_happy_memory(user_text):
    triggers = ["remember that", "write down that", "save this", "that was so amazing", "this deserves a happy dance","I am so happy"]
    memory_content = user_text
    for t in triggers:
        if t in user_text.lower():
            parts = user_text.lower().split(t, 1)
            if len(parts) > 1:
                memory_content = parts[1].strip().capitalize()
            break
            
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "memory": memory_content
    }

    file_path = "cara_memory/good_things.json"
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        return True, memory_content
    except Exception as e:
        print(f"Error saving memory: {e}")
        return False, None

# 7. INITIALIZE MODEL
full_system_instruction = base_system_instruction + get_memory_context()
model = genai.GenerativeModel(
    'models/gemini-2.0-flash-001',
    system_instruction=full_system_instruction
)

conversation_log = [] 
chat_session = model.start_chat(history=[])

def main():
    print("\n Cara is Waking Up...")
    print("-----------------------------------")
    speak_text("I'm here, and I can see you now.")

    while True:
        # --- A. LISTEN ---
        user_input = recognize_from_microphone()

        if not user_input:
            continue

        print(f"\nElida: {user_input}")
        conversation_log.append({"role": "user", "content": user_input})

        # --- B. CHECK EXIT ---
        if "goodbye" in user_input.lower() or "bye cara" in user_input.lower():
            farewell = "Goodbye Elida! I'll be dreaming of bees until you return."
            print(f"Cara: {farewell}")
            speak_text(farewell)
            conversation_log.append({"role": "assistant", "content": farewell})
            break

        #  C. CHECK MEMORY SAVE  
        if "remember that" in user_input.lower() or "save this" in user_input.lower():
            success, saved_mem = save_happy_memory(user_input)
            if success:
                print(f" >> Joy Jar Updated: {saved_mem}")
                user_input += f"\n[SYSTEM: You successfully saved this memory to your permanent 'Joy Jar'. Confirm this to Elida warmly.]"

        #   D. INJECT VISUAL CONTEXT (THE MAGIC PART)  
        # We grab the variable that the background thread is updating
        visual_context = f"[System Note: Visual sensors detect the user looks {current_visual_emotion}]"
        
        # We combine audio input + visual context for Gemini
        combined_prompt = f"{user_input} \n{visual_context}"
        print(f" >> Context injected: {current_visual_emotion}")

        #  E. GENERATE RESPONSE  
        try:
            response = chat_session.send_message(combined_prompt)
            cara_reply = response.text.strip()
        except Exception as e:
            print(f"Gemini Error: {e}")
            cara_reply = "Oh dear, my thoughts got a little tangled."

        print(f"Cara: {cara_reply}")
        conversation_log.append({"role": "assistant", "content": cara_reply})

        # --- F. ACTIONS ---
        speak_text(cara_reply)
        
        # Detect emotion from text (How SHE feels about what she said)
        text_emotion = detect_emotion(cara_reply) 
        print(f"[Internal Emotion]: {text_emotion}")

        # Hardware Reaction
        arduino_char = 'N'
        if "happy" in text_emotion or "joy" in text_emotion:
            arduino_char = 'H'
        elif "sad" in text_emotion or "concern" in text_emotion:
            arduino_char = 'B'
        
        send_emotion_to_arduino(arduino_char) 
        
        # NOTE: Ensure your ROS head controller isn't fighting this one!
        head_controller.express_emotion(text_emotion) 
        react_to_emotion(text_emotion)

    #  CLEANUP 
    print("\n[Saving Memory...]")
    try:
        saved_path = summarize_conversation_for_memory(conversation_log)
        print(f"Session saved to: {saved_path}")
    except Exception as e:
        print(f"Failed to save memory: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nForce stopping...")
    finally:
        print("Shutting down hardware threads...")
        head_controller._run_thread = False 
        # Shut down ROS cleanly
        rclpy.shutdown()
