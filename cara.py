import os
import json
import glob
import atexit
import time
from datetime import datetime
from dotenv import load_dotenv

# --- AI & Audio Imports ---
import google.generativeai as genai
from cara_audio_io import recognize_from_microphone, speak_text, get_recognition_stats

# --- Hardware & Emotion Imports ---
from sentiment_detection import detect_emotion, react_to_emotion
from serial_control import initialize_arduino, send_emotion_to_arduino
# Assuming HeadController is in a file named head_controller.py
from head_controller import HeadController 

# --- Memory Imports ---
from cara_summarizer import summarize_conversation_for_memory

# 1. SETUP ENVIRONMENT
load_dotenv()
print("SUCCESS: Libraries loaded.")

# 2. INITIALIZE HARDWARE
initialize_arduino()
head_controller = HeadController()
# Start head controller thread if it requires background processing
if hasattr(head_controller, 'start'):
    head_controller.start()

# 3. CONFIGURE GEMINI
gemini_api_key = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

# 4. LOAD BASE PROMPT
with open("cara_prompt.json", "r") as f:
    prompt_config = json.load(f)
base_system_instruction = prompt_config["system_instruction"]

# 5. MEMORY RETRIEVAL FUNCTIONS (RAG)
import random # Add this to imports

# ... (Previous imports)

def get_memory_context():
    """
    Reads Profile, Recent Sessions, and a 'Handful of Good Things'.
    Returns a string to inject into the system prompt.
    """
    context_str = "\n\n[MEMORY SYSTEM ACTIVATED]\n"
    
    # --- A. Load Profile (Long-term) ---
    try:
        with open("cara_memory/profile.json", "r") as f:
            profile = json.load(f)
            context_str += f"USER PROFILE:\n{json.dumps(profile, indent=2)}\n\n"
    except FileNotFoundError:
        context_str += "USER PROFILE: Not found.\n\n"

    # --- B. Load Recent Sessions (Medium-term) ---
    session_files = glob.glob("cara_memory/sessions/*.json")
    session_files.sort(reverse=True)
    recent_files = session_files[:3] # Last 3 conversations
    
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

    # --- C. Load 'Good Things' (The Jar of Joy) ---
    # We pick 3 random positive memories to keep 'top of mind'
    try:
        with open("cara_memory/good_things.json", "r") as f:
            good_things = json.load(f)
            
        if good_things:
            # Pick up to 3 random items
            selected_joy = random.sample(good_things, min(len(good_things), 3))
            
            context_str += "\nHAPPY MEMORIES BANK (Use these to cheer Elida up if she is sad):\n"
            for item in selected_joy:
                context_str += f"- On {item['date']}: {item['memory']}\n"
    except FileNotFoundError:
        print("No good_things.json found.")

    return context_str
def save_happy_memory(user_text):
    """
    Extracts the memory from the user's input and saves it to good_things.json.
    """
    # 1. Clean up the trigger phrase to get just the memory
    # Triggers: "remember that", "write down that", "add to memory"
    triggers = ["remember that", "write down that", "save this"]
    
    memory_content = user_text
    for t in triggers:
        if t in user_text.lower():
            # Split the string at the trigger and take the second part
            parts = user_text.lower().split(t, 1)
            if len(parts) > 1:
                memory_content = parts[1].strip()
                # Restore original casing overlap (simple method) or just capitalize
                memory_content = memory_content.capitalize()
            break
            
    # 2. Prepare the entry
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "memory": memory_content
    }

    # 3. Load, Append, Save
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
        
# 6. INITIALIZE MODEL WITH DYNAMIC PROMPT
# We perform the retrieval NOW to build the initial state.
full_system_instruction = base_system_instruction + get_memory_context()

model = genai.GenerativeModel(
    'models/gemini-2.0-flash-001',
    system_instruction=full_system_instruction
)

# 7. CONVERSATION STATE
# We keep the raw log for the summarizer later
conversation_log = [] 
# We keep the chat history for Gemini context window
chat_session = model.start_chat(history=[])

def main():
    print("\n Cara is Waking Up...")
    print("-----------------------------------")
    speak_text("I'm here, and my ears are ready to listen.")

    while True:
        # --- A. LISTEN ---
        user_input = recognize_from_microphone()

        if not user_input:
            continue

        print(f"\nElida: {user_input}")
        
        # Log for summarizer
        conversation_log.append({"role": "user", "content": user_input})

        # --- B. CHECK EXIT ---
        if "goodbye" in user_input.lower() or "bye cara" in user_input.lower():
            farewell = "Goodbye Elida! I'll be dreaming of bees until you return."
            print(f"Cara: {farewell}")
            speak_text(farewell)
            conversation_log.append({"role": "assistant", "content": farewell})
            break
        # --- NEW: CHECK FOR MEMORY SAVE REQUEST ---
        # If user explicitly asks to remember something good
        if "remember that" in user_input.lower() or "save this" in user_input.lower():
            success, saved_mem = save_happy_memory(user_input)
            if success:
                print(f" >> Joy Jar Updated: {saved_mem}")
                # We modify the prompt slightly so Cara knows she succeeded
                # This is a "System Note" injected into the conversation flow
                user_input += f"\n[SYSTEM: You successfully saved this memory to your permanent 'Joy Jar'. Confirm this to Elida warmly.]"
        # --- C. GENERATE RESPONSE ---
        try:
            response = chat_session.send_message(user_input)
            cara_reply = response.text.strip()
        except Exception as e:
            print(f"Gemini Error: {e}")
            cara_reply = "Oh dear, my thoughts got a little tangled. Can you say that again?"

        print(f"Cara: {cara_reply}")
        conversation_log.append({"role": "assistant", "content": cara_reply})

        # --- D. PARALLEL ACTIONS (Speak, Move, Emotion) ---
        # 1. Speak
        speak_text(cara_reply)

        # 2. Detect Emotion from Cara's own words (How does SHE feel?)
        emotion = detect_emotion(cara_reply) 
        # (Or detect from User input if you prefer: detect_emotion(user_input))
        print(f"[Internal Emotion]: {emotion}")

        # 3. Hardware Reaction
        # Send to Arduino (Assuming 'H' for happy, 'B' for sad/blink based on your Arduino code)
        # We need to map the full word "happy" to the char 'H'
        arduino_char = 'N' # Neutral default
        if "happy" in emotion or "joy" in emotion:
            arduino_char = 'H'
        elif "sad" in emotion or "concern" in emotion:
            arduino_char = 'B' # Blinking for sad/empathetic
        
        send_emotion_to_arduino(arduino_char) 
        
        # 4. Head Controller
        # Assuming express_emotion takes the string
        head_controller.express_emotion(emotion) 
        
        # 5. Body Movement (from sentiment_detection)
        react_to_emotion(emotion)

    # --- E. CLEANUP & MEMORY SAVE ---
    print("\n[Saving Memory...]")
    try:
        # This calls your cara_summarizer.py to generate a JSON summary
        # and save it to cara_memory/sessions/
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
        # Ensure hardware shuts down safely
        print("Shutting down hardware threads...")
        head_controller._run_thread = False 
        # Add any Arduino close commands if needed
