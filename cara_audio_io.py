from elevenlabs import ElevenLabs, save
import os
import speech_recognition as sr
import subprocess
import json
import time
from pathlib import Path

class AdaptiveSpeechRecognizer:
    """
    Self-calibrating speech recognizer that adapts to environment and hardware.
    Learns from successful recognitions and adjusts settings automatically.
    """
    
    def __init__(self, config_file="speech_config.json"):
        self.config_file = config_file
        self.recognizer = sr.Recognizer()
        self.config = self.load_or_create_config()
        self.apply_config()
        
        # Track performance for adaptive learning
        self.recent_successes = []
        self.recent_failures = []
        
    def load_or_create_config(self):
        """Load existing config or create with safe defaults"""
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                print(f"Loaded existing speech config (success rate: {config.get('success_rate', 0):.1%})")
                return config
        
        # Safe initial defaults - conservative to avoid false triggers
        return {
            "energy_threshold": None,  # Will auto-calibrate
            "dynamic_energy_threshold": True,
            "dynamic_energy_adjustment_damping": 0.15,
            "dynamic_energy_ratio": 1.5,
            "pause_threshold": 0.8,
            "phrase_threshold": 0.3,
            "non_speaking_duration": 0.5,
            "calibration_duration": 1.0,
            "success_count": 0,
            "total_attempts": 0,
            "success_rate": 0.0,
            "last_calibration": None
        }
    
    def save_config(self):
        """Save current configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def apply_config(self):
        """Apply configuration to recognizer"""
        r = self.recognizer
        r.dynamic_energy_threshold = self.config["dynamic_energy_threshold"]
        r.dynamic_energy_adjustment_damping = self.config["dynamic_energy_adjustment_damping"]
        r.dynamic_energy_ratio = self.config["dynamic_energy_ratio"]
        r.pause_threshold = self.config["pause_threshold"]
        r.phrase_threshold = self.config["phrase_threshold"]
        r.non_speaking_duration = self.config["non_speaking_duration"]
        
        if self.config["energy_threshold"] is not None:
            r.energy_threshold = self.config["energy_threshold"]
    
    def should_recalibrate(self):
        """Determine if recalibration is needed"""
        if self.config["last_calibration"] is None:
            return True
        
        # Recalibrate if success rate drops below 70%
        if self.config["total_attempts"] > 5 and self.config["success_rate"] < 0.7:
            print("Success rate low - recalibrating...")
            return True
        
        # Recalibrate every 50 attempts to adapt to gradual changes
        if self.config["total_attempts"] % 50 == 0 and self.config["total_attempts"] > 0:
            print("Periodic recalibration...")
            return True
        
        return False
    
    def calibrate(self, source):
        """Smart calibration that adapts to current environment"""
        print("\nCalibrating to your environment...")
        print("   (Please remain quiet for a moment)")
        
        # Initial calibration
        self.recognizer.adjust_for_ambient_noise(
            source, 
            duration=self.config["calibration_duration"]
        )
        
        base_energy = self.recognizer.energy_threshold
        print(f" Base noise level: {base_energy:.0f}")
        
        # Add adaptive margin based on past performance
        if self.config["success_rate"] < 0.6 and self.config["total_attempts"] > 3:
            # If struggling, be more sensitive
            adjusted_energy = base_energy * 0.9
            print("Increasing sensitivity (low success rate)")
        elif self.config["success_rate"] > 0.9 and self.config["total_attempts"] > 10:
            # If doing great, can be slightly less sensitive to reduce false triggers
            adjusted_energy = base_energy * 1.1
            print(" Slightly reducing sensitivity (high success rate)")
        else:
            adjusted_energy = base_energy
        
        self.config["energy_threshold"] = adjusted_energy
        self.config["last_calibration"] = time.time()
        self.recognizer.energy_threshold = adjusted_energy
        
        print(f"   ✓ Calibrated! Threshold: {adjusted_energy:.0f}")
        self.save_config()
    
    def record_success(self, text_length):
        """Record successful recognition for learning"""
        self.config["success_count"] += 1
        self.config["total_attempts"] += 1
        self.config["success_rate"] = self.config["success_count"] / self.config["total_attempts"]
        
        self.recent_successes.append({
            "timestamp": time.time(),
            "text_length": text_length,
            "energy": self.recognizer.energy_threshold
        })
        
        # Keep only last 20 results
        if len(self.recent_successes) > 20:
            self.recent_successes.pop(0)
        
        self.save_config()
    
    def record_failure(self, failure_type):
        """Record recognition failure for learning"""
        self.config["total_attempts"] += 1
        if self.config["total_attempts"] > 0:
            self.config["success_rate"] = self.config["success_count"] / self.config["total_attempts"]
        
        self.recent_failures.append({
            "timestamp": time.time(),
            "type": failure_type,
            "energy": self.recognizer.energy_threshold
        })
        
        if len(self.recent_failures) > 20:
            self.recent_failures.pop(0)
        
        # Adapt if we're seeing patterns
        if len(self.recent_failures) >= 3:
            recent = self.recent_failures[-3:]
            failure_types = [f["type"] for f in recent]
            
            # If multiple timeouts, might be too sensitive or user speaking quietly
            if failure_types.count("timeout") >= 2:
                print(" Detecting silence issues - increasing sensitivity next time")
            
            # If multiple unknown values, might be audio quality or threshold issue
            if failure_types.count("unknown") >= 2:
                print(" Detecting clarity issues - will recalibrate")
        
        self.save_config()
    
    def recognize_from_microphone(self, device_index=0):
        """Adaptive speech recognition with auto-calibration"""
        r = self.recognizer
        
        with sr.Microphone(device_index=device_index, sample_rate=16000, chunk_size=1024) as source:
            # Recalibrate if needed
            if self.should_recalibrate():
                self.calibrate(source)
            else:
                # Quick ambient adjustment (doesn't change saved threshold)
                r.adjust_for_ambient_noise(source, duration=0.3)
            
            print("\n🎧 Cara is listening... (speak naturally)")
            
            try:
                # Adaptive timeouts based on past performance
                timeout = 10
                phrase_limit = 20
                
                audio = r.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit
                )
                
            except sr.WaitTimeoutError:
                print(" Silence detected - no speech heard")
                self.record_failure("timeout")
                return ""
        
        try:
            print(" Processing...")
            
            recognized_text = r.recognize_google(
                audio,
                language="en-US",
                show_all=False
            )
            
            print(f"✓ Elida said: {recognized_text}")
            self.record_success(len(recognized_text))
            return recognized_text
            
        except sr.UnknownValueError:
            print(" Audio received but couldn't understand - try speaking more clearly")
            self.record_failure("unknown")
            return ""
        except sr.RequestError as e:
            print(f" Network error: {e}")
            self.record_failure("network")
            return ""
    
    def get_stats(self):
        """Return current performance statistics"""
        return {
            "success_rate": f"{self.config['success_rate']:.1%}",
            "total_attempts": self.config["total_attempts"],
            "current_threshold": self.recognizer.energy_threshold,
            "recent_successes": len(self.recent_successes),
            "recent_failures": len(self.recent_failures)
        }


# Global instance for easy use
_adaptive_recognizer = None

def recognize_from_microphone(device_index=0):
    """
    Main function - uses adaptive recognizer that learns and improves over time.
    First call will calibrate, subsequent calls will use learned settings.
    """
    global _adaptive_recognizer
    
    if _adaptive_recognizer is None:
        _adaptive_recognizer = AdaptiveSpeechRecognizer()
    
    return _adaptive_recognizer.recognize_from_microphone(device_index)


def get_recognition_stats():
    """Get current performance statistics"""
    if _adaptive_recognizer:
        return _adaptive_recognizer.get_stats()
    return None


def reset_calibration():
    """Force recalibration on next recognition attempt"""
    if _adaptive_recognizer:
        _adaptive_recognizer.config["last_calibration"] = None
        _adaptive_recognizer.save_config()
        print("Calibration reset - will recalibrate on next use")


def speak_text(text):
    """Generate and play speech - optimized version"""
    print(" Cara is speaking...")

    try:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            print("[ERROR] ELEVENLABS_API_KEY not found.")
            return

        client = ElevenLabs(api_key=api_key)

        audio = client.text_to_speech.convert(
            text=text,
            voice_id="H8BjWxFjrzNszTO74noq",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )

        # Save and convert
        mp3_path = "temp_cara_speech.mp3"
        save(audio, mp3_path)

        wav_path = "temp_cara_speech.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "44100", "-ac", "1", wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Play through USB speaker
        subprocess.run(
            ["aplay", "-D", "plughw:1,0", wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Clean up
        for f in [mp3_path, wav_path]:
            if os.path.exists(f):
                os.remove(f)

    except Exception as e:
        print(f"Speech error: {e}")


# # Example usage:
# if __name__ == "__main__":
#     print("=== Adaptive Speech Recognition Test ===\n")
    
#     # First use will calibrate
#     text = recognize_from_microphone()
    
#     if text:
#         speak_text(f"Elida said: {text}")
    
#     # Check stats
#     stats = get_recognition_stats()
#     if stats:
#         print(f"\n Stats: {stats}")
