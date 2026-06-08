#!/usr/bin/env python3
"""
Wake Up - Clap-Activated App Launcher
Control your computer with voice and claps!

Say a wake word to activate, then use clap patterns to launch apps.
Uses openWakeWord for fast, offline wake word detection.

GitHub: https://github.com/tpateeq/wake-up

Requirements:
    pip install openwakeword pyaudio numpy requests feedparser

openWakeWord built-in models (no API key needed):
    hey_jarvis, alexa, hey_mycroft, hey_rhasspy, ok_nabu
    Or provide a path to a custom .tflite model file.
"""

import pyaudio
import numpy as np
import subprocess
import time
import sys
import os
import platform
from collections import deque
import struct
import signal
import requests
import feedparser

try:
    from openwakeword.model import Model as OWWModel
except ImportError:
    print("❌ openwakeword not installed!")
    print("\nInstall it with:")
    print("  pip install openwakeword")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Built-in model names (no download needed):
#   "hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy", "ok_nabu"
#
# To use a custom model, set WAKE_WORD_MODEL to a path like:
#   WAKE_WORD_MODEL = "/path/to/my_model.tflite"
#
WAKE_WORD_MODEL = "hey_jarvis"

# Inference framework — use "onnx" (default); install tflite-runtime for "tflite"
INFERENCE_FRAMEWORK = "onnx"

# Confidence score (0.0 – 1.0) required to trigger the wake word
WAKE_WORD_THRESHOLD = 0.5

# openWakeWord expects 16 kHz mono audio
SAMPLE_RATE = 16000

# Frame size recommended by openWakeWord: 1280 samples (~80 ms at 16 kHz)
FRAME_LENGTH = 1280
# ---------------------------------------------------------------------------


class UnifiedLauncher:
    """Unified wake word and clap detection with single audio stream."""

    def __init__(self, wake_word_model=WAKE_WORD_MODEL,
                 wake_word_threshold=WAKE_WORD_THRESHOLD,
                 clap_threshold=1800, debug=False):

        self.wake_word_model_path = wake_word_model
        self.wake_word_threshold = wake_word_threshold
        self.clap_threshold = clap_threshold
        self.debug = debug

        # Detect operating system
        self.os_type = platform.system()   # 'Darwin', 'Windows', or 'Linux'
        print(f"🖥️  Detected OS: {self.os_type}")

        # State management
        self.is_active = False
        self.activation_time = 0
        self.active_duration = 5
        self.running = True

        # Clap detection state
        self.clap_times = []
        self.last_clap_time = 0
        self.clap_interval = 0.7
        self.previous_amplitude = 0
        self.amplitude_history = deque(maxlen=10)
        self.pending_double_since = None   # set after 2nd clap; wait for possible 3rd

        # Audio settings
        self.sample_rate = SAMPLE_RATE
        self.frame_length = FRAME_LENGTH

        # ── Initialise openWakeWord ──────────────────────────────────────────
        print("🔧 Loading openWakeWord model…")
        try:
            # wakeword_models accepts a list of built-in names or .tflite paths
            self.oww = OWWModel(
                wakeword_models=[self.wake_word_model_path],
                inference_framework="onnx"
            )
            model_label = os.path.basename(self.wake_word_model_path)
            print(f"✅ Wake word model '{model_label}' loaded successfully!")
            print("💡 Runs 100 % locally — no internet needed!\n")
        except Exception as e:
            print(f"❌ Error initialising openWakeWord: {e}")
            print("\n💡 Available built-in models:")
            print("   hey_jarvis, alexa, hey_mycroft, hey_rhasspy, ok_nabu")
            print("💡 Or pass a path to a custom .tflite model file.")
            sys.exit(1)

        # PyAudio
        self.pa = pyaudio.PyAudio()
        self.audio_stream = None

        # Clean-exit handler
        signal.signal(signal.SIGINT, self.signal_handler)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n👋 Shutting down…")
        self.running = False

    def start_audio_stream(self):
        """Start the unified audio stream."""
        try:
            self.audio_stream = self.pa.open(
                rate=self.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.frame_length
            )
            model_label = os.path.basename(self.wake_word_model_path)
            print(f"🎧 Listening for wake word '{model_label}'…")
            print("💡 Say the wake word to start clap detection\n")
        except Exception as e:
            print(f"❌ Error opening audio stream: {e}")
            sys.exit(1)

    def cleanup(self):
        """Release all resources."""
        print("Cleaning up…")
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.pa:
            self.pa.terminate()
        print("Goodbye!")

    # ── Wake-word detection ──────────────────────────────────────────────────

    def detect_wake_word(self, pcm_tuple):
        """
        Run openWakeWord inference on one audio frame.

        pcm_tuple : tuple of int16 samples (length == FRAME_LENGTH)
        Returns   : True if wake word detected above threshold.
        """
        try:
            # openWakeWord wants a flat numpy int16 array
            audio_array = np.array(pcm_tuple, dtype=np.int16)

            # predict() updates internal scores and returns a dict
            prediction = self.oww.predict(audio_array)

            # Check every model's score against the threshold
            for model_name, score in prediction.items():
                if self.debug:
                    print(f"[OWW] {model_name}: {score:.3f}")
                if score >= self.wake_word_threshold:
                    # Reset scores so the same detection doesn't fire twice
                    self.oww.reset()
                    return True
            return False
        except Exception as e:
            if self.debug:
                print(f"Wake word error: {e}")
            return False

    # ── Clap detection ───────────────────────────────────────────────────────

    def detect_clap(self, pcm):
        """
        Detect clap patterns from audio data.
        Returns 0 (nothing yet), 2 (double clap), or 3 (triple clap).

        Strategy: never confirm a double clap immediately — wait
        TRIPLE_WAIT seconds after the 2nd clap to see if a 3rd arrives.
        """
        TRIPLE_WAIT = 0.6   # seconds to wait after 2nd clap before confirming double

        try:
            audio_data = np.array(pcm, dtype=np.int16)
            amplitude = np.abs(audio_data).max()

            self.amplitude_history.append(amplitude)
            current_time = time.time()

            if self.debug and amplitude > 500:
                print(f"Amplitude: {amplitude} (threshold: {self.clap_threshold})")

            amplitude_jump = amplitude - self.previous_amplitude
            sharp_attack = amplitude_jump > (self.clap_threshold * 0.4)
            loud_enough = amplitude > self.clap_threshold

            if len(self.amplitude_history) >= 3:
                avg_recent = sum(self.amplitude_history) / len(self.amplitude_history)
                not_sustained = avg_recent < (self.clap_threshold * 0.5)
            else:
                not_sustained = True

            is_clap = loud_enough and (sharp_attack or not_sustained)

            if is_clap and current_time - self.last_clap_time > 0.1:
                self.clap_times.append(current_time)
                self.last_clap_time = current_time
                print(f"👏 Clap #{len(self.clap_times)} detected!")

                # Drop claps outside the rolling window
                self.clap_times = [
                    t for t in self.clap_times
                    if current_time - t < self.clap_interval * 3
                ]

                # ── Triple clap: 3 claps within window → fire immediately ──
                if len(self.clap_times) >= 3:
                    time_span = self.clap_times[-1] - self.clap_times[-3]
                    if time_span < self.clap_interval * 2.5:
                        self.clap_times.clear()
                        self.pending_double_since = None
                        return 3

                # ── After 2nd clap: start the wait window ──
                if len(self.clap_times) >= 2:
                    time_span = self.clap_times[-1] - self.clap_times[-2]
                    if time_span < self.clap_interval:
                        self.pending_double_since = current_time
                        print("⏳ 2 claps — waiting briefly for a 3rd…")

            self.previous_amplitude = amplitude

            # ── Check if the double-clap wait window has expired ──
            if getattr(self, "pending_double_since", None):
                waited = current_time - self.pending_double_since
                if waited >= TRIPLE_WAIT:
                    self.pending_double_since = None
                    self.clap_times.clear()
                    return 2

            # Expire stale clap history
            if self.clap_times and current_time - self.clap_times[-1] > self.clap_interval * 2:
                self.clap_times.clear()
                self.pending_double_since = None

            return 0

        except Exception as e:
            if self.debug:
                print(f"Clap detection error: {e}")
            return 0

    # ── Activation helpers ───────────────────────────────────────────────────

    def activate(self):
        """Activate clap listening mode."""
        self.is_active = True
        self.activation_time = time.time()
        print("\n" + "=" * 60)
        print("✨ WAKE WORD DETECTED! Listening for claps…")
        print("👏👏  Double clap = Launch Dev workspace")
        print("👏👏👏 Triple clap = Launch Trading workspace")
        print(f"⏱️  You have {self.active_duration} seconds…")
        print("=" * 60 + "\n")

    def deactivate(self):
        """Deactivate clap listening mode."""
        self.is_active = False
        print("\n⏰ Time's up! Say the wake word to try again.\n")

    def is_still_active(self):
        """Return True if still inside the active listening window."""
        if not self.is_active:
            return False
        if time.time() - self.activation_time > self.active_duration:
            self.deactivate()
            return False
        return True

    # ── Briefing / helpers ───────────────────────────────────────────────────

    def briefing(self):
        weather = self.get_weather()
        news = self.get_news()
        message = (
            "Welcome Mayank. "
            "Good afternoon. "
            f"{weather} "
            f"Top news today: {news} "
            "Opening your development workspace."
        )
        self.speak(message)

    def get_weather(self):
        try:
            r = requests.get("https://wttr.in/Mumbai?format=j1", timeout=5)
            data = r.json()
            temp = data["current_condition"][0]["temp_C"]
            desc = data["current_condition"][0]["weatherDesc"][0]["value"]
            return f"Temperature is {temp} degrees. Weather is {desc}."
        except Exception:
            return "Weather data is currently unavailable."

    def get_news(self):
        try:
            feed = feedparser.parse("https://news.google.com/rss")
            return feed.entries[0].title
        except Exception:
            return "Unable to retrieve the latest news."

    def speak(self, text):
        """Text-to-speech via the system 'say' command (macOS)."""
        subprocess.Popen(["say", "-v", "Samantha", text])

    # ── OS-level launchers ───────────────────────────────────────────────────

    def _launch_app_macos(self, app_name, path=None, args=None):
        cmd = ["open", "-a", app_name]
        if path:
            cmd.append(path)
        if args:
            cmd.extend(["--args"] + args)
        subprocess.Popen(cmd)

    def _launch_app_windows(self, app_command, args=None):
        if args:
            subprocess.Popen([app_command] + args, shell=True)
        else:
            subprocess.Popen(["start", app_command], shell=True)

    def _launch_app_linux(self, app_command, args=None):
        if args:
            subprocess.Popen([app_command] + args)
        else:
            subprocess.Popen([app_command])

    # ── Workspace launchers ──────────────────────────────────────────────────

    def launch_dev_workspace(self):
        print("\n🚀 DOUBLE CLAP → Developer Workspace\n")
        self.briefing()

        urls = [
            "https://finnhub.io/",
            "https://supabase.com/dashboard/project/vzzlwpejsdrvpjjizzfd/database/schemas",
            "https://www.marketaux.com/account/dashboard",
            "http://localhost:8080/dashboard",
            "https://claude.ai",
        ]

        if self.os_type == "Windows":
            subprocess.Popen(["start", "cursor"], shell=True)
            time.sleep(0.5)
            subprocess.Popen(["start", "chrome"] + urls, shell=True)
        elif self.os_type == "Darwin":
            subprocess.Popen(["open", "-a", "Cursor"])
            subprocess.Popen(["open", "-a", "Google Chrome"] + urls)
        elif self.os_type == "Linux":
            subprocess.Popen(["cursor"])
            subprocess.Popen(["google-chrome"] + urls)

        print("✅ Developer workspace ready!\n")

    def launch_trading_workspace(self):
        print("\n📈 TRIPLE CLAP → Trading Workspace\n")

        urls = [
            "https://docs.google.com/spreadsheets/d/1yRgZS4IMvrdsMzAtBHhBKfA7uXSpUX5Orr_4yfDpXJc/edit?pli=1&gid=0#gid=0",
            "https://www.forexfactory.com/",
            "https://www.tradingview.com/chart/?symbol=NASDAQ%3AAAPL",
        ]

        if self.os_type == "Windows":
            subprocess.Popen(["start", "chrome"] + urls, shell=True)
        elif self.os_type == "Darwin":
            subprocess.Popen(["open", "-a", "Google Chrome"] + urls)
        elif self.os_type == "Linux":
            subprocess.Popen(["google-chrome"] + urls)

        print("✅ Trading workspace ready!\n")

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        """Main run loop."""
        self.start_audio_stream()

        try:
            while self.running:
                # Read one audio frame
                pcm_bytes = self.audio_stream.read(
                    self.frame_length, exception_on_overflow=False
                )
                pcm = struct.unpack_from("h" * self.frame_length, pcm_bytes)

                if not self.is_active:
                    # Listen for wake word
                    if self.detect_wake_word(pcm):
                        self.activate()

                elif self.is_still_active():
                    # Listen for clap pattern
                    clap_type = self.detect_clap(pcm)

                    if clap_type == 2:
                        self.launch_dev_workspace()
                        self.deactivate()
                    elif clap_type == 3:
                        self.launch_trading_workspace()
                        self.deactivate()

        except KeyboardInterrupt:
            print("\n\n👋 Shutting down…")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  👏 WAKE UP - Clap Launcher  (openWakeWord edition)")
    print("=" * 70)
    print("\n🚀 100 % LOCAL — No API key needed!")
    print("🗣️  Say wake word → 👏👏 Double clap  → Dev workspace")
    print("🗣️  Say wake word → 👏👏👏 Triple clap → Trading workspace")
    print("\nPress Ctrl+C to exit\n")

    debug_mode = "--debug" in sys.argv

    # Allow overriding the model from the CLI:  --model hey_mycroft
    model = WAKE_WORD_MODEL
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]

    if not debug_mode:
        print("💡 Tip: run with '--debug' to see amplitude & OWW confidence")
        print("💡 Tip: run with '--model hey_mycroft' to change wake word")
        print("💡 Built-in models: hey_jarvis | alexa | hey_mycroft | hey_rhasspy | ok_nabu\n")

    launcher = UnifiedLauncher(
        wake_word_model=model,
        wake_word_threshold=WAKE_WORD_THRESHOLD,
        clap_threshold=1800,
        debug=debug_mode,
    )
    launcher.run()


if __name__ == "__main__":
    main()