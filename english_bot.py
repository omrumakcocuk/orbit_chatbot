import asyncio
import collections
import os
import queue
import random
import re
import sys
import time
import warnings

import numpy as np
import pyaudio
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from google import genai
from google.genai import types
from kokoro_onnx import Kokoro
from piper import PiperVoice
from piper.config import SynthesisConfig

try:
    import sherpa_onnx
except ModuleNotFoundError:
    sherpa_onnx = None

load_dotenv()
warnings.filterwarnings("ignore")

TEXT_MODE_ONLY = "--text-mode" in sys.argv

def _parse_device_index(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value

TOTAL_LOGICAL_CORES = os.cpu_count() or 4
WHISPER_THREADS = TOTAL_LOGICAL_CORES

INPUT_SAMPLE_RATE = 16000
WHISPER_SAMPLE_RATE = 16000
CHANNELS = 2
MIC_GAIN = float(os.environ.get("MIC_GAIN", "25.0"))
MIC_FORMAT = pyaudio.paInt16
MIC_CHUNK = 1024
SPEECH_THRESHOLD = 400
SPEECH_THRESHOLD_FACTOR = 1.55
CALIBRATION_DURATION = 0.3
SILENCE_DURATION = 0.6
WAIT_FOR_SPEECH_TIMEOUT = 10.0  
RECORDING_SAFETY_TIMEOUT = 60.0
BLOCK_DURATION = 0.05
MIN_SPEECH_DURATION = 0.15
MIN_RECORDING_AFTER_SPEECH_START = 0.20
INPUT_DEVICE = _parse_device_index(os.environ.get("CHATBOT_INPUT_DEVICE"))
OUTPUT_DEVICE = _parse_device_index(os.environ.get("CHATBOT_OUTPUT_DEVICE"))
SELECTED_INPUT_DEVICE = INPUT_DEVICE
SELECTED_OUTPUT_DEVICE = OUTPUT_DEVICE
MIC_START_THRESHOLD = float(SPEECH_THRESHOLD)
MIC_STOP_THRESHOLD = float(SPEECH_THRESHOLD) * 0.8

KOKORO_MODEL = "voices/kokoro-v1.0.onnx"
KOKORO_VOICES = "voices/voices-v1.0.bin"
TTS_ENGINE = os.environ.get("TTS_ENGINE", "piper").strip().lower()
kokoro_tts = None
piper_tts = None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = None

GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"
STT_ENGINE = os.environ.get("STT_ENGINE", "sherpa").strip().lower()
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny.en")
WHISPER_TRANSCRIBE_OPTIONS = {
    "beam_size": 1,
    "best_of": 1,
    "language": "en",
    "vad_filter": False,
    "without_timestamps": True,
    "condition_on_previous_text": False,
    "temperature": 0.0,
}
SHERPA_SAMPLE_RATE = WHISPER_SAMPLE_RATE
SHERPA_TOKENS = os.environ.get("SHERPA_TOKENS", "").strip()
SHERPA_PARAFORMER = os.environ.get("SHERPA_PARAFORMER", "").strip()

LLM_MAX_TOKENS = 96
LLM_TEMPERATURE = 0.3
MAX_HISTORY_MESSAGES = 4
FIRST_TTS_WORDS = 5
NEXT_TTS_WORDS = 10
TTS_SPEECH_SPEED = 1.3

perf_metrics = {}
conversation_history = []
user_has_spoken = True
EXIT_COMMANDS = {
    "bye", "goodbye", "good bye", "bye bye", "see you", "exit", "stop", "quit"
}

# --- RASTGELE SESLİ ASİSTAN SEÇİMİ ---
# Program her açıldığında aşağıdaki asistanlardan biri seçilir.
# Seçilen asistan, konuşma boyunca değişmeden kullanılır.
VOICE_OPTIONS = [
    {
        "id": "am_adam",
        "name": "Bryce",
        "piper_model": "voices/en_US-bryce-medium.onnx",
        "piper_config": "voices/en_US-bryce-medium.onnx.json",
        "intro": (
            "Hello, I'm Bryce. I'm ready. "
            "What would you like to talk about today?"
        ),
        "farewell": (
            "Goodbye! It was great talking to you. Have a wonderful day!"
        ),
        "system_prompt": (
            "You are Bryce, a friendly, natural and confident American male "
            "English conversation partner. Always reply in English. "
            "Answer in one or two short, complete, conversational sentences. "
            "Never stop mid-sentence."
        ),
    },
    {
        "id": "af_heart",
        "name": "Amy",
        "piper_model": "voices/en_US-amy-medium.onnx",
        "piper_config": "voices/en_US-amy-medium.onnx.json",
        "intro": (
            "Hi there, I'm Amy. I'm ready. "
            "What would you like to talk about?"
        ),
        "farewell": (
            "Bye for now! I really enjoyed our chat. See you next time!"
        ),
        "system_prompt": (
            "You are Amy, a warm, friendly and helpful American female "
            "English conversation partner. Always reply in English. "
            "Answer in one or two short, complete, conversational sentences. "
            "Never stop mid-sentence."
        ),
    },
]

selected_voice = random.choice(VOICE_OPTIONS)
ACTIVE_VOICE_ID = selected_voice["id"]
ACTIVE_VOICE_NAME = selected_voice["name"]
ACTIVE_PIPER_MODEL = selected_voice["piper_model"]
ACTIVE_PIPER_CONFIG = selected_voice["piper_config"]
ACTIVE_INTRO_TEXT = selected_voice["intro"]
ACTIVE_FAREWELL_TEXT = selected_voice.get("farewell", "Goodbye! Have a great day!")
ACTIVE_SYSTEM_PROMPT = selected_voice["system_prompt"]

def reset_metrics():
    global perf_metrics
    perf_metrics = {
        "rec_start": 0.0,
        "speech_start": 0.0,
        "rec_end": 0.0,
        "stt_start": 0.0,
        "stt_end": 0.0,
        "llm_start": 0.0,
        "llm_first_token": 0.0,
        "llm_end": 0.0,
        "first_tts_chunk_queued": 0.0,
        "tts_first_start": 0.0,
        "tts_first_ready": 0.0,
        "first_audio_played": 0.0
    }

def print_audio_devices():
    try:
        devices = sd.query_devices()
        default_input, default_output = sd.default.device
        print("\nAvailable audio devices:")
        for index, device in enumerate(devices):
            flags = []
            if device["max_input_channels"] > 0:
                flags.append("input")
            if device["max_output_channels"] > 0:
                flags.append("output")
            if index == default_input:
                flags.append("default-in")
            if index == default_output:
                flags.append("default-out")
            print(f"  [{index}] {device['name']} ({', '.join(flags) or 'n/a'})")
    except Exception as e:
        print(f"⚠️ Could not list audio devices: {e}")


def resolve_audio_devices():
    try:
        default_input, default_output = sd.default.device
        input_device = INPUT_DEVICE if INPUT_DEVICE is not None else default_input
        output_device = OUTPUT_DEVICE if OUTPUT_DEVICE is not None else default_output

        input_info = sd.query_devices(input_device, "input") if input_device is not None else None
        output_info = sd.query_devices(output_device, "output") if output_device is not None else None

        print("\nAudio device selection:")
        if input_info:
            print(f"  Input : {input_device} -> {input_info['name']}")
        else:
            print("  Input : none")
        if output_info:
            print(f"  Output: {output_device} -> {output_info['name']}")
        else:
            print("  Output: none")

        return input_device, output_device
    except Exception as e:
        print(f"⚠️ Audio device selection failed: {e}")
        print_audio_devices()
        return INPUT_DEVICE, OUTPUT_DEVICE


def downsample_audio(audio_data, source_rate, target_rate):
    if source_rate == target_rate or len(audio_data) == 0:
        return audio_data.astype(np.float32, copy=False)

    duration = len(audio_data) / float(source_rate)
    target_length = max(1, int(duration * target_rate))
    source_positions = np.linspace(0, len(audio_data) - 1, num=len(audio_data), dtype=np.float32)
    target_positions = np.linspace(0, len(audio_data) - 1, num=target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio_data).astype(np.float32)


def calibrate_microphone_sync(fs, channels):
    print("🎚️ Calibrating microphone; please stay quiet...", flush=True)
    frame_count = max(1, int(fs * CALIBRATION_DURATION))
    recording = sd.rec(
        frame_count,
        samplerate=fs,
        channels=channels,
        dtype="int16",
        device=SELECTED_INPUT_DEVICE,
    )
    sd.wait()

    audio = np.asarray(recording, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]

    block_size = max(1, int(fs * BLOCK_DURATION))
    calibration_levels = []
    for offset in range(0, len(audio), block_size):
        block = audio[offset:offset + block_size]
        if len(block) == 0:
            continue
        channel_rms = np.sqrt(np.mean(np.square(block), axis=0))
        calibration_levels.append(float(np.max(channel_rms)))

    noise_floor = float(np.median(calibration_levels)) if calibration_levels else 0.0
    start_threshold = max(
        float(SPEECH_THRESHOLD),
        noise_floor * SPEECH_THRESHOLD_FACTOR,
        noise_floor + 35.0,
    )
    stop_threshold = max(
        float(SPEECH_THRESHOLD) * 0.8,
        noise_floor * 1.35,
        start_threshold * 0.72,
    )
    print(
        f"🎚️ Mic ready -> noise={noise_floor:.1f}, "
        f"start={start_threshold:.1f}, stop={stop_threshold:.1f}.",
        flush=True,
    )
    return start_threshold, stop_threshold

async def warmup_llm():
    global gemini_client
    print("Checking Gemini API configuration...")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        await gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents="Hi",
        )
        print("✅ Gemini API connection tested successfully.")
    except Exception as e:
        raise RuntimeError(f"Gemini API connection failed: {e}") from e


def load_tts_model():
    global kokoro_tts, piper_tts

    if TTS_ENGINE == "piper":
        print(f"Loading Piper TTS model ({ACTIVE_VOICE_NAME})...")
        piper_tts = PiperVoice.load(
            ACTIVE_PIPER_MODEL,
            config_path=ACTIVE_PIPER_CONFIG,
        )
        print(f"✅ Piper TTS loaded ({ACTIVE_VOICE_NAME}).")
        return

    if TTS_ENGINE == "kokoro":
        print("Loading Kokoro TTS model...")
        kokoro_tts = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
        print("✅ Kokoro TTS loaded.")
        return

    raise RuntimeError(f"Unsupported TTS_ENGINE: {TTS_ENGINE}. Use 'piper' or 'kokoro'.")


def create_tts_audio(text, speed=TTS_SPEECH_SPEED):
    if TTS_ENGINE == "piper":
        if piper_tts is None:
            raise RuntimeError("Piper TTS is not loaded")

        synthesis_config = SynthesisConfig(length_scale=1.0 / speed)
        chunks = list(piper_tts.synthesize(text, synthesis_config))
        if not chunks:
            raise RuntimeError("Piper generated no audio")

        samples = np.concatenate([chunk.audio_float_array for chunk in chunks])
        return samples, chunks[0].sample_rate

    if kokoro_tts is None:
        raise RuntimeError("Kokoro TTS is not loaded")
    return kokoro_tts.create(
        text,
        voice=ACTIVE_VOICE_ID,
        speed=speed,
        lang="en-us",
    )

def record_audio_sync(fs, channels):
    pya = pyaudio.PyAudio()
    recorded_frames = []
    start_time = time.perf_counter()
    speech_started = False
    first_speech_block_time = 0.0
    silence_time = 0.0

    pre_speech_buffer_size = int(0.35 / BLOCK_DURATION)
    pre_speech_blocks = collections.deque(maxlen=pre_speech_buffer_size)
    start_threshold = MIC_START_THRESHOLD
    stop_threshold = MIC_STOP_THRESHOLD
    input_device = SELECTED_INPUT_DEVICE

    try:
        print(f"🎚️ Record config -> fs={fs}, channels={channels}", flush=True)
        stream = pya.open(
            format=MIC_FORMAT,
            channels=channels,
            rate=fs,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=MIC_CHUNK,
        )
        try:
            while True:
                current_time = time.perf_counter()
                elapsed_total = current_time - start_time
                
                if not speech_started and elapsed_total > WAIT_FOR_SPEECH_TIMEOUT:
                    break

                try:
                    data = stream.read(MIC_CHUNK, exception_on_overflow=False)
                except Exception as e:
                    print(f"⚠️ Audio read error: {e}", flush=True)
                    continue

                mono_block = np.frombuffer(data, dtype=np.int16)[0::2].astype(np.float32)
                mono_block = np.clip(mono_block * MIC_GAIN, -32768.0, 32767.0)
                rms = float(np.sqrt(np.mean(np.square(mono_block))))

                if not speech_started:
                    if rms > start_threshold:
                        pre_speech_blocks.append(mono_block)
                        speech_started = True
                        print(f"✅ Speech detected (RMS={rms:.1f}).", flush=True)
                        first_speech_block_time = current_time
                        perf_metrics["speech_start"] = current_time
                        recorded_frames.extend(pre_speech_blocks)
                        pre_speech_blocks.clear()
                    else:
                        pre_speech_blocks.append(mono_block)
                else:
                    recorded_frames.append(mono_block)
                    if rms > stop_threshold:
                        silence_time = 0.0
                    else:
                        silence_time += BLOCK_DURATION

                    speech_duration = current_time - first_speech_block_time
                    if speech_duration >= RECORDING_SAFETY_TIMEOUT:
                        print("⚠️ Recording safety timeout reached; transcribing.", flush=True)
                        break
                    if (
                        speech_duration > MIN_RECORDING_AFTER_SPEECH_START
                        and silence_time >= SILENCE_DURATION
                    ):
                        break
        finally:
            stream.stop_stream()
            stream.close()
                            
        if not speech_started:
            if (time.perf_counter() - start_time) > WAIT_FOR_SPEECH_TIMEOUT:
                print("⌛ Listening timeout reached without speech.", flush=True)
                return "TIMEOUT"
            return None
            
        if (time.perf_counter() - first_speech_block_time) < MIN_SPEECH_DURATION:
            return None
        
        if len(recorded_frames) > 0:
            recording = np.concatenate(recorded_frames, axis=0)
            recording = recording.flatten().astype(np.float32) / 32768.0
            return downsample_audio(recording, fs, WHISPER_SAMPLE_RATE)
        return None
    except Exception as e:
        print(f"⚠️ Microphone open/read error: {e}", flush=True)
        return None
    finally:
        pya.terminate()

def transcribe_audio_sync(stt_model, audio_data):
    if STT_ENGINE == "sherpa":
        try:
            stream = stt_model.create_stream()
            stream.accept_waveform(SHERPA_SAMPLE_RATE, audio_data)
            stt_model.decode_stream(stream)
            result = getattr(stream, "result", "")
            if hasattr(result, "text"):
                result = result.text
            text = str(result).strip()
            print(f"🔎 STT raw: {text or '[empty]'}")
            return text
        except Exception as e:
            print(f"❌ sherpa-onnx transcription error: {e}", flush=True)
            return ""

    try:
        segments, _ = stt_model.transcribe(
            audio_data,
            **WHISPER_TRANSCRIBE_OPTIONS,
        )
        segments = list(segments)
        raw_segments = []
        valid_texts = []

        for segment in segments:
            seg_text = segment.text.strip()
            if seg_text:
                raw_segments.append(seg_text)
            if segment.no_speech_prob <= 0.85 and segment.avg_logprob >= -1.5 and seg_text:
                valid_texts.append(seg_text)

        raw_text = " ".join(raw_segments).strip()
        text = " ".join(valid_texts).strip() if valid_texts else raw_text
        if raw_text:
            debug_segments = " | ".join(
                f"{segment.text.strip()} (ns={segment.no_speech_prob:.2f}, lp={segment.avg_logprob:.2f})"
                for segment in segments
                if segment.text.strip()
            )
            print(f"🔎 STT raw: {debug_segments}")
        if not text:
            return ""
        
        lower_text = text.lower()
        # İNGİLİZCE HALÜSİNASYONLAR
        hallucinations = ["thank you for watching", "thanks for watching", "please subscribe", "subscribe to my channel", "amara.org"]
        for h in hallucinations: lower_text = lower_text.replace(h, "")
        cleaned_text = re.sub(r'[^\w\s]', '', lower_text).strip()
        if len(cleaned_text) <= 1:
            print("⚠️ STT result discarded after cleanup.")
            return ""
        return text
    except Exception as e:
        print(f"❌ Whisper transcription error: {e}", flush=True)
        return ""


def warmup_stt_model(stt_model):
    if STT_ENGINE == "sherpa":
        print("Warming up sherpa-onnx model...")
        silence = np.zeros(SHERPA_SAMPLE_RATE, dtype=np.float32)
        stream = stt_model.create_stream()
        stream.accept_waveform(SHERPA_SAMPLE_RATE, silence)
        stt_model.decode_stream(stream)
        print("✅ sherpa-onnx model ready.")
        return

    print("Warming up Whisper model...")
    silence = np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32)
    segments, _ = stt_model.transcribe(
        silence,
        **WHISPER_TRANSCRIBE_OPTIONS,
    )
    list(segments)
    print("✅ Whisper model ready.")


def load_stt_model():
    if STT_ENGINE == "whisper":
        return WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
            cpu_threads=WHISPER_THREADS,
        )

    if STT_ENGINE == "sherpa":
        if sherpa_onnx is None:
            raise RuntimeError("sherpa_onnx is not installed")
        if not SHERPA_PARAFORMER or not SHERPA_TOKENS:
            raise RuntimeError(
                "Set SHERPA_PARAFORMER and SHERPA_TOKENS to use STT_ENGINE=sherpa"
            )
        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=SHERPA_PARAFORMER,
            tokens=SHERPA_TOKENS,
            num_threads=TOTAL_LOGICAL_CORES,
            sample_rate=SHERPA_SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
            provider="cpu",
        )

    raise RuntimeError(f"Unsupported STT_ENGINE: {STT_ENGINE}. Use 'whisper' or 'sherpa'.")

def remove_emojis(text):
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]|[\u2600-\u27BF]|[\u2300-\u23FF]|[\u2B50-\u2B55]', flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_text_for_tts(text):
    text = remove_emojis(text)
    text = text.replace("\n", " ").strip()
    # ÖNEMLİ DÜZELTME: İngilizce'deki kesme işaretini (') koruyoruz (I'm, don't bozulmaması için)
    text = re.sub(r'[^\w\s.,!?\']', '', text)  
    return re.sub(r'\s+', ' ', text).strip()

# --- ASYNC QUEUE PIPELINE WORKERS ---

stt_queue = asyncio.Queue()
llm_queue = asyncio.Queue()
tts_queue = asyncio.Queue()
playback_queue = asyncio.Queue()
audio_queue = asyncio.Queue() # Bellekte tutulan ses parçaları için kuyruk
can_listen_event = asyncio.Event()
shutdown_event = asyncio.Event()

class ContinuousAudioPlayer:
    def __init__(self):
        self.stream = None
        self.stream_format = None
        self.started = False

    def _ensure_stream(self, sample_rate, channels, dtype):
        stream_format = (sample_rate, channels, dtype)
        if self.stream is not None and self.stream_format != stream_format:
            self.close()

        if self.stream is None:
            self.stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=dtype,
                device=SELECTED_OUTPUT_DEVICE,
            )
            self.stream_format = stream_format

        if not self.started:
            self.stream.start()
            self.started = True

    def play(self, samples, sample_rate):
        audio = np.asarray(samples)
        if audio.dtype not in (np.float32, np.float64, np.int16, np.int32):
            audio = audio.astype(np.float32)
        audio = np.ascontiguousarray(audio)
        channels = audio.shape[1] if audio.ndim > 1 else 1
        self._ensure_stream(sample_rate, channels, audio.dtype.name)
        self.stream.write(audio)

    def finish(self):
        if self.stream is not None and self.started:
            self.stream.stop()
            self.started = False

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
            self.stream_format = None
            self.started = False


def play_audio_sync(samples, sample_rate):
    sd.play(samples, sample_rate, device=SELECTED_OUTPUT_DEVICE)
    sd.wait()


def is_exit_command(text):
    normalized = re.sub(r'[^\w\s]', '', text.lower()).strip()
    return normalized in EXIT_COMMANDS


async def say_farewell(user_text=None):
    if user_text:
        print(f"🗣️ You: {user_text}")
    print(f"🤖 Assistant ({ACTIVE_VOICE_NAME}): {ACTIVE_FAREWELL_TEXT}")

    try:
        samples, sample_rate = await asyncio.to_thread(
            create_tts_audio,
            ACTIVE_FAREWELL_TEXT,
            TTS_SPEECH_SPEED,
        )
        await asyncio.to_thread(play_audio_sync, samples, sample_rate)
    except Exception as e:
        print(f"Farewell playback error: {e}")

    print("\n👋 Assistant shut down cleanly.")
    shutdown_event.set()


async def vad_worker():
    global user_has_spoken
    while True:
        await can_listen_event.wait()
        
        if user_has_spoken:
            print("\n🎤 Listening...")
            user_has_spoken = False
            
        reset_metrics()
        perf_metrics["rec_start"] = time.perf_counter()
        
        audio_data = await asyncio.to_thread(record_audio_sync, INPUT_SAMPLE_RATE, CHANNELS)
        perf_metrics["rec_end"] = time.perf_counter()
        
        if isinstance(audio_data, np.ndarray):
            can_listen_event.clear()
            await stt_queue.put(audio_data)
        elif audio_data == "TIMEOUT":
            await asyncio.sleep(0.1)
        else:
            await asyncio.sleep(0.1)


async def text_input_worker():
    while True:
        await can_listen_event.wait()
        user_text = await asyncio.to_thread(input, "\n⌨️ You: ")
        user_text = user_text.strip()
        if not user_text:
            continue

        if is_exit_command(user_text):
            await say_farewell()
            return

        can_listen_event.clear()
        reset_metrics()
        now = time.perf_counter()
        perf_metrics["rec_start"] = now
        perf_metrics["rec_end"] = now
        await llm_queue.put(user_text)

async def stt_worker(stt_model):
    global user_has_spoken
    while True:
        audio_data = await stt_queue.get()
        perf_metrics["stt_start"] = time.perf_counter()
        text = await asyncio.to_thread(transcribe_audio_sync, stt_model, audio_data)
        perf_metrics["stt_end"] = time.perf_counter()
        
        if text:
            user_has_spoken = True
            if is_exit_command(text):
                await say_farewell(text)
                return
            await llm_queue.put(text)
        else:
            await tts_queue.put(None)

async def llm_worker():
    generation_config = types.GenerateContentConfig(
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
        system_instruction=ACTIVE_SYSTEM_PROMPT,
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL",
            include_thoughts=False
        )
    )
    
    while True:
        text = await llm_queue.get()
        print(f"🗣️ You: {text}")
        
        print("🤖 Assistant: ", end="", flush=True)
        perf_metrics["llm_start"] = time.perf_counter()
        
        full_response = ""
        try:
            messages = [*conversation_history, {"role": "user", "parts": [{"text": text}]}]
            response = await gemini_client.aio.models.generate_content_stream(
                model=GEMINI_MODEL_NAME,
                contents=messages,
                config=generation_config
            )
            
            async for chunk in response:
                try:
                    content = chunk.text
                except Exception:
                    content = ""
                    
                if content:
                    if perf_metrics["llm_first_token"] == 0.0:
                        perf_metrics["llm_first_token"] = time.perf_counter()
                    
                    full_response += content
                    
                    clean_chunk = remove_emojis(content)
                    display_chunk = re.sub(r'[^\w\s\']', '', clean_chunk)
                    print(display_chunk, end="", flush=True)
                    
                    await tts_queue.put(clean_chunk)
                    
            if full_response.strip():
                conversation_history.append({"role": "user", "parts": [{"text": text}]})
                conversation_history.append({"role": "model", "parts": [{"text": full_response.strip()}]})
                conversation_history[:] = conversation_history[-MAX_HISTORY_MESSAGES:]
                if conversation_history and conversation_history[0]["role"] == "model":
                    conversation_history.pop(0)
        except Exception as e:
            print(f"\n❌ Gemini LLM Error: {e}")
            
        perf_metrics["llm_end"] = time.perf_counter()
        print() 
        await tts_queue.put(None)

async def tts_chunk_worker():
    sentence_buffer = ""
    delimiters = re.compile(r'([.?!\n]+)')
    soft_delimiters = re.compile(r'([,;:]+)')
    is_first_chunk = True
    playback_sequence = 0
    playback_turn_id = 0
    
    while True:
        chunk = await tts_queue.get()
        
        if chunk is None:
            current_turn_id = playback_turn_id
            if sentence_buffer.strip():
                clean_sent = clean_text_for_tts(sentence_buffer)
                if any(c.isalnum() for c in clean_sent): 
                    await playback_queue.put((current_turn_id, playback_sequence, clean_sent))
                    playback_sequence += 1
            sentence_buffer = ""
            is_first_chunk = True
            await playback_queue.put(("END", current_turn_id, playback_sequence))
            playback_turn_id += 1
            playback_sequence = 0
        else:
            sentence_buffer += chunk
            
            while True:
                match = delimiters.search(sentence_buffer)
                if not match:
                    chunk_word_limit = FIRST_TTS_WORDS if is_first_chunk else NEXT_TTS_WORDS
                    soft_match = soft_delimiters.search(sentence_buffer)
                    if soft_match and len(sentence_buffer[:soft_match.end()].split()) >= chunk_word_limit:
                        end_idx = soft_match.end()
                        sentence = sentence_buffer[:end_idx].strip()
                        sentence_buffer = sentence_buffer[end_idx:]

                        clean_sent = clean_text_for_tts(sentence)
                        if any(c.isalnum() for c in clean_sent):
                            if perf_metrics["first_tts_chunk_queued"] == 0.0:
                                perf_metrics["first_tts_chunk_queued"] = time.perf_counter()
                            await playback_queue.put((playback_turn_id, playback_sequence, clean_sent))
                            playback_sequence += 1
                            is_first_chunk = False
                        continue

                    # İlk ses çıkışını hızlandırmak için ilk cümlecik erken ateşlenir.
                    if len(sentence_buffer.split()) >= chunk_word_limit:
                        words = sentence_buffer.split()
                        first_part = " ".join(words[:chunk_word_limit]).strip()
                        sentence_buffer = " ".join(words[chunk_word_limit:]).strip()
                        if sentence_buffer:
                            sentence_buffer = " " + sentence_buffer
                        
                        clean_sent = clean_text_for_tts(first_part)
                        if any(c.isalnum() for c in clean_sent): 
                            if perf_metrics["first_tts_chunk_queued"] == 0.0:
                                perf_metrics["first_tts_chunk_queued"] = time.perf_counter()
                            await playback_queue.put((playback_turn_id, playback_sequence, clean_sent))
                            playback_sequence += 1
                            is_first_chunk = False
                        continue
                    break
                    
                end_idx = match.end()
                sentence = sentence_buffer[:end_idx].strip()
                sentence_buffer = sentence_buffer[end_idx:]
                
                clean_sent = clean_text_for_tts(sentence)
                if any(c.isalnum() for c in clean_sent): 
                    if perf_metrics["first_tts_chunk_queued"] == 0.0:
                        perf_metrics["first_tts_chunk_queued"] = time.perf_counter()
                    await playback_queue.put((playback_turn_id, playback_sequence, clean_sent))
                    playback_sequence += 1
                    is_first_chunk = False
                
# YENİ: Metni sese çevirir ve bellekteki audio parçalarını kuyruğa yollar
async def tts_generator_worker():
    while True:
        item = await playback_queue.get()
        
        if isinstance(item, tuple) and item and item[0] == "END":
            _, turn_id, total_sequences = item
            await audio_queue.put(("END", turn_id, total_sequences))
        else:
            turn_id, sequence_id, sentence = item

            try:
                if perf_metrics["tts_first_start"] == 0.0:
                    perf_metrics["tts_first_start"] = time.perf_counter()
                samples, sample_rate = await asyncio.to_thread(
                    create_tts_audio, sentence, TTS_SPEECH_SPEED
                )
                if perf_metrics["tts_first_ready"] == 0.0:
                    perf_metrics["tts_first_ready"] = time.perf_counter()
                await audio_queue.put((turn_id, sequence_id, samples, sample_rate))
                
            except Exception as e:
                print(f"❌ {TTS_ENGINE.title()} generation error: {e}")
                await audio_queue.put(("SKIP", turn_id, sequence_id))

# YENİ: Hazır ses parçalarını sırası korunarak çalar
async def audio_player_worker():
    current_turn_id = 0
    next_sequence_id = 0
    pending_audio = {}
    skipped_audio = set()
    turn_end_markers = {}
    audio_player = ContinuousAudioPlayer()

    try:
        while True:
            item = await audio_queue.get()

            if isinstance(item, tuple) and item and item[0] == "END":
                _, turn_id, total_sequences = item
                turn_end_markers[turn_id] = total_sequences
            elif isinstance(item, tuple) and item and item[0] == "SKIP":
                _, turn_id, sequence_id = item
                skipped_audio.add((turn_id, sequence_id))
            else:
                turn_id, sequence_id, samples, sample_rate = item
                pending_audio[(turn_id, sequence_id)] = (samples, sample_rate)

            while True:
                sequence_key = (current_turn_id, next_sequence_id)
                if sequence_key in skipped_audio:
                    skipped_audio.remove(sequence_key)
                    next_sequence_id += 1
                    continue
                if sequence_key not in pending_audio:
                    break

                samples, sample_rate = pending_audio.pop(sequence_key)
                if perf_metrics["first_audio_played"] == 0.0:
                    perf_metrics["first_audio_played"] = time.perf_counter()

                try:
                    await asyncio.to_thread(audio_player.play, samples, sample_rate)
                except Exception as e:
                    print(f"❌ Audio play error: {e}")

                next_sequence_id += 1

            expected_sequences = turn_end_markers.get(current_turn_id)
            if expected_sequences is not None and next_sequence_id >= expected_sequences:
                try:
                    await asyncio.to_thread(audio_player.finish)
                except Exception as e:
                    print(f"❌ Audio finish error: {e}")

                rec_time = perf_metrics["rec_end"] - perf_metrics["rec_start"]
                wait_time = (
                    perf_metrics["speech_start"] - perf_metrics["rec_start"]
                    if perf_metrics["speech_start"] > 0
                    else 0.0
                )
                capture_time = (
                    perf_metrics["rec_end"] - perf_metrics["speech_start"]
                    if perf_metrics["speech_start"] > 0
                    else rec_time
                )
                stt_time = perf_metrics["stt_end"] - perf_metrics["stt_start"]
                llm_first = perf_metrics["llm_first_token"] - perf_metrics["llm_start"] if perf_metrics["llm_start"] > 0 else 0
                llm_total = perf_metrics["llm_end"] - perf_metrics["llm_start"] if perf_metrics["llm_start"] > 0 else 0
                chunk_wait = perf_metrics["first_tts_chunk_queued"] - perf_metrics["llm_first_token"] if perf_metrics["first_tts_chunk_queued"] > 0 and perf_metrics["llm_first_token"] > 0 else 0
                tts_gen = perf_metrics["tts_first_ready"] - perf_metrics["tts_first_start"] if perf_metrics["tts_first_ready"] > 0 and perf_metrics["tts_first_start"] > 0 else 0
                playback_wait = perf_metrics["first_audio_played"] - perf_metrics["tts_first_ready"] if perf_metrics["first_audio_played"] > 0 and perf_metrics["tts_first_ready"] > 0 else 0
                reaction_time = perf_metrics["first_audio_played"] - perf_metrics["rec_end"] if perf_metrics["first_audio_played"] > 0 else 0

                if perf_metrics["llm_start"] > 0:
                    print(
                        f"\n⏱️ Timing -> Rec: {rec_time:.2f}s "
                        f"(Wait: {wait_time:.2f}s, Speech/end: {capture_time:.2f}s) | "
                        f"Whisper: {stt_time:.2f}s | Gemini First: {llm_first:.2f}s "
                        f"(Total: {llm_total:.2f}s) | TTS Queue: {chunk_wait:.2f}s "
                        f"| TTS Gen: {tts_gen:.2f}s | Play Wait: {playback_wait:.2f}s "
                        f"| 🔥 Reaction: {reaction_time:.2f}s"
                    )
                    print("-" * 50)

                turn_end_markers.pop(current_turn_id, None)
                current_turn_id += 1
                next_sequence_id = 0
                can_listen_event.set()
    finally:
        try:
            audio_player.close()
        except Exception as e:
            print(f"❌ Audio close error: {e}")

async def main():
    global SELECTED_INPUT_DEVICE, SELECTED_OUTPUT_DEVICE
    global MIC_START_THRESHOLD, MIC_STOP_THRESHOLD
    SELECTED_INPUT_DEVICE, SELECTED_OUTPUT_DEVICE = resolve_audio_devices()

    print(
        f"\n🚀 Perfect English Chatbot Started "
        f"({TTS_ENGINE.title()} TTS, {STT_ENGINE.title()} STT & Google Gemini API)."
    )
    await asyncio.to_thread(load_tts_model)
    await warmup_llm()

    if TEXT_MODE_ONLY:
        print("⌨️ Text mode enabled. Type your message and press Enter.")
        asyncio.create_task(text_input_worker())
    else:
        stt_model = await asyncio.to_thread(load_stt_model)
        await asyncio.to_thread(warmup_stt_model, stt_model)
        try:
            MIC_START_THRESHOLD, MIC_STOP_THRESHOLD = await asyncio.to_thread(
                calibrate_microphone_sync,
                INPUT_SAMPLE_RATE,
                CHANNELS,
            )
        except Exception as e:
            print(f"⚠️ Microphone calibration failed; using default thresholds: {e}")
        asyncio.create_task(vad_worker())
        asyncio.create_task(stt_worker(stt_model))
    asyncio.create_task(llm_worker())
    asyncio.create_task(tts_chunk_worker())
    asyncio.create_task(tts_generator_worker())
    asyncio.create_task(audio_player_worker())
    
    # Giriş cümlesini de yeni sisteme uygun güvenli şekilde hazırlıyoruz
    try:
        intro_text = ACTIVE_INTRO_TEXT
        print(f"🤖 Assistant ({ACTIVE_VOICE_NAME}): {intro_text}")
        
        samples, sample_rate = await asyncio.to_thread(
            create_tts_audio,
            intro_text,
            TTS_SPEECH_SPEED,
        )
        await asyncio.to_thread(play_audio_sync, samples, sample_rate)
        
        conversation_history.append({"role": "user", "parts": [{"text": "Hello! Are you ready?"}]})
        conversation_history.append({"role": "model", "parts": [{"text": intro_text}]})
    except Exception as e:
        print(f"Intro playback error: {e}")
        
    can_listen_event.set()
    await shutdown_event.wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Exited.")
    except Exception as error:
        print(f"\n❌ Startup error: {error}")
