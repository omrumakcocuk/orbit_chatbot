import os
import json
import re
import time
import asyncio
import collections
import ollama
import tempfile
import aiohttp
import soundfile as sf
import random
from kokoro_onnx import Kokoro

import numpy as np
import sounddevice as sd

from faster_whisper import WhisperModel
import warnings
warnings.filterwarnings("ignore")

# --- DİNAMİK OTOMATİK ÇEKİRDEK AYARI ---
TOTAL_LOGICAL_CORES = os.cpu_count() or 4
LLM_THREADS = max(1, int(TOTAL_LOGICAL_CORES * 0.75))
WHISPER_THREADS = max(1, TOTAL_LOGICAL_CORES - 1)

# --- YAPILANDIRMA (Hızlı Kesme Odaklı) ---
SAMPLE_RATE = 16000 
CHANNELS = 1
SPEECH_THRESHOLD = 4000
START_BLOCK_COUNT = 2    
SILENCE_DURATION = 0.7   
WAIT_FOR_SPEECH_TIMEOUT = 10.0  
BLOCK_DURATION = 0.05
MIN_SPEECH_DURATION = 0.3   
MIN_RECORDING_AFTER_SPEECH_START = 0.3

# İNGİLİZCE MODELLER
KOKORO_MODEL = "voices/kokoro-v1.0.onnx"
KOKORO_VOICES = "voices/voices-v1.0.bin"

try:
    print("Loading Kokoro TTS model...")
    kokoro_tts = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    print("✅ Kokoro TTS loaded.")
except Exception as e:
    print(f"⚠️ Failed to load Kokoro TTS: {e}")
    kokoro_tts = None

OLLAMA_MODEL = "gemma4:e2b"
WHISPER_MODEL_SIZE = "base"

LLM_CONTEXT_SIZE = 1024
LLM_MAX_TOKENS = 150
LLM_TEMPERATURE = 0.3
MAX_HISTORY_MESSAGES = 6

perf_metrics = {}
conversation_history = []
user_has_spoken = True

# --- RASTGELE SESLİ ASİSTAN SEÇİMİ ---
# Program her açıldığında aşağıdaki asistanlardan biri seçilir.
# Seçilen asistan, konuşma boyunca değişmeden kullanılır.
VOICE_OPTIONS = [
    {
        "id": "am_adam",
        "name": "Adam",
        "intro": (
            "Hello, I'm Adam. I'm ready. "
            "What would you like to talk about today?"
        ),
        "system_prompt": (
            "You are Adam, a friendly, natural and confident American male "
            "English conversation partner. Always reply in English. "
            "Keep your answers short and conversational."
        ),
    },
    {
        "id": "af_heart",
        "name": "Heart",
        "intro": (
            "Hi there, I'm Heart. I'm ready. "
            "What would you like to talk about?"
        ),
        "system_prompt": (
            "You are Heart, a warm, friendly and helpful American female "
            "English conversation partner. Always reply in English. "
            "Keep your answers short and conversational."
        ),
    },
]

selected_voice = random.choice(VOICE_OPTIONS)
ACTIVE_VOICE_ID = selected_voice["id"]
ACTIVE_VOICE_NAME = selected_voice["name"]
ACTIVE_INTRO_TEXT = selected_voice["intro"]
ACTIVE_SYSTEM_PROMPT = selected_voice["system_prompt"]

def reset_metrics():
    global perf_metrics
    perf_metrics = {
        "rec_start": 0.0,
        "rec_end": 0.0,
        "stt_start": 0.0,
        "stt_end": 0.0,
        "llm_start": 0.0,
        "llm_first_token": 0.0,
        "llm_end": 0.0,
        "tts_total": 0.0,
        "first_audio_played": 0.0
    }

def setup_directories():
    os.makedirs("voices", exist_ok=True)

async def warmup_llm():
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": "Reply with only: Ready"}],
        "stream": False,
        "think": False,
        "keep_alive": -1,
        "options": {
            "num_ctx": LLM_CONTEXT_SIZE,
            "num_thread": LLM_THREADS,
            "temperature": 0.0,
            "num_predict": 10
        }
    }

    try:
        # Modelin ilk yüklenmesi 3 saniyeden uzun sürebildiği için güvenli timeout.
        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                await response.json()

        print("✅ Ollama warmup completed.")

    except Exception as e:
        # Warmup başarısız olsa bile chatbot çalışmaya devam eder.
        print(f"⚠️ Ollama warmup error: {e}")

def record_audio_sync(fs, channels):
    block_size = int(fs * BLOCK_DURATION)
    recorded_frames = []
    start_time = time.perf_counter()
    speech_started = False
    consecutive_high_blocks = 0
    max_rms = 0.0
    first_speech_block_time = 0.0
    silence_time = 0.0
    
    pre_speech_buffer_size = int(0.6 / BLOCK_DURATION)
    pre_speech_blocks = collections.deque(maxlen=pre_speech_buffer_size)
    
    try:
        # with bloğu kullanmak, hata durumunda bile stream'in (mikrofon akışının) kesin kapanmasını sağlar
        with sd.InputStream(samplerate=fs, channels=channels, dtype='int16', blocksize=block_size) as stream:
            
            while True:
                current_time = time.perf_counter()
                elapsed_total = current_time - start_time
                
                if not speech_started and elapsed_total > WAIT_FOR_SPEECH_TIMEOUT:
                    break
                    
                data, overflowed = stream.read(block_size)
                rms = np.sqrt(np.mean(np.square(data.astype(np.float32))))
                if rms > max_rms: max_rms = rms
                    
                if not speech_started:
                    if rms > SPEECH_THRESHOLD:
                        consecutive_high_blocks += 1
                        pre_speech_blocks.append(data)
                        if consecutive_high_blocks >= START_BLOCK_COUNT:
                            speech_started = True
                            first_speech_block_time = current_time
                            recorded_frames.extend(pre_speech_blocks)
                            pre_speech_blocks.clear()
                    else:
                        consecutive_high_blocks = 0
                        pre_speech_blocks.append(data)
                else:
                    recorded_frames.append(data)
                    if rms > SPEECH_THRESHOLD:
                        silence_time = 0.0
                    else:
                        silence_time += BLOCK_DURATION
                        
                    if (current_time - first_speech_block_time) > MIN_RECORDING_AFTER_SPEECH_START:
                        if silence_time > SILENCE_DURATION:
                            break
                            
        if not speech_started:
            if (time.perf_counter() - start_time) > WAIT_FOR_SPEECH_TIMEOUT:
                return "TIMEOUT"
            return None
            
        if max_rms < SPEECH_THRESHOLD: return None
        if (time.perf_counter() - first_speech_block_time) < MIN_SPEECH_DURATION: return None
        
        if len(recorded_frames) > 0:
            recording = np.concatenate(recorded_frames, axis=0)
            return recording.flatten().astype(np.float32) / 32768.0
        return None
    except Exception:
        return None

def transcribe_audio_sync(stt_model, audio_data):
    try:
        segments, _ = stt_model.transcribe(
            audio_data, beam_size=2, language="en", # İNGİLİZCE
            vad_filter=False, without_timestamps=True,
            condition_on_previous_text=False, temperature=0.0
        )
        valid_texts = [segment.text.strip() for segment in segments if segment.no_speech_prob <= 0.65 and segment.avg_logprob >= -1.0]
        if not valid_texts: return ""
        text = " ".join(valid_texts).strip()
        
        lower_text = text.lower()
        # İNGİLİZCE HALÜSİNASYONLAR
        hallucinations = ["thank you for watching", "thanks for watching", "please subscribe", "subscribe to my channel", "amara.org", "bye"]
        for h in hallucinations: lower_text = lower_text.replace(h, "")
        if len(re.sub(r'[^\w\s]', '', lower_text).strip()) <= 1: return ""
        return text
    except Exception:
        return ""

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
audio_queue = asyncio.Queue() # YENİ: Hazır ses dosyaları için kuyruk
can_listen_event = asyncio.Event()

async def vad_worker():
    global user_has_spoken
    consecutive_timeouts = 0  
    while True:
        await can_listen_event.wait()
        
        if user_has_spoken:
            print("\n🎤 Listening...")
            user_has_spoken = False
            
        reset_metrics()
        perf_metrics["rec_start"] = time.perf_counter()
        
        audio_data = await asyncio.to_thread(record_audio_sync, SAMPLE_RATE, CHANNELS)
        perf_metrics["rec_end"] = time.perf_counter()
        
        if isinstance(audio_data, np.ndarray):
            consecutive_timeouts = 0
            can_listen_event.clear()
            await stt_queue.put(audio_data)
        elif audio_data == "TIMEOUT":
            await asyncio.sleep(0.1)
        else:
            await asyncio.sleep(0.1)

async def stt_worker(stt_model):
    global user_has_spoken
    while True:
        audio_data = await stt_queue.get()
        perf_metrics["stt_start"] = time.perf_counter()
        text = await asyncio.to_thread(transcribe_audio_sync, stt_model, audio_data)
        perf_metrics["stt_end"] = time.perf_counter()
        
        if text:
            user_has_spoken = True
            await llm_queue.put(text)
        else:
            await tts_queue.put(None)
        stt_queue.task_done()

async def llm_worker():
    global conversation_history
    system_prompt = ACTIVE_SYSTEM_PROMPT
    
    client = ollama.AsyncClient()
    
    while True:
        text = await llm_queue.get()
        print(f"🗣️ You: {text}")
        
        print("🤖 Assistant: ", end="", flush=True)
        perf_metrics["llm_start"] = time.perf_counter()
        
        full_response = ""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                *conversation_history,
                {"role": "user", "content": text}
            ]
            
            response = await client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                think=False,
                stream=True,
                keep_alive=-1,
                options={
                    "num_ctx": LLM_CONTEXT_SIZE, 
                    "num_thread": LLM_THREADS, 
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": LLM_MAX_TOKENS
                }
            )
            
            async for chunk in response:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    if perf_metrics["llm_first_token"] == 0.0:
                        perf_metrics["llm_first_token"] = time.perf_counter()
                    
                    full_response += content
                    
                    clean_chunk = remove_emojis(content)
                    display_chunk = re.sub(r'[^\w\s\']', '', clean_chunk)
                    print(display_chunk, end="", flush=True)
                    
                    await tts_queue.put(clean_chunk)
                    
            if full_response.strip():
                conversation_history.append({"role": "user", "content": text})
                conversation_history.append({"role": "assistant", "content": full_response.strip()})
                conversation_history[:] = conversation_history[-MAX_HISTORY_MESSAGES:]
        except Exception as e:
            print(f"\n❌ LLM Error: {e}")
            
        perf_metrics["llm_end"] = time.perf_counter()
        print() 
        await tts_queue.put(None)
        llm_queue.task_done()

async def tts_chunk_worker():
    sentence_buffer = ""
    delimiters = re.compile(r'([.?!:\n]+)')
    
    while True:
        chunk = await tts_queue.get()
        
        if chunk is None:
            if sentence_buffer.strip():
                clean_sent = clean_text_for_tts(sentence_buffer)
                if any(c.isalnum() for c in clean_sent): 
                    await playback_queue.put(clean_sent)
            sentence_buffer = ""
            await playback_queue.put(None)
        else:
            sentence_buffer += chunk
            
            while True:
                match = delimiters.search(sentence_buffer)
                if not match:
                    break
                    
                end_idx = match.end()
                sentence = sentence_buffer[:end_idx].strip()
                sentence_buffer = sentence_buffer[end_idx:]
                
                clean_sent = clean_text_for_tts(sentence)
                if any(c.isalnum() for c in clean_sent): 
                    await playback_queue.put(clean_sent)
                
        tts_queue.task_done()

# YENİ: Sadece metni Piper'a gönderip sesi geçici bir wav dosyasına kaydeder
async def tts_generator_worker():
    while True:
        item = await playback_queue.get()
        
        if item is None:
            await audio_queue.put(None)
        else:
            sentence = item
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_filepath = temp_file.name
            temp_file.close()

            try:
                if kokoro_tts:
                    samples, sample_rate = await asyncio.to_thread(
                        kokoro_tts.create, sentence, voice=ACTIVE_VOICE_ID, speed=1.20, lang="en-us"
                    )
                    await asyncio.to_thread(sf.write, temp_filepath, samples, sample_rate)
                    await audio_queue.put(temp_filepath)
                else:
                    raise Exception("Kokoro TTS not loaded")
                
            except Exception as e:
                print(f"❌ Kokoro generation error: {e}")
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                
        playback_queue.task_done()

# YENİ: Sadece hazır ses dosyalarını alır ve hiç beklemeden arka arkaya çalar
async def audio_player_worker():
    while True:
        filepath = await audio_queue.get()
        
        if filepath is None:
            rec_time = perf_metrics["rec_end"] - perf_metrics["rec_start"]
            stt_time = perf_metrics["stt_end"] - perf_metrics["stt_start"]
            llm_first = perf_metrics["llm_first_token"] - perf_metrics["llm_start"] if perf_metrics["llm_start"] > 0 else 0
            llm_total = perf_metrics["llm_end"] - perf_metrics["llm_start"] if perf_metrics["llm_start"] > 0 else 0
            reaction_time = perf_metrics["first_audio_played"] - perf_metrics["rec_end"] if perf_metrics["first_audio_played"] > 0 else 0
            
            if perf_metrics["llm_start"] > 0:
                print(f"\n⏱️ Timing -> Rec: {rec_time:.2f}s | Whisper: {stt_time:.2f}s | Ollama First: {llm_first:.2f}s (Total: {llm_total:.2f}s) | 🔥 Reaction: {reaction_time:.2f}s")
                print("-" * 50)
                
            can_listen_event.set()
        else:
            if perf_metrics["first_audio_played"] == 0.0:
                perf_metrics["first_audio_played"] = time.perf_counter()
                
            try:
                play_cmd = f"aplay -q {filepath}"
                process = await asyncio.create_subprocess_shell(play_cmd)
                await process.wait() 
            except Exception as e:
                print(f"❌ Audio play error: {e}")
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
                
        audio_queue.task_done()

async def main():
    setup_directories()
    print("\n🚀 Perfect English Chatbot Started (Parallel Piper Generator & Aplay).")
    await warmup_llm()
    
    stt_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=WHISPER_THREADS)
    
    asyncio.create_task(vad_worker())
    asyncio.create_task(stt_worker(stt_model))
    asyncio.create_task(llm_worker())
    asyncio.create_task(tts_chunk_worker())
    asyncio.create_task(tts_generator_worker())
    asyncio.create_task(audio_player_worker())
    
    # Giriş cümlesini de yeni sisteme uygun güvenli şekilde hazırlıyoruz
    try:
        temp_intro = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_intro_path = temp_intro.name
        temp_intro.close()
        
        intro_text = ACTIVE_INTRO_TEXT
        print(f"🤖 Assistant ({ACTIVE_VOICE_NAME}): {intro_text}")
        
        if kokoro_tts:
            samples, sample_rate = await asyncio.to_thread(
                kokoro_tts.create, intro_text, voice=ACTIVE_VOICE_ID, speed=1.15, lang="en-us"
            )
            await asyncio.to_thread(sf.write, temp_intro_path, samples, sample_rate)
        
        play_cmd = f"aplay -q {temp_intro_path}"
        play_proc = await asyncio.create_subprocess_shell(play_cmd)
        await play_proc.wait()
        
        conversation_history.append({"role": "assistant", "content": intro_text})
    except Exception as e:
        print(f"Intro playback error: {e}")
    finally:
        if os.path.exists(temp_intro_path):
            os.remove(temp_intro_path)
        
    can_listen_event.set()
    while True: await asyncio.sleep(1)

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: print("\n👋 Exited.")