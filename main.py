import os
import json
import re
import time
import asyncio
import collections
import aiohttp

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
SPEECH_THRESHOLD = 5000 
START_BLOCK_COUNT = 2    
SILENCE_DURATION = 0.3   #
WAIT_FOR_SPEECH_TIMEOUT = 10.0  
BLOCK_DURATION = 0.05
MIN_SPEECH_DURATION = 0.3   
MIN_RECORDING_AFTER_SPEECH_START = 0.3

PIPER_MODEL = "voices/tr_TR-dfki-medium.onnx"
OLLAMA_MODEL = "gemma2:2b"
WHISPER_MODEL_SIZE = "base"

perf_metrics = {}

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
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "merhaba",
        "stream": False,
        "keep_alive": -1,
        "options": {"num_ctx": 1024, "num_thread": LLM_THREADS, "temperature": 0.2, "num_predict": 1}
    }
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload)
    except Exception:
        pass

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
        stream = sd.InputStream(samplerate=fs, channels=channels, dtype='int16', blocksize=block_size)
        stream.start()
        
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
                    
        stream.stop()
        stream.close()
        
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
            audio_data, beam_size=2, language="tr",
            vad_filter=False, without_timestamps=True,
            condition_on_previous_text=False, temperature=0.0
        )
        valid_texts = [segment.text.strip() for segment in segments if segment.no_speech_prob <= 0.65 and segment.avg_logprob >= -1.0]
        if not valid_texts: return ""
        text = " ".join(valid_texts).strip()
        
        lower_text = text.lower()
        hallucinations = ["kanalıma abone olmayı unutmayın", "izlediğiniz için teşekkürler", "altyazı", "hoşça kalın", "abone ol"]
        for h in hallucinations: lower_text = lower_text.replace(h, "")
        if len(re.sub(r'[^\w\s]', '', lower_text).strip()) <= 1: return ""
        return text
    except Exception:
        return ""

def remove_emojis(text):
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]|[\u2600-\u27BF]|[\u2300-\u23FF]|[\u2B50-\u2B55]', flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def convert_time_format(match):
    hours, minutes = match.groups()
    return f"saat {int(hours)} {int(minutes)}"

def clean_text_for_tts(text):
    text = remove_emojis(text)
    text = re.sub(r'\b([0-1]?[0-9]|2[0-3]):([0-5][0-9])\b', convert_time_format, text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r'[^\w\s.,!?]', '', text)  
    return re.sub(r'\s+', ' ', text).strip()

# --- ASYNC QUEUE PIPELINE WORKERS ---

stt_queue = asyncio.Queue()
llm_queue = asyncio.Queue()
tts_queue = asyncio.Queue()
playback_queue = asyncio.Queue()
can_listen_event = asyncio.Event()

async def vad_worker():
    consecutive_timeouts = 0  
    is_listening_printed = False
    while True:
        await can_listen_event.wait()
        
        if not is_listening_printed:
            print("\n🎤 Dinleniyor...")
            is_listening_printed = True
            
        reset_metrics()
        perf_metrics["rec_start"] = time.perf_counter()
        
        audio_data = await asyncio.to_thread(record_audio_sync, SAMPLE_RATE, CHANNELS)
        perf_metrics["rec_end"] = time.perf_counter()
        
        if audio_data is not None:
            consecutive_timeouts = 0  
            is_listening_printed = False
            can_listen_event.clear()
            await stt_queue.put(audio_data)
        else:
            if consecutive_timeouts == 0:
                can_listen_event.clear()
                await tts_queue.put("10 saniyedir sesinizi alamıyorum, lütfen tekrar eder misiniz?")
                await tts_queue.put(None)
                consecutive_timeouts += 1
                is_listening_printed = False
            else:
                await asyncio.sleep(0.1)

async def stt_worker(stt_model):
    consecutive_unintelligible = 0
    while True:
        audio_data = await stt_queue.get()
        perf_metrics["stt_start"] = time.perf_counter()
        text = await asyncio.to_thread(transcribe_audio_sync, stt_model, audio_data)
        perf_metrics["stt_end"] = time.perf_counter()
        
        if text:
            consecutive_unintelligible = 0
            await llm_queue.put(text)
        else:
            consecutive_unintelligible += 1
            if consecutive_unintelligible == 2:
                await tts_queue.put("Ne dediğinizi tam anlayamadım, tekrar eder misiniz?")
            await tts_queue.put(None)
        stt_queue.task_done()

async def llm_worker():
    url = "http://localhost:11434/api/generate"
    
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        system_prompt = "Sen Türkçe konuşan samimi bir insansın."
    
    async with aiohttp.ClientSession() as session:
        while True:
            text = await llm_queue.get()
            print(f"🗣️ Siz: {text}")
            
            print("🤖 Asistan: ", end="", flush=True)
            perf_metrics["llm_start"] = time.perf_counter()
            
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": f"{system_prompt}\nKullanıcı: {text}\nAsistan:",
                "stream": True,
                "keep_alive": -1, 
                "options": {
                    "num_ctx": 1024, 
                    "num_thread": LLM_THREADS, 
                    "temperature": 0.2,
                    "num_predict": 45
                }
            }
            
            try:
                async with session.post(url, json=payload) as response:
                    async for line in response.content:
                        if line:
                            try:
                                body = json.loads(line.decode('utf-8'))
                                chunk = body.get("response", "")
                                if chunk:
                                    if perf_metrics["llm_first_token"] == 0.0:
                                        perf_metrics["llm_first_token"] = time.perf_counter()
                                    
                                    # Arka plandaki ses bölücü için noktalamalar kalmalı
                                    clean_chunk = remove_emojis(chunk)
                                    
                                    # Ekrana basarken noktalamaları ve emojileri tamamen yok et
                                    display_chunk = re.sub(r'[^\w\s]', '', clean_chunk)
                                    print(display_chunk, end="", flush=True)
                                    
                                    await tts_queue.put(clean_chunk)
                                if body.get("done", False): break
                            except json.JSONDecodeError: pass
            except Exception as e:
                print(f"\n❌ LLM Hatası: {e}")
                
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

async def playback_worker():
    process = None
    t1 = 0
    first_audio_sent = False
    
    while True:
        item = await playback_queue.get()
        
        if item is None:
            if process:
                process.stdin.close()
                await process.wait()
                perf_metrics["tts_total"] += (time.perf_counter() - t1)
                process = None
                
            rec_time = perf_metrics["rec_end"] - perf_metrics["rec_start"]
            stt_time = perf_metrics["stt_end"] - perf_metrics["stt_start"]
            llm_first = perf_metrics["llm_first_token"] - perf_metrics["llm_start"] if perf_metrics["llm_start"] > 0 else 0
            llm_total = perf_metrics["llm_end"] - perf_metrics["llm_start"] if perf_metrics["llm_start"] > 0 else 0
            
            reaction_time = perf_metrics["first_audio_played"] - perf_metrics["rec_end"] if perf_metrics["first_audio_played"] > 0 else 0
            
            print(f"\n⏱️ Süreler -> Kayıt: {rec_time:.2f}s | Whisper: {stt_time:.2f}s | Ollama İlk Kelime: {llm_first:.2f}s (Toplam: {llm_total:.2f}s) | 🔥 Tepki: {reaction_time:.2f}s")
            print("-" * 50)
            
            first_audio_sent = False
            can_listen_event.set()
        else:
            sentence = item
            if not process:
                t1 = time.perf_counter()
                if perf_metrics["first_audio_played"] == 0.0:
                    perf_metrics["first_audio_played"] = time.perf_counter()
                    
                try:
                    piper_aplay_cmd = f"venv/bin/piper --model {PIPER_MODEL} --length_scale 0.80 --sentence_silence 0.01 --output_raw | aplay -q -r 22050 -f S16_LE -t raw -"
                    process = await asyncio.create_subprocess_shell(
                        piper_aplay_cmd,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                except Exception as e:
                    print(f"❌ Oynatma başlatma hatası: {e}")
                    process = None
            
            if process:
                try:
                    process.stdin.write((sentence + "\n").encode('utf-8'))
                    await process.stdin.drain()
                except Exception as e:
                    print(f"❌ Oynatma yazma hatası: {e}")
        playback_queue.task_done()

async def main():
    setup_directories()
    print("\n🚀 Hızlı Kesme Akış Sistemi Başlatıldı (Paralel Piper+Aplay).")
    await warmup_llm()
    
    stt_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=WHISPER_THREADS)
    
    asyncio.create_task(vad_worker())
    asyncio.create_task(stt_worker(stt_model))
    asyncio.create_task(llm_worker())
    asyncio.create_task(tts_chunk_worker())
    asyncio.create_task(playback_worker())
    
    try:
        piper_cmd = f"venv/bin/piper --model {PIPER_MODEL} --length_scale 0.80 --sentence_silence 0.01 --output_raw | aplay -q -r 22050 -f S16_LE -t raw -"
        process = await asyncio.create_subprocess_shell(
            piper_cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await process.communicate(input="Merhaba.".encode('utf-8'))
    except Exception:
        pass
        
    can_listen_event.set()
    while True: await asyncio.sleep(1)

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: print("\n👋 Çıkış yapıldı.")
