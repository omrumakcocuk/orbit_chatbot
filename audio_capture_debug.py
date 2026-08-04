import os
import time
import wave
from collections import deque

import numpy as np
import pyaudio


INPUT_SAMPLE_RATE = 16000
CHANNELS = 2
MIC_FORMAT = pyaudio.paInt16
MIC_CHUNK = 1024
MIC_GAIN = float(os.environ.get("MIC_GAIN", "15.0"))
SPEECH_THRESHOLD = float(os.environ.get("SPEECH_THRESHOLD", "400"))
SILENCE_DURATION = float(os.environ.get("SILENCE_DURATION", "0.6"))
WAIT_FOR_SPEECH_TIMEOUT = float(os.environ.get("WAIT_FOR_SPEECH_TIMEOUT", "10.0"))
RECORDING_SAFETY_TIMEOUT = float(os.environ.get("RECORDING_SAFETY_TIMEOUT", "60.0"))
PRE_SPEECH_DURATION = 0.35
OUTPUT_WAV = "debug_capture_bot_mic.wav"


def parse_device_index(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def resolve_pulse_input_device(pya):
    requested_device = parse_device_index(os.environ.get("CHATBOT_INPUT_DEVICE"))
    if requested_device is not None:
        info = pya.get_device_info_by_index(requested_device)
        print("Audio device selection:")
        print(f"  Input : {requested_device} -> {info['name']}")
        return requested_device

    pulse_index = None
    for index in range(pya.get_device_count()):
        info = pya.get_device_info_by_index(index)
        if "pulse" in info["name"].lower() and info["maxInputChannels"] > 0:
            pulse_index = index
            break

    print("Audio device selection:")
    if pulse_index is None:
        print("  Input : none")
    else:
        info = pya.get_device_info_by_index(pulse_index)
        print(f"  Input : {pulse_index} -> {info['name']}")
    return pulse_index


def save_wav(path, audio_data, sample_rate):
    pcm16 = np.clip(audio_data, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def record_until_silence():
    pya = pyaudio.PyAudio()
    input_device = resolve_pulse_input_device(pya)
    if input_device is None:
        pya.terminate()
        raise RuntimeError("No Pulse input device found")

    frames = []
    chunk_duration = MIC_CHUNK / float(INPUT_SAMPLE_RATE)
    pre_speech_chunks = deque(
        maxlen=max(1, int(PRE_SPEECH_DURATION / chunk_duration))
    )
    speech_started = False
    listening_started_at = time.monotonic()
    speech_started_at = None
    silence_time = 0.0
    stop_threshold = SPEECH_THRESHOLD * 0.8

    print(
        f"🎙️ Waiting for speech at {INPUT_SAMPLE_RATE} Hz, gain={MIC_GAIN}, "
        f"threshold={SPEECH_THRESHOLD:.0f}..."
    )
    print(f"Recording will stop after {SILENCE_DURATION:.1f}s of silence.")

    stream = pya.open(
        format=MIC_FORMAT,
        channels=CHANNELS,
        rate=INPUT_SAMPLE_RATE,
        input=True,
        input_device_index=input_device,
        frames_per_buffer=MIC_CHUNK,
    )

    try:
        while True:
            data = stream.read(MIC_CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16)[0::2].astype(np.float32)
            samples = np.clip(samples * MIC_GAIN, -32768.0, 32767.0)
            rms = float(np.sqrt(np.mean(np.square(samples))))
            now = time.monotonic()

            if not speech_started:
                pre_speech_chunks.append(samples)
                if rms > SPEECH_THRESHOLD:
                    speech_started = True
                    speech_started_at = now
                    frames.extend(pre_speech_chunks)
                    pre_speech_chunks.clear()
                    print(f"✅ Speech detected (RMS={rms:.1f}). Speak now...")
                elif now - listening_started_at >= WAIT_FOR_SPEECH_TIMEOUT:
                    raise RuntimeError(
                        f"No speech detected within {WAIT_FOR_SPEECH_TIMEOUT:.1f} seconds"
                    )
                continue

            frames.append(samples)
            if rms > stop_threshold:
                silence_time = 0.0
            else:
                silence_time += chunk_duration

            speech_duration = now - speech_started_at
            if silence_time >= SILENCE_DURATION:
                print(f"🛑 Speech ended ({silence_time:.1f}s silence detected).")
                break
            if speech_duration >= RECORDING_SAFETY_TIMEOUT:
                print("⚠️ Recording safety timeout reached.")
                break
    finally:
        stream.stop_stream()
        stream.close()
        pya.terminate()

    if not frames:
        raise RuntimeError("No audio was captured")

    recording = np.concatenate(frames, axis=0)
    return recording.astype(np.float32) / 32768.0


def main():
    mono_audio = record_until_silence()
    duration = len(mono_audio) / float(INPUT_SAMPLE_RATE)
    rms = float(np.sqrt(np.mean(np.square(mono_audio)))) if len(mono_audio) else 0.0

    print(
        "📦 Captured mono audio:",
        f"samples={len(mono_audio)}",
        f"duration={duration:.3f}s",
        f"rms={rms:.4f}",
    )

    save_wav(OUTPUT_WAV, mono_audio, INPUT_SAMPLE_RATE)
    print(f"✅ Saved capture to {OUTPUT_WAV}")
    print(f"▶️ Play it with: aplay {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
