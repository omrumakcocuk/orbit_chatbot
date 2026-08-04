import os
import wave

import numpy as np
import pyaudio


INPUT_SAMPLE_RATE = 16000
CHANNELS = 2
MIC_FORMAT = pyaudio.paInt16
MIC_CHUNK = 1024
MIC_GAIN = float(os.environ.get("MIC_GAIN", "15.0"))
RECORD_SECONDS = 5
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


def record_fixed_duration():
    pya = pyaudio.PyAudio()
    input_device = resolve_pulse_input_device(pya)
    if input_device is None:
        pya.terminate()
        raise RuntimeError("No Pulse input device found")

    total_chunks = max(1, int((INPUT_SAMPLE_RATE * RECORD_SECONDS) / MIC_CHUNK))
    frames = []

    print(
        f"🎙️ Recording {RECORD_SECONDS} seconds "
        f"at {INPUT_SAMPLE_RATE} Hz, gain={MIC_GAIN}..."
    )

    stream = pya.open(
        format=MIC_FORMAT,
        channels=CHANNELS,
        rate=INPUT_SAMPLE_RATE,
        input=True,
        input_device_index=input_device,
        frames_per_buffer=MIC_CHUNK,
    )

    try:
        for _ in range(total_chunks):
            data = stream.read(MIC_CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16)[0::2].astype(np.float32)
            samples = np.clip(samples * MIC_GAIN, -32768.0, 32767.0)
            frames.append(samples)
    finally:
        stream.stop_stream()
        stream.close()
        pya.terminate()

    if not frames:
        raise RuntimeError("No audio was captured")

    recording = np.concatenate(frames, axis=0)
    return recording.astype(np.float32) / 32768.0


def main():
    mono_audio = record_fixed_duration()
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
