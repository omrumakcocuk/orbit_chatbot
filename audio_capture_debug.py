import os
import queue
import wave

import numpy as np
import sounddevice as sd


INPUT_SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
CHANNELS = 2
BLOCK_DURATION = 0.05
RECORD_SECONDS = 10
RAW_OUTPUT_WAV = "debug_capture_48k.wav"
DOWNSAMPLED_OUTPUT_WAV = "debug_capture_16k.wav"


def parse_device_index(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def resolve_input_device():
    requested_device = parse_device_index(os.environ.get("CHATBOT_INPUT_DEVICE"))
    default_input, _default_output = sd.default.device
    input_device = requested_device if requested_device is not None else default_input

    print("Audio device selection:")
    if input_device is None:
        print("  Input : none")
        return None

    input_info = sd.query_devices(input_device, "input")
    print(f"  Input : {input_device} -> {input_info['name']}")
    return input_device


def downsample_audio(audio_data, source_rate, target_rate):
    if source_rate == target_rate or len(audio_data) == 0:
        return audio_data.astype(np.float32, copy=False)

    duration = len(audio_data) / float(source_rate)
    target_length = max(1, int(duration * target_rate))
    print(
        "🔄 Downsampling audio:",
        f"samples={len(audio_data)} @ {source_rate} Hz",
        f"-> target_samples={target_length} @ {target_rate} Hz",
    )
    print(
        f"   duration={duration:.3f}s, ratio={source_rate / float(target_rate):.2f}:1"
    )
    source_positions = np.linspace(0, len(audio_data) - 1, num=len(audio_data), dtype=np.float32)
    target_positions = np.linspace(0, len(audio_data) - 1, num=target_length, dtype=np.float32)
    downsampled = np.interp(target_positions, source_positions, audio_data).astype(np.float32)
    print(
        "✅ Downsampling complete:",
        f"output_samples={len(downsampled)}",
        f"output_duration={len(downsampled) / float(target_rate):.3f}s",
    )
    return downsampled


def save_wav(path, audio_data, sample_rate):
    pcm16 = np.clip(audio_data, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def record_fixed_duration(fs, channels, seconds, input_device):
    block_size = int(fs * BLOCK_DURATION)
    total_blocks = int(seconds / BLOCK_DURATION)
    audio_blocks = queue.Queue()
    recorded_frames = []

    def audio_callback(indata, _frames, _callback_time, status):
        if status:
            print(f"⚠️ Audio callback status: {status}", flush=True)
        audio_blocks.put(indata.copy())

    print(f"🎙️ Recording {seconds} seconds at {fs} Hz...")
    with sd.InputStream(
        samplerate=fs,
        channels=channels,
        dtype="int16",
        blocksize=block_size,
        device=input_device,
        callback=audio_callback,
    ):
        for _ in range(total_blocks):
            data = audio_blocks.get()
            audio_block = data.astype(np.float32)
            if audio_block.ndim == 1:
                audio_block = audio_block[:, None]

            channel_rms = np.sqrt(np.mean(np.square(audio_block), axis=0))
            active_channel = int(np.argmax(channel_rms))
            mono_block = audio_block[:, active_channel]
            recorded_frames.append(mono_block)

    if not recorded_frames:
        raise RuntimeError("No audio was captured")

    recording = np.concatenate(recorded_frames, axis=0)
    return recording.flatten().astype(np.float32) / 32768.0


def main():
    input_device = resolve_input_device()
    if input_device is None:
        raise RuntimeError("No input device available")

    mono_48k = record_fixed_duration(
        INPUT_SAMPLE_RATE,
        CHANNELS,
        RECORD_SECONDS,
        input_device,
    )
    print(
        "📦 Captured mono audio:",
        f"samples={len(mono_48k)}",
        f"duration={len(mono_48k) / float(INPUT_SAMPLE_RATE):.3f}s",
    )
    mono_16k = downsample_audio(mono_48k, INPUT_SAMPLE_RATE, TARGET_SAMPLE_RATE)

    save_wav(RAW_OUTPUT_WAV, mono_48k, INPUT_SAMPLE_RATE)
    save_wav(DOWNSAMPLED_OUTPUT_WAV, mono_16k, TARGET_SAMPLE_RATE)

    print(f"✅ Saved 48 kHz mono capture to {RAW_OUTPUT_WAV}")
    print(f"✅ Saved 16 kHz mono capture to {DOWNSAMPLED_OUTPUT_WAV}")


if __name__ == "__main__":
    main()
