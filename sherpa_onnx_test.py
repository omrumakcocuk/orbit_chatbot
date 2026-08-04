import collections
import os
import queue
import time

import numpy as np
import sounddevice as sd

try:
    import sherpa_onnx
except ModuleNotFoundError as exc:
    raise SystemExit(
        "sherpa_onnx is not installed. Run: pip install sherpa-onnx"
    ) from exc


def _parse_device_index(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


TOTAL_LOGICAL_CORES = os.cpu_count() or 4

INPUT_SAMPLE_RATE = 48000
SHERPA_SAMPLE_RATE = 16000
CHANNELS = 2
SPEECH_THRESHOLD = float(os.environ.get("SPEECH_THRESHOLD", "400"))
SPEECH_THRESHOLD_FACTOR = 1.55
CALIBRATION_DURATION = 0.3
SILENCE_DURATION = 0.6
WAIT_FOR_SPEECH_TIMEOUT = 10.0
RECORDING_SAFETY_TIMEOUT = 60.0
BLOCK_DURATION = 0.05
MIN_SPEECH_DURATION = 0.15
MIN_RECORDING_AFTER_SPEECH_START = 0.20
INPUT_DEVICE = _parse_device_index(os.environ.get("CHATBOT_INPUT_DEVICE"))
SELECTED_INPUT_DEVICE = INPUT_DEVICE
MIC_START_THRESHOLD = float(SPEECH_THRESHOLD)
MIC_STOP_THRESHOLD = float(SPEECH_THRESHOLD) * 0.8

SHERPA_TOKENS = os.environ.get("SHERPA_TOKENS", "").strip()
SHERPA_PARAFORMER = os.environ.get("SHERPA_PARAFORMER", "").strip()
SHERPA_ENCODER = os.environ.get("SHERPA_ENCODER", "").strip()
SHERPA_DECODER = os.environ.get("SHERPA_DECODER", "").strip()
SHERPA_JOINER = os.environ.get("SHERPA_JOINER", "").strip()


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
    except Exception as error:
        print(f"⚠️ Could not list audio devices: {error}")


def resolve_audio_device():
    try:
        default_input, _default_output = sd.default.device
        input_device = INPUT_DEVICE if INPUT_DEVICE is not None else default_input
        input_info = sd.query_devices(input_device, "input") if input_device is not None else None

        print("\nAudio device selection:")
        if input_info:
            print(f"  Input : {input_device} -> {input_info['name']}")
        else:
            print("  Input : none")

        return input_device
    except Exception as error:
        print(f"⚠️ Audio device selection failed: {error}")
        print_audio_devices()
        return INPUT_DEVICE


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


def record_audio_sync(fs, channels):
    block_size = int(fs * BLOCK_DURATION)
    recorded_frames = []
    start_time = time.perf_counter()
    speech_started = False
    first_speech_block_time = 0.0
    silence_time = 0.0

    pre_speech_buffer_size = int(0.35 / BLOCK_DURATION)
    pre_speech_blocks = collections.deque(maxlen=pre_speech_buffer_size)
    audio_blocks = queue.Queue()
    start_threshold = MIC_START_THRESHOLD
    stop_threshold = MIC_STOP_THRESHOLD

    def audio_callback(indata, _frames, _callback_time, status):
        if status:
            print(f"⚠️ Audio callback status: {status}", flush=True)
        audio_blocks.put(indata.copy())

    try:
        print(f"🎚️ Record config -> fs={fs}, channels={channels}", flush=True)
        with sd.InputStream(
            samplerate=fs,
            channels=channels,
            dtype="int16",
            blocksize=block_size,
            device=SELECTED_INPUT_DEVICE,
            callback=audio_callback,
        ):
            while True:
                current_time = time.perf_counter()
                elapsed_total = current_time - start_time

                if not speech_started and elapsed_total > WAIT_FOR_SPEECH_TIMEOUT:
                    break

                try:
                    data = audio_blocks.get(timeout=max(BLOCK_DURATION * 2, 0.1))
                except queue.Empty:
                    continue

                audio_block = data.astype(np.float32)
                if audio_block.ndim == 1:
                    audio_block = audio_block[:, None]

                channel_rms = np.sqrt(np.mean(np.square(audio_block), axis=0))
                active_channel = int(np.argmax(channel_rms))
                rms = float(channel_rms[active_channel])
                mono_block = audio_block[:, active_channel]

                if not speech_started:
                    if rms > start_threshold:
                        pre_speech_blocks.append(mono_block)
                        speech_started = True
                        print(f"✅ Speech detected (RMS={rms:.1f}).", flush=True)
                        first_speech_block_time = current_time
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
            return downsample_audio(recording, fs, SHERPA_SAMPLE_RATE)
        return None
    except Exception as error:
        print(f"⚠️ Microphone open/read error: {error}", flush=True)
        return None


def create_recognizer():
    print("Loading sherpa-onnx recognizer...")

    if SHERPA_PARAFORMER:
        if not SHERPA_TOKENS:
            raise RuntimeError("Missing Sherpa model env var: SHERPA_TOKENS")
        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=SHERPA_PARAFORMER,
            tokens=SHERPA_TOKENS,
            num_threads=TOTAL_LOGICAL_CORES,
            sample_rate=SHERPA_SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
            provider="cpu",
        )
    else:
        missing = [
            name for name, value in (
                ("SHERPA_TOKENS", SHERPA_TOKENS),
                ("SHERPA_ENCODER", SHERPA_ENCODER),
                ("SHERPA_DECODER", SHERPA_DECODER),
                ("SHERPA_JOINER", SHERPA_JOINER),
            ) if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing Sherpa model env vars: " + ", ".join(missing)
            )
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=SHERPA_ENCODER,
            decoder=SHERPA_DECODER,
            joiner=SHERPA_JOINER,
            tokens=SHERPA_TOKENS,
            num_threads=TOTAL_LOGICAL_CORES,
            sample_rate=SHERPA_SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
            provider="cpu",
        )

    print("✅ sherpa-onnx recognizer ready.")
    return recognizer


def transcribe_audio_sync(recognizer, audio_data):
    try:
        stream = recognizer.create_stream()
        stream.accept_waveform(SHERPA_SAMPLE_RATE, audio_data)
        recognizer.decode_stream(stream)
        result = getattr(stream, "result", "")
        if hasattr(result, "text"):
            result = result.text
        return str(result).strip()
    except Exception as error:
        print(f"❌ sherpa-onnx transcription error: {error}", flush=True)
        return ""


def main():
    global SELECTED_INPUT_DEVICE, MIC_START_THRESHOLD, MIC_STOP_THRESHOLD

    SELECTED_INPUT_DEVICE = resolve_audio_device()
    recognizer = create_recognizer()
    MIC_START_THRESHOLD, MIC_STOP_THRESHOLD = calibrate_microphone_sync(
        INPUT_SAMPLE_RATE,
        CHANNELS,
    )

    print("\n🚀 sherpa-onnx test started. Speak in English. Ctrl+C to exit.")
    while True:
        print("\n🎤 Listening...")
        audio_data = record_audio_sync(INPUT_SAMPLE_RATE, CHANNELS)
        if isinstance(audio_data, np.ndarray):
            print(
                "📦 Captured audio:",
                f"samples={len(audio_data)}",
                f"duration={len(audio_data) / float(SHERPA_SAMPLE_RATE):.2f}s",
            )
            text = transcribe_audio_sync(recognizer, audio_data)
            print(f"📝 sherpa-onnx: {text or '[empty]'}")
        elif audio_data == "TIMEOUT":
            continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exited.")
