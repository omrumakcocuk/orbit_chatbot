# Orbit Chatbot - Local Voice Assistant

Orbit Chatbot, tamamen yerel (çevrimdışı) ve CPU üzerinde çalışabilen, yüksek hızlı bir sesli asistan projesidir. Sistem; kullanıcının sesini algılamak için VAD (Voice Activity Detection), sesi metne çevirmek için `faster-whisper`, doğal dil işleme için `Ollama (Gemma2)` ve metni sese dönüştürmek için `Piper TTS` kullanır. Akış (streaming) mimarisi sayesinde inanılmaz düşük tepki sürelerine sahiptir.

## 🚀 Özellikler
- **Gerçek Zamanlı Tepki:** Cümleler oluşturulurken anında seslendirilmeye başlar (Streaming TTS).
- **Hızlı Kesme (VAD):** Siz konuşurken asistan anında susar ve sizi dinler.
- **Tamamen Yerel:** İnternet bağlantısı gerektirmez (API maliyeti yoktur).

---

## 🛠️ Kurulum Rehberi

Projenin bilgisayarınızda çalışabilmesi için aşağıdaki adımları sırasıyla uygulayın:

### 1. Python Kütüphanelerini Kurun
Sisteme gerekli Python modüllerini kurmak için terminalde şu komutu çalıştırın:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ollama ve Dil Modelini İndirin
Yapay zekanın mantık motoru için **Ollama** yüklü olmalıdır. Yüklü değilse [ollama.com](https://ollama.com) adresinden indirin. Ardından asistanın beyni olan modeli indirin:
```bash
ollama run gemma2:2b
```

### 3. Piper TTS (Ses Motoru) Kurulumu
Asistanın Türkçe konuşabilmesi için Piper motoruna ve ses modeline ihtiyacı vardır.
1. [Piper GitHub](https://github.com/rhasspy/piper/releases) sayfasından sisteminize uygun olan sürümü (örn: `piper_linux_x86_64.tar.gz`) indirin.
2. Arşivden çıkan `piper` çalıştırılabilir dosyasını projenin içindeki `venv/bin/` klasörüne kopyalayın.
3. [Piper Voices](https://huggingface.co/rhasspy/piper-voices/tree/main/tr/tr_TR/dfki/medium) sayfasından Türkçe modeli indirin:
   - `tr_TR-dfki-medium.onnx`
   - `tr_TR-dfki-medium.onnx.json`
4. Proje ana dizininde `voices` adında bir klasör oluşturun ve indirdiğiniz bu iki dosyayı içine atın.

---

## 🎧 Çalıştırma

Tüm kurulumlar tamamlandıktan sonra, sanal ortamınız (venv) aktifken asistanı başlatın:

```bash
python main.py
```
Mikrofonunuza konuşarak asistanla sohbete başlayabilirsiniz!
