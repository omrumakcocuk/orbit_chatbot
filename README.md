# Orbit Chatbot - Local & Real-time AI Voice Assistant

Orbit Chatbot, tamamen yerel (çevrimdışı) ve CPU/GPU üzerinde yüksek performansla çalışabilen, gerçek zamanlı ve düşük gecikmeli (low-latency) bir İngilizce konuşma ve pratik asistanı projesidir. Proje, ihtiyacınıza göre kullanabileceğiniz iki farklı asistan modunu barındırır:

1. **Doğal ve Akıcı Sohbet Asistanı (`english_bot.py`):** Raspberry Pi'de hızlı çalışması için varsayılan olarak **Piper TTS** kullanır. İstenirse mevcut Kokoro dosyalarıyla `TTS_ENGINE=kokoro` ayarlanarak **Kokoro TTS** motoruna geri dönülebilir.
2. **Seviye Tabanlı İngilizce Eğitmeni (`main.py`):** Hafif ve hızlı **Piper TTS** kullanarak A1, A2, B1 ve B2 seviyelerinden birini seçmenize olanak tanır ve kelimeleri seviyenize göre adapte ederek size pratik yaptırır.

Sistem; kullanıcının konuşmasını tam zamanında yakalamak ve sessizlik bittiği an yanıtlamak için **VAD** (Voice Activity Detection), yüksek doğrulukla sesi metne dökebilmek için **faster-whisper** ve akıl yürüten arka plan modeli olarak **Ollama (`gemma4:e2b`)** kullanır. Tamamen asenkron (async queue pipeline) ve akış (streaming) mantığında tasarlanmıştır; asistan cümleyi üretirken anında kelimeleri sentezleyip kesintisiz seslendirir.

---

## 🌟 Öne Çıkan Özellikler

- **Anında Yanıt (Streaming TTS Pipeline):** Metin oluşturma ve ses senkronizasyonu eş zamanlı işler; cümleyi beklemeden kelimeleri sırasıyla ses dosyasına dönüştürüp aralıksız çalar.
- **Gerçekçi ve Ultra-Doğal Sesler (Kokoro):** Kokoro'nun `am_adam` ve `af_heart` ses modelleri ile donuk veya mekanik olmayan, doğal ve vurgulu İngilizce telaffuzları duyarsınız.
- **Süre Kısıtsız Konuşma & Akıllı VAD:** Cümlelerinizi dilediğiniz uzunlukta kurabilirsiniz; sistem herhangi bir katı saniye sınırı ile sizi kesmez, yalnızca gerçekten konuşmanızı bitirdiğinizde veya durakladığınızda söze girer.
- **Temiz Terminal Deneyimi:** Gereksiz bildirimleri engeller, sohbet boyunca yalnızca bir defa `🎤 Listening...` durumunu gösterir ve konuşmanızı algıladığı anda arayüzden çeker.
- **%100 Yerel, Sıfır Maliyet:** Verileriniz bilgisayarınızdan dışarı çıkmaz; internet bağlantısı veya bulut API anahtarı gerektirmez.

---

## 🛠️ Kurulum ve Hazırlık

Projenin Linux ortamında sorunsuz çalışması için adımları sırasıyla uygulayın:

### 1. Sanal Ortam (Venv) ve Kütüphaneler
Proje dizininde terminali açıp sanal ortamı kurun ve kütüphaneleri (Ollama, Kokoro, Whisper vb.) yükleyin:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ollama ve Yapay Zeka Dil Modelinin İndirilmesi
Asistanın arkasındaki dil motoru için [Ollama](https://ollama.com) kurulu olmalıdır. Ardından asistanın kullandığı **`gemma4:e2b`** modelini indirin:

```bash
ollama run gemma4:e2b
```

### 3. Ses Modellerini Yerleştirme (`voices/` klasörü)
Projenizin ana dizinindeki `voices` klasörü içerisine kullanmak istediğiniz seniaryoya uygun model dosyalarını yerleştirin:

#### A) Kokoro TTS Kurulumu (Opsiyonel)
Kokoro modunu kullanmak için aşağıda belirtilen model dosyaları `voices/` içerisine bırakılmalıdır:
- `kokoro-v1.0.onnx`
- `voices-v1.0.bin`

#### B) Piper TTS Kurulumu (Varsayılan)
`pip install -r requirements.txt` komutu Piper Python paketini kurar. Aşağıdaki model ve JSON dosyalarını `voices/` içerisine indirin:
- [Amy medium](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/medium): `en_US-amy-medium.onnx` ve `en_US-amy-medium.onnx.json`
- [Bryce medium](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/bryce/medium): `en_US-bryce-medium.onnx` ve `en_US-bryce-medium.onnx.json`

---

## 🎧 Çalıştırma ve Kullanim

Kurulumlar tamamlandıktan sonra, sanal ortam (venv) aktifken istediğiniz asistan scriptini çalıştırabilirsiniz:

### 1. Akıcı ve Doğal Sohbet (Piper TTS)
Her başlatmada rastgele olarak **Amy** veya **Bryce** sesli asistanlarından biri ile bağlandığınız, gündelik İngilizce pratik modudur:

```bash
python english_bot.py
```

Kokoro motorunu kullanmak için:

```bash
TTS_ENGINE=kokoro python english_bot.py
```

### 2. Seviye Belirlemeli İngilizce Pratik Eğitmeni (Piper TTS)
Başlangıçta mikrofonunuza *“A1, A2, B1 or B2”* diyerek seviye seçimi yapabildiğiniz rehber modudur:

```bash
python main.py
```

---

## ⚙️ Özelleştirmeler ve Geliştirici Tüyoları
- **Mikrofon Hassasiyeti (VAD):** Eğer arka plan gürültünüz çok yüksekse ve asistan sürekli dinlemeye tetikleniyorsa, `english_bot.py` dosyasındaki `SPEECH_THRESHOLD` (örn: 4000/5000) değerini artırıp, `SILENCE_DURATION` değerini hassasiyetinize göre değiştirebilirsiniz.
- **Otomatik İşlemci Tahsisatı:** Çalıştırılan Python dosyası sisteminizin CPU çekirdek sayısını dinamik olarak ölçer ve arka planda Whisper ile Ollama iş parçacıklarını (threads) en düşük gecikmeyle çalışacak şekilde ayarlar.

---
*Orbit Chatbot — Akıcı pratikler ve keyifli sohbetler dileriz! 🚀🗣️*
