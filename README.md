# 🎙️ Dera — AI Audio Transcriber

**Dera** è un'applicazione desktop moderna e intuitiva per la trascrizione locale di file audio e video. Alimentata da **Faster-Whisper**, permette di trasformare parlato in testo con estrema precisione, rispettando la privacy (tutto avviene sul tuo PC) e offrendo strumenti avanzati come il ripristino della punteggiatura tramite AI.



- **🚀 Engine Faster-Whisper:** Utilizza l'implementazione più veloce esistente del modello Whisper di OpenAI.
- **🧠 5 Livelli di Precisione:** Scegli tra i modelli `tiny`, `base`, `small`, `medium` e `large-v3` in base alle tue risorse hardware.
- **✍️ Punteggiatura AI:** Integrazione con `deepmultilingualpunctuation` per aggiungere automaticamente virgole, punti e punti interrogativi al testo trascritto.
- **🔇 Filtro VAD (Voice Activity Detection):** Salta automaticamente i silenzi per velocizzare l'elaborazione.
- **⚡ Accelerazione Hardware:** Supporto completo per GPU NVIDIA (CUDA) per trascrizioni fino a 20 volte più veloci.
- **📂 Export Flessibile:** Esporta in modalità "Grezza" (con timestamp) o "Pulita" (testo pronto per documenti).
- **🎨 Interfaccia Moderna:** UI curata in Dark Mode realizzata con `CustomTkinter`.

## 🛠️ Requisiti di sistema

- **Python:** 3.8 o superiore.
- **FFmpeg:** Necessario per la gestione dei file multimediali.
  - *Windows:* `choco install ffmpeg` o scarica da ffmpeg.org
  - *Mac:* `brew install ffmpeg`
  - *Linux:* `sudo apt install ffmpeg`

## 🚀 Installazione

1. **Clona il repository:**
   ```bash
   git clone [https://github.com/tuo-username/dera-transcriber.git](https://github.com/tuo-username/dera-transcriber.git)
   cd dera-transcriber


   Installa le dipendenze:
   pip install -r requirements.txt

   (Opzionale) Per supporto GPU NVIDIA:
   pip install torch --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install faster-whisper[cuda]

📖 Utilizzo
Avvia l'applicazione con:

python main.py
Seleziona un file (mp3, mp4, wav, mkv, ecc.).

Configura le impostazioni (Lingua, Modello, Punteggiatura).

Avvia l'elaborazione e attendi il risultato.

Copia o Salva il testo generato.

🤖 Vibe Coding & Trasparenza
Questo progetto è stato realizzato seguendo la filosofia del Vibe Coding: l'idea e l'orchestrazione logica sono umane, mentre la stesura del codice e l'ottimizzazione sono state supportate dall'Intelligenza Artificiale per garantire velocità di sviluppo e pulizia del codice.

📄 Licenza
Distribuito sotto Licenza MIT. Vedi LICENSE per maggiori informazioni.

Realizzato con ❤️ per semplificare la sbobinatura di interviste, lezioni e podcast.
