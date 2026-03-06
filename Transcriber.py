import customtkinter as ctk
from tkinter import filedialog, messagebox
from faster_whisper import WhisperModel
import threading
import os
import re

# ── Punteggiatura: import opzionale con patch compatibilità ──────────────────
PUNCT_AVAILABLE = False
PunctuationModel = None

try:
    # Patch per transformers >= 4.31: rimpiazza grouped_entities con aggregation_strategy
    from transformers import pipelines
    _original_check = pipelines.token_classification.TokenClassificationPipeline._sanitize_parameters

    def _patched_sanitize(self, **kwargs):
        if "grouped_entities" in kwargs:
            val = kwargs.pop("grouped_entities")
            kwargs.setdefault(
                "aggregation_strategy", "simple" if val else "none"
            )
        return _original_check(self, **kwargs)

    pipelines.token_classification.TokenClassificationPipeline._sanitize_parameters = _patched_sanitize

    from deepmultilingualpunctuation import PunctuationModel
    PUNCT_AVAILABLE = True
except ImportError:
    PUNCT_AVAILABLE = False
except Exception:
    PUNCT_AVAILABLE = False


class TranscriberApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Dera — AI Audio Transcriber")
        self.geometry("960x820")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── Variabili di stato ────────────────────────────────────────────────
        self.selected_file   = ""
        self.is_processing   = False
        self.model_choice    = ctk.StringVar(value="small")
        self.language_choice = ctk.StringVar(value="it")
        self.export_mode     = ctk.StringVar(value="Pulito")
        self.device_choice   = ctk.StringVar(value="cpu")
        self.punct_enabled   = ctk.BooleanVar(value=PUNCT_AVAILABLE)
        self.vad_enabled     = ctk.BooleanVar(value=True)
        self.beam_size_var   = ctk.StringVar(value="5")

        # ── Dati modelli ──────────────────────────────────────────────────────
        self.model_data = {
            "tiny":     {"params": "39M",   "ram": "~75 MB",  "prec": "Scarsa",               "vel": "⚡⚡⚡⚡"},
            "base":     {"params": "74M",   "ram": "~150 MB", "prec": "Decente",              "vel": "⚡⚡⚡"},
            "small":    {"params": "244M",  "ram": "~500 MB", "prec": "Ottima (Consigliato)", "vel": "⚡⚡"},
            "medium":   {"params": "769M",  "ram": "~1.5 GB", "prec": "Eccellente",           "vel": "⚡"},
            "large-v3": {"params": "1550M", "ram": "~3.1 GB", "prec": "Stato dell'arte",      "vel": "🐢"},
        }

        # ── Lingue ────────────────────────────────────────────────────────────
        self.languages = {
            "Italiano":   "it",
            "Inglese":    "en",
            "Spagnolo":   "es",
            "Francese":   "fr",
            "Tedesco":    "de",
            "Automatico": None,
        }

        # ── Tab ───────────────────────────────────────────────────────────────
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)
        self.tabview.add("Trascrizione")
        self.tabview.add("Impostazioni")
        self.tabview.add("💡 Guida ottimizzazione")

        self.setup_transcription_tab()
        self.setup_settings_tab()
        self.setup_guide_tab()

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — TRASCRIZIONE
    # ══════════════════════════════════════════════════════════════════════════
    def setup_transcription_tab(self):
        tab = self.tabview.tab("Trascrizione")
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tab, text="Dera", font=("Helvetica", 32, "bold"),
                     text_color="#57bbff").pack(pady=(10, 0))
        ctk.CTkLabel(tab, text="AI Audio Transcriber",
                     font=("Helvetica", 13), text_color="gray").pack(pady=(0, 6))

        self.btn_select = ctk.CTkButton(
            tab, text="📂  Seleziona File Audio o Video",
            command=self.select_file, height=40)
        self.btn_select.pack(pady=10)

        self.label_file = ctk.CTkLabel(
            tab, text="Nessun file selezionato", text_color="gray")
        self.label_file.pack(pady=4)

        self.progress_bar = ctk.CTkProgressBar(tab, mode="indeterminate")
        self.progress_bar.pack(pady=12, padx=40, fill="x")
        self.progress_bar.set(0)

        self.label_status = ctk.CTkLabel(
            tab, text="", text_color="#57bbff", font=("Helvetica", 12))
        self.label_status.pack(pady=2)

        self.btn_start = ctk.CTkButton(
            tab, text="▶  Avvia Elaborazione",
            command=self.start_thread, state="disabled",
            fg_color="#1f6aa5", font=("Helvetica", 14, "bold"))
        self.btn_start.pack(pady=8)

        self.text_output = ctk.CTkTextbox(
            tab, height=300, font=("Consolas", 12))
        self.text_output.pack(pady=10, padx=15, fill="both", expand=True)

        btn_row = ctk.CTkFrame(tab, fg_color="transparent")
        btn_row.pack(pady=(0, 10))

        self.btn_copy = ctk.CTkButton(
            btn_row, text="📋  Copia testo",
            command=self.copy_to_clipboard,
            state="disabled", height=32, fg_color="#2d6a2d")
        self.btn_copy.pack(side="left", padx=10)

        self.btn_clear = ctk.CTkButton(
            btn_row, text="🗑  Pulisci",
            command=lambda: self.text_output.delete("1.0", "end"),
            height=32, fg_color="#6a2d2d")
        self.btn_clear.pack(side="left", padx=10)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — IMPOSTAZIONI
    # ══════════════════════════════════════════════════════════════════════════
    def setup_settings_tab(self):
        tab = self.tabview.tab("Impostazioni")
        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # ── 1. Modello ────────────────────────────────────────────────────────
        self._section_label(scroll, "1. Selezione Modello AI")
        self.model_menu = ctk.CTkOptionMenu(
            scroll, values=list(self.model_data.keys()),
            variable=self.model_choice)
        self.model_menu.pack(pady=5)

        table_frame = ctk.CTkFrame(scroll)
        table_frame.pack(pady=8, padx=20, fill="x")

        headers = ["Modello", "Parametri", "RAM", "Precisione", "Velocità"]
        for i, h in enumerate(headers):
            ctk.CTkLabel(table_frame, text=h,
                         font=("Helvetica", 11, "bold"),
                         text_color="#3a7ebf").grid(
                row=0, column=i, padx=12, pady=5)

        for r, (m_name, m_info) in enumerate(self.model_data.items(), 1):
            vals = [m_name, m_info["params"], m_info["ram"],
                    m_info["prec"], m_info["vel"]]
            for c, v in enumerate(vals):
                ctk.CTkLabel(table_frame, text=v).grid(
                    row=r, column=c, padx=12, pady=2)

        # ── 2. Lingua ─────────────────────────────────────────────────────────
        self._section_label(scroll, "2. Lingua del parlato")
        self.lang_menu = ctk.CTkOptionMenu(
            scroll, values=list(self.languages.keys()),
            command=self.update_lang_var)
        self.lang_menu.pack(pady=5)
        self.lang_menu.set("Italiano")
        ctk.CTkLabel(scroll,
                     text="💡 Specificare la lingua aumenta la velocità e riduce gli errori.",
                     text_color="gray", font=("Helvetica", 11)).pack(pady=(0, 5))

        # ── 3. Dispositivo ────────────────────────────────────────────────────
        self._section_label(scroll, "3. Dispositivo di calcolo")
        dev_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        dev_frame.pack()
        ctk.CTkRadioButton(dev_frame, text="CPU  (universale)",
                           variable=self.device_choice,
                           value="cpu").pack(side="left", padx=20)
        ctk.CTkRadioButton(dev_frame,
                           text="GPU CUDA  (10-20× più veloce — richiede NVIDIA + CUDA toolkit)",
                           variable=self.device_choice,
                           value="cuda").pack(side="left", padx=20)
        ctk.CTkLabel(scroll,
                     text="💡 Con GPU anche large-v3 è rapido. Su CPU preferisci small o base.",
                     text_color="gray", font=("Helvetica", 11)).pack(pady=(2, 8))

        # ── 4. Beam Size ──────────────────────────────────────────────────────
        self._section_label(scroll, "4. Beam Size  (precisione vs velocità)")
        beam_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        beam_frame.pack()
        for val, label in [("1", "1 — Velocissimo"), ("3", "3 — Bilanciato"),
                            ("5", "5 — Preciso (default)"), ("8", "8 — Massima precisione")]:
            ctk.CTkRadioButton(beam_frame, text=label,
                               variable=self.beam_size_var,
                               value=val).pack(side="left", padx=12)
        ctk.CTkLabel(scroll,
                     text="💡 Beam size alto = più ipotesi valutate → più preciso ma più lento.",
                     text_color="gray", font=("Helvetica", 11)).pack(pady=(2, 8))

        # ── 5. VAD ────────────────────────────────────────────────────────────
        self._section_label(scroll, "5. Filtro VAD  (Voice Activity Detection)")
        ctk.CTkSwitch(scroll,
                      text="Attiva VAD — salta automaticamente i silenzi",
                      variable=self.vad_enabled).pack(pady=5)
        ctk.CTkLabel(scroll,
                     text="💡 Consigliato sempre attivo: riduce i tempi del 20-40% su audio con pause.",
                     text_color="gray", font=("Helvetica", 11)).pack(pady=(0, 8))

        # ── 6. Punteggiatura ──────────────────────────────────────────────────
        self._section_label(scroll, "6. Ripristino Punteggiatura del parlato")

        if PUNCT_AVAILABLE:
            status_text  = "✅  deepmultilingualpunctuation installato e pronto"
            status_color = "#4CAF50"
        else:
            status_text  = "⚠️  Libreria non trovata — installa con:  pip install deepmultilingualpunctuation"
            status_color = "#FFA726"

        ctk.CTkLabel(scroll, text=status_text, text_color=status_color,
                     font=("Helvetica", 11)).pack(pady=4)

        self.punct_switch = ctk.CTkSwitch(
            scroll,
            text="Attiva punteggiatura AI  (aggiunge virgole, punti, punti interrogativi mancanti)",
            variable=self.punct_enabled,
            state="normal" if PUNCT_AVAILABLE else "disabled")
        self.punct_switch.pack(pady=5)
        ctk.CTkLabel(scroll,
                     text="💡 Post-processa il testo trascritto. Aggiunge ~5-10 sec di elaborazione extra.",
                     text_color="gray", font=("Helvetica", 11)).pack(pady=(0, 8))

        # ── 7. Esportazione ───────────────────────────────────────────────────
        self._section_label(scroll, "7. Modalità Esportazione")
        radio_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        radio_frame.pack()
        ctk.CTkRadioButton(radio_frame, text="Grezzo (con Timestamp)",
                           variable=self.export_mode,
                           value="Grezzo").pack(side="left", padx=20)
        ctk.CTkRadioButton(radio_frame, text="Pulito (solo Testo)",
                           variable=self.export_mode,
                           value="Pulito").pack(side="left", padx=20)
        ctk.CTkLabel(scroll,
                     text="💡 'Grezzo' è utile per sottotitoli e revisioni. 'Pulito' per documenti e note.",
                     text_color="gray", font=("Helvetica", 11)).pack(pady=(2, 14))

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — GUIDA OTTIMIZZAZIONE
    # ══════════════════════════════════════════════════════════════════════════
    def setup_guide_tab(self):
        tab = self.tabview.tab("💡 Guida ottimizzazione")
        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        guide_sections = [
            ("🚀  Velocità massima su CPU",
             "• Usa il modello tiny o base\n"
             "• Imposta Beam Size a 1 o 3\n"
             "• Attiva il filtro VAD\n"
             "• Specifica sempre la lingua (evita 'Automatico')\n"
             "• Chiudi altre applicazioni pesanti durante l'elaborazione"),

            ("🎯  Massima precisione",
             "• Usa il modello large-v3\n"
             "• Imposta Beam Size a 8\n"
             "• Attiva la punteggiatura AI\n"
             "• Usa una GPU NVIDIA se disponibile\n"
             "• Assicurati che l'audio sia pulito e senza rumori di fondo"),

            ("⚖️  Bilanciamento consigliato (impostazioni default)",
             "• Modello: small\n"
             "• Beam Size: 5\n"
             "• VAD: attivo\n"
             "• Dispositivo: CPU (o GPU se disponibile)\n"
             "• Lingua: specificata manualmente\n"
             "• Punteggiatura AI: attiva se installata"),

            ("🎙️  Qualità audio — il fattore più importante",
             "• Registra in ambienti silenziosi\n"
             "• Usa un microfono direzionale se possibile\n"
             "• Evita eco e riverbero (es. stanze vuote)\n"
             "• Il formato non è critico: mp3, wav, m4a vanno tutti bene\n"
             "• L'audio ideale per Whisper è 16kHz mono\n"
             "• Puoi pre-processare l'audio con Audacity per ridurre i rumori\n"
             "• Voce troppo bassa o troppo alta = più errori di trascrizione"),

            ("🖥️  Come abilitare la GPU NVIDIA",
             "1. Installa CUDA Toolkit 11.x o 12.x → https://developer.nvidia.com/cuda-downloads\n"
             "2. Installa cuDNN compatibile con la tua versione CUDA\n"
             "3. Esegui:  pip install faster-whisper[cuda]\n"
             "4. Verifica:  python -c \"import torch; print(torch.cuda.is_available())\"\n"
             "5. Seleziona 'GPU CUDA' nelle Impostazioni e avvia la trascrizione"),

            ("📦  Librerie aggiuntive consigliate",
             "deepmultilingualpunctuation  (punteggiatura automatica, supporta italiano)\n"
             "  → pip install deepmultilingualpunctuation\n\n"
             "ffmpeg  (necessario per file video mp4, mkv, ecc.)\n"
             "  → Windows: scarica da ffmpeg.org e aggiungi al PATH di sistema\n"
             "  → Linux:   sudo apt install ffmpeg\n"
             "  → Mac:     brew install ffmpeg\n\n"
             "torch con CUDA  (per accelerazione GPU)\n"
             "  → pip install torch --index-url https://download.pytorch.org/whl/cu118"),

            ("💾  Gestione della RAM",
             "• tiny / base:    funzionano con 4 GB di RAM\n"
             "• small:          consigliati almeno 4 GB liberi\n"
             "• medium:         almeno 6 GB liberi\n"
             "• large-v3:       almeno 8 GB (16 GB consigliati)\n\n"
             "I modelli vengono scaricati una volta sola e poi restano in cache:\n"
             "  ~/.cache/huggingface/hub/"),

            ("📁  Dove vengono salvati i file",
             "• Al termine si apre una finestra di salvataggio: scegli tu la cartella\n"
             "• Il suffisso _GREZZO.txt o _PULITO.txt viene aggiunto automaticamente\n"
             "• Puoi anche copiare il testo direttamente con il tasto 📋 Copia testo\n"
             "• I file sono in UTF-8: compatibili con Word, LibreOffice, Notepad++"),

            ("🔤  Come funziona la Punteggiatura AI",
             "• Usa il modello deepmultilingualpunctuation (basato su BERT multilingue)\n"
             "• Analizza il testo grezzo trascritto e aggiunge: , . ? !\n"
             "• Supporta italiano, inglese, spagnolo, francese, tedesco e altre lingue\n"
             "• Non modifica le parole, aggiunge solo la punteggiatura mancante\n"
             "• Richiede ~400 MB di RAM aggiuntiva al primo utilizzo (scarica il modello)\n"
             "• Particolarmente utile per interviste, podcast e riunioni"),
        ]

        for title, body in guide_sections:
            ctk.CTkLabel(scroll, text=title,
                         font=("Helvetica", 14, "bold"),
                         text_color="#57bbff",
                         anchor="w").pack(fill="x", pady=(14, 2), padx=5)
            box = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=8)
            box.pack(fill="x", padx=5, pady=(0, 4))
            ctk.CTkLabel(box, text=body,
                         font=("Consolas", 12),
                         justify="left",
                         anchor="w",
                         wraplength=860).pack(padx=15, pady=10, anchor="w")

    # ══════════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════════════
    def _section_label(self, parent, text: str):
        ctk.CTkLabel(parent, text=text,
                     font=("Helvetica", 15, "bold")).pack(pady=(16, 4))

    def update_lang_var(self, choice):
        value = self.languages[choice]
        self.language_choice.set(value if value is not None else "")

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Media", "*.mp3 *.wav *.m4a *.mp4 *.mkv *.flac *.ogg")])
        if file_path:
            self.selected_file = file_path
            self.label_file.configure(
                text=os.path.basename(file_path), text_color="#57bbff")
            self.btn_start.configure(state="normal")

    def copy_to_clipboard(self):
        text = self.text_output.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copiato", "Testo copiato negli appunti!")

    # ── Thread-safe helpers ───────────────────────────────────────────────────
    def _ui_append(self, text: str):
        self.after(0, self._do_append, text)

    def _do_append(self, text: str):
        self.text_output.insert("end", text)
        self.text_output.see("end")

    def _ui_status(self, text: str):
        self.after(0, self.label_status.configure, {"text": text})

    def _ui_reset_controls(self):
        self.after(0, self._do_reset_controls)

    def _do_reset_controls(self):
        self.is_processing = False
        self.btn_start.configure(state="normal")
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.label_status.configure(text="")

    # ══════════════════════════════════════════════════════════════════════════
    #  ELABORAZIONE
    # ══════════════════════════════════════════════════════════════════════════
    def start_thread(self):
        if not self.selected_file or self.is_processing:
            return
        self.is_processing = True
        self.btn_start.configure(state="disabled")
        self.btn_copy.configure(state="disabled")
        self.progress_bar.start()
        self.text_output.delete("1.0", "end")

        model_name = self.model_choice.get()
        lang_raw   = self.language_choice.get()
        lang       = lang_raw if lang_raw else None
        mode       = self.export_mode.get()
        device     = self.device_choice.get()
        beam_size  = int(self.beam_size_var.get())
        use_vad    = self.vad_enabled.get()
        use_punct  = self.punct_enabled.get() and PUNCT_AVAILABLE

        self._ui_append(
            f"> Modello: {model_name} | Lingua: {lang or 'Auto'} | "
            f"Export: {mode} | Dispositivo: {device.upper()} | "
            f"Beam: {beam_size} | VAD: {'✓' if use_vad else '✗'} | "
            f"Punteggiatura AI: {'✓' if use_punct else '✗'}\n"
        )
        self._ui_append("> Inizializzazione in corso…\n\n")

        thread = threading.Thread(
            target=self.run_transcription,
            args=(model_name, lang, mode, device, beam_size, use_vad, use_punct),
            daemon=True,
        )
        thread.start()

    def run_transcription(self, model_name, lang, mode, device,
                          beam_size, use_vad, use_punct):
        try:
            compute_type = "float16" if device == "cuda" else "int8"
            self._ui_status("⏳ Caricamento modello Whisper…")

            model = WhisperModel(model_name, device=device,
                                 compute_type=compute_type)

            self._ui_status("🎙️ Trascrizione in corso…")
            segments, info = model.transcribe(
                self.selected_file,
                beam_size=beam_size,
                language=lang,
                vad_filter=use_vad,
            )

            raw_lines   = []
            clean_parts = []

            for segment in segments:
                mins = int(segment.start // 60)
                secs = int(segment.start % 60)
                timestamp = f"[{mins}:{secs:02d}]"
                txt = segment.text.strip()
                raw_lines.append(f"{timestamp} {txt}")
                clean_parts.append(txt)
                self._ui_append(f"{timestamp} {txt}\n")

            joined_clean = " ".join(clean_parts)

            # ── Punteggiatura AI ──────────────────────────────────────────────
            if use_punct:
                self._ui_status("✍️ Applicazione punteggiatura AI…")
                punct_model  = PunctuationModel()
                joined_clean = punct_model.restore_punctuation(joined_clean)

                self._ui_append("\n── Testo con punteggiatura AI ──────────────\n")
                self._ui_append(joined_clean + "\n")

            # ── Output finale ─────────────────────────────────────────────────
            raw_content   = "\n".join(raw_lines)
            clean_content = self._format_clean_text(joined_clean)
            final_content = raw_content if mode == "Grezzo" else clean_content

            saved_path = self._save_to_disk(final_content, mode)

            if saved_path:
                self.after(0, messagebox.showinfo, "Completato",
                           f"Trascrizione completata in modalità '{mode}'!\n\n"
                           f"File salvato in:\n{saved_path}")
                self.after(0, self.btn_copy.configure, {"state": "normal"})
            else:
                self.after(0, messagebox.showwarning, "Annullato",
                           "Salvataggio annullato dall'utente.")

        except Exception as e:
            self.after(0, messagebox.showerror, "Errore",
                       f"Si è verificato un errore:\n{str(e)}")
        finally:
            self._ui_reset_controls()

    # ══════════════════════════════════════════════════════════════════════════
    #  UTILITÀ
    # ══════════════════════════════════════════════════════════════════════════
    def _format_clean_text(self, text: str) -> str:
        """A capo dopo ogni segno di punteggiatura finale (., !, ?, …)."""
        formatted = re.sub(r'([.!?…]+)\s+', r'\1\n', text)
        return formatted.strip()

    def _save_to_disk(self, content: str, mode: str):
        suffix       = "_GREZZO.txt" if mode == "Grezzo" else "_PULITO.txt"
        default_name = os.path.splitext(
            os.path.basename(self.selected_file))[0] + suffix

        save_path = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[("File di testo", "*.txt"), ("Tutti i file", "*.*")],
            title="Salva trascrizione",
        )
        if not save_path:
            return None

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
        return save_path


if __name__ == "__main__":
    app = TranscriberApp()
    app.mainloop()