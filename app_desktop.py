import os
import sys
import time
import glob
import json
import threading
import queue
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import sounddevice as sd
import torch

from realtime_vc_jp import RealtimeVCEngineJP, get_audio_devices_jp
from infer_adapter import load_model_with_adapter, load_audio, save_audio, infer_stream


class FastLLVCStudioDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fast-LLVC Studio (完全日本版・超低遅延ボイスチェンジャー)")
        self.root.geometry("820x720")
        self.root.minsize(760, 640)

        # Dark Theme Palette
        self.bg_color = "#1e1e2e"
        self.card_bg = "#28283d"
        self.accent_color = "#00adb5"
        self.text_color = "#eeeeee"
        self.subtext_color = "#aaaaaa"
        self.stop_color = "#ff5757"
        self.root.configure(bg=self.bg_color)

        # Engine State
        self.engine = None
        self.is_vc_running = False
        self.current_proc = None

        # Style Configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background=self.bg_color, foreground=self.text_color)
        self.style.configure("TNotebook", background=self.bg_color, borderwidth=0)
        self.style.configure("TNotebook.Tab", background=self.card_bg, foreground=self.text_color, padding=[15, 8], font=("Meiryo", 10, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", self.accent_color)], foreground=[("selected", "#ffffff")])
        self.style.configure("TCombobox", fieldbackground=self.card_bg, background=self.card_bg, foreground=self.text_color)
        self.style.configure("TProgressbar", thickness=12, troughcolor=self.bg_color, background=self.accent_color)

        self._init_ui()
        self._poll_ui_updates()

    def _init_ui(self):
        # Header Banner
        header_frame = tk.Frame(self.root, bg=self.card_bg, pady=12, padx=20)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        title_lbl = tk.Label(header_frame, text="🎙️ Fast-LLVC Studio", font=("Meiryo", 16, "bold"), fg=self.accent_color, bg=self.card_bg)
        title_lbl.pack(side=tk.LEFT)

        # Hardware Badge
        if torch.cuda.is_available():
            hw_text = f"🚀 GPU: {torch.cuda.get_device_name(0)} (BF16最速)"
            hw_fg = "#00ffcc"
        else:
            hw_text = f"⚡ CPU: {os.cpu_count() or 4}スレッド並列処理 (~13ms)"
            hw_fg = "#ffcc00"
        
        self.hw_lbl = tk.Label(header_frame, text=hw_text, font=("Meiryo", 9, "bold"), fg=hw_fg, bg=self.card_bg)
        self.hw_lbl.pack(side=tk.RIGHT)

        # Main Tab Control
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 1: Realtime VC
        self.tab_vc = tk.Frame(self.notebook, bg=self.bg_color, padx=15, pady=15)
        self.notebook.add(self.tab_vc, text="⚡ リアルタイム変身")
        self._build_vc_tab()

        # Tab 2: Auto RVC Distill
        self.tab_distill = tk.Frame(self.notebook, bg=self.bg_color, padx=15, pady=15)
        self.notebook.add(self.tab_distill, text="🎙️ 全自動RVCモデル作成")
        self._build_distill_tab()

        # Tab 3: File Convert
        self.tab_file = tk.Frame(self.notebook, bg=self.bg_color, padx=15, pady=15)
        self.notebook.add(self.tab_file, text="🚀 音声ファイル変換")
        self._build_file_tab()

        # Status Bar
        self.status_bar = tk.Label(self.root, text="準備完了 | Fast-LLVC Studio Japan Edition", font=("Meiryo", 9), fg=self.subtext_color, bg=self.card_bg, anchor="w", padx=10, pady=4)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_vc_tab(self):
        in_devs, out_devs = get_audio_devices_jp()
        self.in_dev_map = {d[0]: d[1] for d in in_devs}
        self.out_dev_map = {d[0]: d[1] for d in out_devs}
        in_names = list(self.in_dev_map.keys())
        out_names = list(self.out_dev_map.keys())

        # Top Split Frame
        top_frame = tk.Frame(self.tab_vc, bg=self.bg_color)
        top_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column: Devices & Model
        left_col = tk.LabelFrame(top_frame, text=" 🎤 デバイス ＆ キャラクター選択 ", font=("Meiryo", 10, "bold"), fg=self.accent_color, bg=self.card_bg, padx=15, pady=10)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        tk.Label(left_col, text="マイク入力デバイス:", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 9)).pack(anchor="w")
        self.in_dev_combo = ttk.Combobox(left_col, values=in_names, state="readonly")
        if in_names: self.in_dev_combo.current(0)
        self.in_dev_combo.pack(fill=tk.X, pady=(2, 10))

        tk.Label(left_col, text="スピーカー / 仮想マイク出力:", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 9)).pack(anchor="w")
        self.out_dev_combo = ttk.Combobox(left_col, values=out_names, state="readonly")
        if out_names: self.out_dev_combo.current(0)
        self.out_dev_combo.pack(fill=tk.X, pady=(2, 10))

        tk.Label(left_col, text="使用するキャラクターモデル:", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 9)).pack(anchor="w")
        models = self._get_model_list()
        self.model_combo = ttk.Combobox(left_col, values=models, state="readonly")
        if models: self.model_combo.current(0)
        self.model_combo.pack(fill=tk.X, pady=(2, 10))

        self.force_cpu_var = tk.BooleanVar(value=(not torch.cuda.is_available()))
        cpu_chk = tk.Checkbutton(left_col, text="⚡ 強制CPUモードで動作", variable=self.force_cpu_var, bg=self.card_bg, fg=self.text_color, selectcolor=self.bg_color, font=("Meiryo", 9))
        cpu_chk.pack(anchor="w", pady=5)

        # Big Start / Stop Button
        self.btn_vc_toggle = tk.Button(left_col, text="🎤 リアルタイム変換を開始する", font=("Meiryo", 12, "bold"), bg=self.accent_color, fg="#ffffff", activebackground="#00838f", activeforeground="#ffffff", relief=tk.FLAT, pady=10, command=self._toggle_vc)
        self.btn_vc_toggle.pack(fill=tk.X, pady=(15, 5))

        self.vc_status_lbl = tk.Label(left_col, text="🛑 停止中", font=("Meiryo", 9, "bold"), fg="#aaaaaa", bg=self.card_bg)
        self.vc_status_lbl.pack()

        # Right Column: Pitch & Sound Tuning
        right_col = tk.LabelFrame(top_frame, text=" 🎛️ 音質 ＆ 声の高さ調整 ", font=("Meiryo", 10, "bold"), fg=self.accent_color, bg=self.card_bg, padx=15, pady=10)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        # Pitch Preset Buttons
        pitch_box = tk.LabelFrame(right_col, text=" 🎵 声の高さ (Key Shift) ", font=("Meiryo", 9, "bold"), fg=self.text_color, bg=self.card_bg, padx=10, pady=8)
        pitch_box.pack(fill=tk.X, pady=(0, 10))

        self.pitch_val_lbl = tk.Label(pitch_box, text="±0 半音 (地声)", font=("Meiryo", 10, "bold"), fg=self.accent_color, bg=self.card_bg)
        self.pitch_val_lbl.pack()

        self.pitch_scale = tk.Scale(pitch_box, from_=-24, to=24, orient=tk.HORIZONTAL, showvalue=0, bg=self.card_bg, fg=self.text_color, highlightthickness=0, command=self._on_pitch_change)
        self.pitch_scale.set(0)
        self.pitch_scale.pack(fill=tk.X, pady=4)

        preset_frame = tk.Frame(pitch_box, bg=self.card_bg)
        preset_frame.pack(fill=tk.X)
        tk.Button(preset_frame, text="ずんだもん (+12)", font=("Meiryo", 8), bg="#393e46", fg="#ffffff", relief=tk.FLAT, command=lambda: self._set_pitch(12)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(preset_frame, text="女性声 (+8)", font=("Meiryo", 8), bg="#393e46", fg="#ffffff", relief=tk.FLAT, command=lambda: self._set_pitch(8)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(preset_frame, text="地声 (0)", font=("Meiryo", 8), bg="#393e46", fg="#ffffff", relief=tk.FLAT, command=lambda: self._set_pitch(0)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(preset_frame, text="男性声 (-12)", font=("Meiryo", 8), bg="#393e46", fg="#ffffff", relief=tk.FLAT, command=lambda: self._set_pitch(-12)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # Gain & Noise Gate
        tk.Label(right_col, text="マイク入力音量:", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 9)).pack(anchor="w")
        self.in_gain_scale = tk.Scale(right_col, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, bg=self.card_bg, fg=self.text_color, highlightthickness=0, command=self._apply_runtime_params)
        self.in_gain_scale.set(1.0)
        self.in_gain_scale.pack(fill=tk.X)

        tk.Label(right_col, text="出力音量:", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 9)).pack(anchor="w")
        self.out_gain_scale = tk.Scale(right_col, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, bg=self.card_bg, fg=self.text_color, highlightthickness=0, command=self._apply_runtime_params)
        self.out_gain_scale.set(1.0)
        self.out_gain_scale.pack(fill=tk.X)

        tk.Label(right_col, text="雑音・打鍵音カット感度 (Noise Gate dB):", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 9)).pack(anchor="w")
        self.gate_scale = tk.Scale(right_col, from_=-80.0, to=-20.0, resolution=1.0, orient=tk.HORIZONTAL, bg=self.card_bg, fg=self.text_color, highlightthickness=0, command=self._apply_runtime_params)
        self.gate_scale.set(-45.0)
        self.gate_scale.pack(fill=tk.X)

        # Latency mode
        tk.Label(right_col, text="遅延モード:", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 9)).pack(anchor="w", pady=(8, 0))
        self.lat_combo = ttk.Combobox(right_col, values=["🔥 爆速超低遅延モード (13.0ms)", "✨ 推奨・超安定モード (26.0ms)", "🛡️ 高安定モード (39.0ms)"], state="readonly")
        self.lat_combo.current(0)
        self.lat_combo.pack(fill=tk.X, pady=2)

        # Bottom VU Meter & Live Stats
        meter_frame = tk.LabelFrame(self.tab_vc, text=" 📊 リアルタイム音声モニター ＆ 遅延状況 ", font=("Meiryo", 9, "bold"), fg=self.accent_color, bg=self.card_bg, padx=15, pady=8)
        meter_frame.pack(fill=tk.X, pady=(10, 0))

        m_grid = tk.Frame(meter_frame, bg=self.card_bg)
        m_grid.pack(fill=tk.X)

        tk.Label(m_grid, text="マイク入力:", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 8)).grid(row=0, column=0, sticky="w")
        self.in_meter = ttk.Progressbar(m_grid, orient=tk.HORIZONTAL, length=240, mode="determinate")
        self.in_meter.grid(row=0, column=1, padx=10, pady=3, sticky="ew")

        tk.Label(m_grid, text="変換後出力:", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 8)).grid(row=1, column=0, sticky="w")
        self.out_meter = ttk.Progressbar(m_grid, orient=tk.HORIZONTAL, length=240, mode="determinate")
        self.out_meter.grid(row=1, column=1, padx=10, pady=3, sticky="ew")

        self.live_stats_lbl = tk.Label(m_grid, text="推論時間: -- ms | E2E遅延: -- ms", fg="#00ffcc", bg=self.card_bg, font=("Meiryo", 9, "bold"))
        self.live_stats_lbl.grid(row=0, column=2, rowspan=2, padx=15)
        m_grid.columnconfigure(1, weight=1)

    def _build_distill_tab(self):
        rvc_models = self._get_rvc_models()

        container = tk.LabelFrame(self.tab_distill, text=" 🎙️ お手持ちのRVCモデルから一撃で低遅延モデルを作成 ", font=("Meiryo", 10, "bold"), fg=self.accent_color, bg=self.card_bg, padx=20, pady=15)
        container.pack(fill=tk.BOTH, expand=True)

        tk.Label(container, text="1. 目的のRVCモデルを選択 (assets/weights/):", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 9)).pack(anchor="w")
        self.rvc_combo = ttk.Combobox(container, values=rvc_models, state="readonly")
        if rvc_models: self.rvc_combo.current(0)
        self.rvc_combo.pack(fill=tk.X, pady=(2, 10))

        tk.Label(container, text="2. 元音声フォルダー (WAV/MP3):", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 9)).pack(anchor="w")
        src_f_frame = tk.Frame(container, bg=self.card_bg)
        src_f_frame.pack(fill=tk.X, pady=(2, 10))
        self.src_dir_entry = tk.Entry(src_f_frame, bg=self.bg_color, fg=self.text_color, font=("Meiryo", 9), insertbackground=self.text_color)
        self.src_dir_entry.insert(0, "test_wavs")
        self.src_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(src_f_frame, text="参照...", bg="#393e46", fg="#ffffff", command=self._browse_src_dir).pack(side=tk.RIGHT, padx=(5, 0))

        tk.Label(container, text="3. 出力モデル名:", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 9)).pack(anchor="w")
        self.out_name_entry = tk.Entry(container, bg=self.bg_color, fg=self.text_color, font=("Meiryo", 9), insertbackground=self.text_color)
        self.out_name_entry.insert(0, "my_custom_voice")
        self.out_name_entry.pack(fill=tk.X, pady=(2, 15))

        btn_row = tk.Frame(container, bg=self.card_bg)
        btn_row.pack(fill=tk.X, pady=5)

        self.btn_start_distill = tk.Button(btn_row, text="🚀 全自動RVC蒸留を開始する", font=("Meiryo", 11, "bold"), bg=self.accent_color, fg="#ffffff", relief=tk.FLAT, pady=8, command=self._start_distill)
        self.btn_start_distill.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.btn_stop_distill = tk.Button(btn_row, text="🛑 中断", font=("Meiryo", 11, "bold"), bg=self.stop_color, fg="#ffffff", relief=tk.FLAT, pady=8, state=tk.DISABLED, command=self._stop_distill)
        self.btn_stop_distill.pack(side=tk.RIGHT, padx=(5, 0))

        self.distill_prog = ttk.Progressbar(container, orient=tk.HORIZONTAL, mode="indeterminate")
        self.distill_prog.pack(fill=tk.X, pady=(15, 5))

        self.distill_log = tk.Text(container, height=8, bg=self.bg_color, fg="#00ffcc", font=("Consolas", 8), insertbackground="#ffffff")
        self.distill_log.pack(fill=tk.BOTH, expand=True)

    def _build_file_tab(self):
        container = tk.LabelFrame(self.tab_file, text=" 🚀 音声ファイル一括変換 ", font=("Meiryo", 10, "bold"), fg=self.accent_color, bg=self.card_bg, padx=20, pady=15)
        container.pack(fill=tk.BOTH, expand=True)

        tk.Label(container, text="変換したい音声ファイル:", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 9)).pack(anchor="w")
        f_row = tk.Frame(container, bg=self.card_bg)
        f_row.pack(fill=tk.X, pady=(2, 10))
        self.file_entry = tk.Entry(f_row, bg=self.bg_color, fg=self.text_color, font=("Meiryo", 9), insertbackground=self.text_color)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(f_row, text="ファイル選択...", bg="#393e46", fg="#ffffff", command=self._browse_audio_file).pack(side=tk.RIGHT, padx=(5, 0))

        tk.Label(container, text="使用モデル:", fg=self.text_color, bg=self.card_bg, font=("Meiryo", 9)).pack(anchor="w")
        self.file_model_combo = ttk.Combobox(container, values=self._get_model_list(), state="readonly")
        if self._get_model_list(): self.file_model_combo.current(0)
        self.file_model_combo.pack(fill=tk.X, pady=(2, 15))

        self.btn_convert_file = tk.Button(container, text="⚡ 音声を一括変換する", font=("Meiryo", 11, "bold"), bg=self.accent_color, fg="#ffffff", relief=tk.FLAT, pady=8, command=self._convert_file)
        self.btn_convert_file.pack(fill=tk.X, pady=5)

        self.file_status_lbl = tk.Label(container, text="", fg="#00ffcc", bg=self.card_bg, font=("Meiryo", 9))
        self.file_status_lbl.pack(pady=10)

    # ----------------- Helper Functions -----------------
    def _get_model_list(self):
        models = ["llvc_models/models/checkpoints/llvc/G_500000.pth"]
        if os.path.exists("my_adapter"):
            for f in glob.glob("my_adapter/*.pth"):
                models.append(os.path.normpath(f))
        return models

    def _get_rvc_models(self):
        weights_dir = "assets/weights"
        os.makedirs(weights_dir, exist_ok=True)
        models = glob.glob(os.path.join(weights_dir, "*.pth"))
        return [os.path.basename(m) for m in models]

    def _set_pitch(self, val):
        self.pitch_scale.set(val)
        self._on_pitch_change(val)

    def _on_pitch_change(self, val):
        val = int(float(val))
        if val == 0: text = "±0 半音 (地声)"
        elif val > 0: text = f"+{val} 半音 (高音化)"
        else: text = f"{val} 半音 (低音化)"
        self.pitch_val_lbl.config(text=text)
        self._apply_runtime_params()

    def _apply_runtime_params(self, *args):
        if self.engine is not None and self.is_vc_running:
            self.engine.update_params(
                in_gain=self.in_gain_scale.get(),
                out_gain=self.out_gain_scale.get(),
                gate_db=self.gate_scale.get(),
                key_shift=self.pitch_scale.get(),
                enable_vocoder=False,
                vocoder_strength=0.4
            )

    def _toggle_vc(self):
        if self.is_vc_running:
            # Stop
            if self.engine:
                self.engine.stop()
                self.engine = None
            self.is_vc_running = False
            self.btn_vc_toggle.config(text="🎤 リアルタイム変換を開始する", bg=self.accent_color)
            self.vc_status_lbl.config(text="🛑 停止中", fg="#aaaaaa")
            self.status_bar.config(text="ボイスチェンジャーを停止しました。")
            self.in_meter["value"] = 0
            self.out_meter["value"] = 0
            self.live_stats_lbl.config(text="推論時間: -- ms | E2E遅延: -- ms")
        else:
            # Start
            in_name = self.in_dev_combo.get()
            out_name = self.out_dev_combo.get()
            in_idx = self.in_dev_map.get(in_name, None)
            out_idx = self.out_dev_map.get(out_name, None)
            model_path = self.model_combo.get()

            lat_text = self.lat_combo.get()
            if "13.0ms" in lat_text: chunk_factor = 1
            elif "26.0ms" in lat_text: chunk_factor = 2
            else: chunk_factor = 3

            try:
                self.engine = RealtimeVCEngineJP(
                    checkpoint_path=model_path,
                    config_path="experiments/llvc/config.json",
                    chunk_factor=chunk_factor,
                    input_device=in_idx,
                    output_device=out_idx,
                    input_gain=float(self.in_gain_scale.get()),
                    output_gain=float(self.out_gain_scale.get()),
                    threshold_db=float(self.gate_scale.get()),
                    key_shift=float(self.pitch_scale.get()),
                    enable_vocoder=False,
                    enable_low_cut=True,
                    force_cpu=bool(self.force_cpu_var.get()),
                    on_latency_update=self._on_latency_update,
                    on_volume_update=self._on_volume_update
                )
                self.engine.start()
                self.is_vc_running = True
                self.btn_vc_toggle.config(text="⏹️ 変換を停止する", bg=self.stop_color)
                self.vc_status_lbl.config(text="🟢 リアルタイム変身中！", fg="#00ffcc")
                self.status_bar.config(text=f"稼働中: {self.engine.device_name} (遅延 {self.engine.chunk_len/self.engine.sr*1000:.1f}ms)")
            except Exception as e:
                messagebox.showerror("起動エラー", f"ボイスチェンジャーの起動に失敗しました:\n{str(e)}")

    def _on_latency_update(self, infer_ms, rtf):
        # Called from background thread
        e2e = (self.engine.chunk_len / self.engine.sr * 1000.0) + infer_ms if self.engine else 0
        self.root.after(0, lambda: self.live_stats_lbl.config(text=f"推論: {infer_ms:.1f}ms | 遅延: {e2e:.1f}ms ({rtf:.1f}x)"))

    def _on_volume_update(self, in_rms, out_rms):
        in_val = min(100, int(in_rms * 400))
        out_val = min(100, int(out_rms * 400))
        self.root.after(0, lambda: self._set_meters(in_val, out_val))

    def _set_meters(self, in_val, out_val):
        self.in_meter["value"] = in_val
        self.out_meter["value"] = out_val

    def _poll_ui_updates(self):
        self.root.after(100, self._poll_ui_updates)

    # ----------------- Tab 2: Distill Methods -----------------
    def _browse_src_dir(self):
        d = filedialog.askdirectory(title="元音声フォルダーを選択")
        if d:
            self.src_dir_entry.delete(0, tk.END)
            self.src_dir_entry.insert(0, d)

    def _start_distill(self):
        rvc_m = self.rvc_combo.get()
        src_d = self.src_dir_entry.get()
        out_n = self.out_name_entry.get()

        if not rvc_m:
            messagebox.showwarning("モデル未選択", "RVCモデルを選択してください。")
            return

        self.btn_start_distill.config(state=tk.DISABLED)
        self.btn_stop_distill.config(state=tk.NORMAL)
        self.distill_prog.start(10)
        self.distill_log.delete(1.0, tk.END)
        self.distill_log.insert(tk.END, f"[開始] RVCペア生成 & 蒸留学習を開始します...\n")

        threading.Thread(target=self._run_distill_worker, args=(src_d, rvc_m, out_n), daemon=True).start()

    def _run_distill_worker(self, src_d, rvc_m, out_n):
        pair_dir = "dataset/train"
        clean_name = os.path.splitext(out_n)[0]

        # Step 1: Pair Build
        cmd1 = ["py", "-3.12", "dataset_builder.py", "-s", src_d, "-m", rvc_m, "-o", pair_dir]
        self.current_proc = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(self.current_proc.stdout.readline, ''):
            if not line: break
            self.root.after(0, lambda l=line: self._append_log(l))
        self.current_proc.wait()

        if self.current_proc.returncode != 0:
            self.root.after(0, self._on_distill_done, False, "ペア生成でエラーが発生しました。")
            return

        # Step 2: Distill
        cmd2 = ["py", "-3.12", "train_distill.py", "-d", pair_dir, "-o", "my_adapter", "-n", clean_name, "-e", "30", "-b", "8"]
        self.current_proc = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(self.current_proc.stdout.readline, ''):
            if not line: break
            self.root.after(0, lambda l=line: self._append_log(l))
        self.current_proc.wait()

        success = (self.current_proc.returncode == 0)
        self.root.after(0, self._on_distill_done, success, f"my_adapter/{clean_name}.pth")

    def _append_log(self, text):
        self.distill_log.insert(tk.END, text)
        self.distill_log.see(tk.END)

    def _stop_distill(self):
        if self.current_proc:
            self.current_proc.terminate()
            self.current_proc = None
        self._on_distill_done(False, "ユーザーにより中止されました。")

    def _on_distill_done(self, success, result_msg):
        self.distill_prog.stop()
        self.btn_start_distill.config(state=tk.NORMAL)
        self.btn_stop_distill.config(state=tk.DISABLED)
        if success:
            messagebox.showinfo("蒸留完了", f"全自動RVC蒸留が完了しました！\nモデル: {result_msg}")
            # Refresh model lists
            m_list = self._get_model_list()
            self.model_combo["values"] = m_list
            self.file_model_combo["values"] = m_list
            self.model_combo.set(result_msg)
        else:
            messagebox.showwarning("処理終了", result_msg)

    # ----------------- Tab 3: File Convert Methods -----------------
    def _browse_audio_file(self):
        f = filedialog.askopenfilename(title="音声ファイルを選択", filetypes=[("Audio Files", "*.wav;*.mp3;*.flac;*.m4a")])
        if f:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, f)

    def _convert_file(self):
        f_path = self.file_entry.get()
        m_path = self.file_model_combo.get()
        if not f_path or not os.path.exists(f_path):
            messagebox.showwarning("ファイル未選択", "変換する音声ファイルを指定してください。")
            return
        
        self.file_status_lbl.config(text="⏳ 音声変換中...")
        self.btn_convert_file.config(state=tk.DISABLED)

        threading.Thread(target=self._run_file_convert_worker, args=(f_path, m_path), daemon=True).start()

    def _run_file_convert_worker(self, f_path, m_path):
        out_path, msg = process_audio_file(f_path, m_path, stream_mode=True, key_shift=0)
        self.root.after(0, lambda: self._on_file_done(out_path, msg))

    def _on_file_done(self, out_path, msg):
        self.btn_convert_file.config(state=tk.NORMAL)
        self.file_status_lbl.config(text=f"✅ {msg}\n保存先: {out_path}")
        messagebox.showinfo("変換完了", f"音声変換が完了しました！\n\n{msg}\n保存先: {out_path}")


def main():
    root = tk.Tk()
    app = FastLLVCStudioDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
