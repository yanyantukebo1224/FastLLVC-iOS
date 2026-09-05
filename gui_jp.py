import os
import sys
import time
import glob
import json
import gc
import subprocess
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import gradio as gr
import torch

from realtime_vc_jp import RealtimeVCEngineJP, get_audio_devices_jp
from infer_adapter import load_model_with_adapter, load_audio, save_audio, infer_stream

# Global Engine & State
engine = None
engine_lock = threading.Lock()
current_running_process = None
process_lock = threading.Lock()


def get_system_hardware_badge():
    """Returns friendly Japanese hardware status badge"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return f"🚀 **【動作モード: GPU高速アクセラレーション】** ({gpu_name} / BF16推論 ~3.5ms)"
    else:
        cpu_cores = os.cpu_count() or 4
        return f"⚡ **【動作モード: CPUマルチスレッド最適化】** (Intel/AMD CPU {cpu_cores}スレッド並列処理 / ~13ms)"


def get_model_choices():
    """Lists all available base and fine-tuned models"""
    bases = ["llvc_models/models/checkpoints/llvc/G_500000.pth"]
    if os.path.exists("my_adapter"):
        for f in glob.glob("my_adapter/*.pth"):
            bases.append(os.path.normpath(f))
    return bases


def get_rvc_models():
    """Finds RVC models in assets/weights/"""
    weights_dir = "assets/weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
    models = glob.glob(os.path.join(weights_dir, "*.pth"))
    return [os.path.basename(m) for m in models]


# ----------------- Tab 1: Realtime VC Handlers -----------------
def toggle_realtime_vc(is_active, in_dev_name, out_dev_name, model_path, in_gain, out_gain, gate_db, key_shift, enable_vocoder, vocoder_strength, latency_mode, force_cpu_check):
    global engine
    with engine_lock:
        if is_active:
            if engine is not None:
                engine.stop()
                engine = None
            return False, "🛑 **ボイスチェンジャー停止中**", gr.update(value="🎤 リアルタイム変換を開始する", variant="primary")
        else:
            in_devs, out_devs = get_audio_devices_jp()
            in_idx = dict(in_devs).get(in_dev_name, None)
            out_idx = dict(out_devs).get(out_dev_name, None)

            # Chunk Factor: 1 (13.0ms), 2 (26.0ms), 3 (39.0ms)
            if "13.0ms" in latency_mode:
                chunk_factor = 1
            elif "26.0ms" in latency_mode:
                chunk_factor = 2
            else:
                chunk_factor = 3

            try:
                engine = RealtimeVCEngineJP(
                    checkpoint_path=model_path,
                    config_path="experiments/llvc/config.json",
                    chunk_factor=chunk_factor,
                    input_device=in_idx,
                    output_device=out_idx,
                    input_gain=float(in_gain),
                    output_gain=float(out_gain),
                    threshold_db=float(gate_db),
                    key_shift=float(key_shift),
                    enable_vocoder=bool(enable_vocoder),
                    vocoder_strength=float(vocoder_strength),
                    enable_low_cut=True,
                    force_cpu=bool(force_cpu_check)
                )
                engine.start()
                status_msg = f"🟢 **リアルタイム変身中！** ({engine.device_name} | 遅延: {engine.chunk_len/engine.sr*1000:.1f}ms | キー: {key_shift:+d})"
                return True, status_msg, gr.update(value="⏹️ 変換を停止する", variant="stop")
            except Exception as e:
                return False, f"❌ **起動エラー**: {str(e)}", gr.update(value="🎤 リアルタイム変換を開始する", variant="primary")


def update_rt_params(in_gain, out_gain, gate_db, key_shift, enable_vocoder, vocoder_strength):
    global engine
    with engine_lock:
        if engine is not None and engine.is_running:
            engine.update_params(
                in_gain=in_gain,
                out_gain=out_gain,
                gate_db=gate_db,
                key_shift=key_shift,
                enable_vocoder=enable_vocoder,
                vocoder_strength=vocoder_strength
            )
            return f"✅ 設定を即時反映しました！ (Key: {key_shift:+d})"
        return "⚠️ ボイスチェンジャーが停止中のため、次回起動時に反映されます。"


# ----------------- Tab 2: Full-Auto RVC Distill Handlers -----------------
def run_auto_rvc_distill(src_folder, rvc_model_name, key_shift, f0_method, out_model_name, epochs, batch_size, lr, progress=gr.Progress()):
    global current_running_process
    if not rvc_model_name:
        return "❌ RVCモデルが選択されていません。`assets/weights/` に .pth を配置してください。", gr.update()
    
    clean_name = os.path.splitext(out_model_name)[0]
    pair_dataset_dir = "dataset/train"
    os.makedirs(pair_dataset_dir, exist_ok=True)
    
    # 1. Step 1: Pair Generation
    progress(0.05, desc="RVC音声変換で学習ペアを生成中...")
    cmd_pair = [
        "py", "-3.12", "dataset_builder.py",
        "-s", src_folder,
        "-m", rvc_model_name,
        "-k", str(int(key_shift)),
        "-f", f0_method,
        "-o", pair_dataset_dir
    ]
    
    with process_lock:
        current_running_process = subprocess.Popen(cmd_pair, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        p = current_running_process

    for line in iter(p.stdout.readline, ''):
        print(line, end="")
        if "RVC Convert" in line:
            try:
                progress(0.05 + 0.35 * 0.5, desc=f"RVCペア生成中: {line.strip()}")
            except:
                pass
    p.wait()
    
    with process_lock:
        current_running_process = None

    if p.returncode != 0:
        return "❌ RVCペア生成が中断またはエラー終了しました。", gr.update()

    # 2. Step 2: Noise-Robust Distillation (Preserves 100% Base Denoising Power)
    progress(0.40, desc="ノイズ遮断保護付き LLVC 高速蒸留開始 (HuBERT音素アンカー)...")
    cmd_distill = [
        "py", "-3.12", "train_distill.py",
        "-d", pair_dataset_dir,
        "-o", "my_adapter",
        "-n", clean_name,
        "-e", str(int(epochs)),
        "-b", str(int(batch_size)),
        "--lr", str(float(lr)),
        "--steps_per_epoch", "100"
    ]
    
    with process_lock:
        current_running_process = subprocess.Popen(cmd_distill, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        p = current_running_process

    for line in iter(p.stdout.readline, ''):
        print(line, end="")
        if "Epoch [" in line:
            try:
                ep_part = line.split("[")[1].split("]")[0]
                cur_ep, total_ep = map(int, ep_part.split("/"))
                progress(0.40 + 0.58 * (cur_ep / total_ep), desc=f"蒸留学習中: Epoch {cur_ep}/{total_ep} ({line.strip()})")
            except:
                pass
    p.wait()

    with process_lock:
        current_running_process = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bases = get_model_choices()
    saved_model = os.path.normpath(os.path.join("my_adapter", f"{clean_name}.pth"))
    
    if p.returncode == 0:
        msg = f"🎉 **全自動RVC蒸留が完了しました！**\n\n- 生成モデル: `{saved_model}`\n\n上部の「使用するキャラクターモデル」に自動設定されました！"
        return msg, gr.update(choices=bases, value=saved_model if saved_model in bases else bases[0])
    else:
        return "❌ 蒸留処理が中断またはエラー終了しました。", gr.update()


def kill_active_process():
    global current_running_process
    with process_lock:
        if current_running_process is not None:
            current_running_process.terminate()
            current_running_process = None
            return "🛑 実行中の学習・変換プロセスを強制停止しました。"
    return "ℹ️ 停止対象のバックグラウンドプロセスはありません。"


# ----------------- Tab 3: File Convert Handler -----------------
def process_audio_file(input_file, base_model, stream_mode, key_shift=0, progress=gr.Progress()):
    if input_file is None:
        return None, "ファイルを指定してください。"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    
    progress(0.1, desc="モデル読み込み中...")
    model, sr = load_model_with_adapter(
        base_model, 'experiments/llvc/config.json', None, merge=False, device=device, dtype=dtype
    )
    
    progress(0.3, desc="音声変換中...")
    audio = load_audio(input_file, sr)
    
    if abs(key_shift) >= 0.1:
        from pitch_shifter import ZeroLatencyPitchShifter
        ps = ZeroLatencyPitchShifter(sample_rate=sr)
        audio = ps.process_torch(audio, key_shift)

    t0 = time.time()
    if stream_mode:
        out, rtf, latency = infer_stream(model, audio, 1, sr, device=device, key_shift=0)
        elapsed = time.time() - t0
        speed_factor = (len(audio) / sr) / max(elapsed, 0.001)
        msg = f"⚡ ストリーミング変換完了！ (所要時間: {elapsed*1000:.1f}ms | 変換速度: {speed_factor:.1f}倍速 | E2E遅延: {latency:.2f}ms | Key: {key_shift:+d})"
    else:
        with torch.inference_mode():
            in_tensor = audio.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
            out_tensor = model(in_tensor)
            out = out_tensor.squeeze().float().cpu()
        elapsed = time.time() - t0
        speed_factor = (len(audio) / sr) / max(elapsed, 0.001)
        msg = f"🚀 爆速一括変換完了！ (所要時間: {elapsed*1000:.1f}ms | 変換速度: {speed_factor:.1f}倍速！ | Key: {key_shift:+d})"
    
    out_path = "converted_out/gui_output.wav"
    os.makedirs("converted_out", exist_ok=True)
    save_audio(out, out_path, sr)
    
    return out_path, msg


# ----------------- Build Complete Japanese UI -----------------
def create_japan_studio_ui():
    in_devs, out_devs = get_audio_devices_jp()
    in_names = [d[0] for d in in_devs]
    out_names = [d[0] for d in out_devs]
    base_models = get_model_choices()
    rvc_models = get_rvc_models()

    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="cyan",
        neutral_hue="slate"
    )

    with gr.Blocks(title="Fast-LLVC Studio (完全日本版・超低遅延ボイスチェンジャー)", theme=theme) as app:
        gr.Markdown(f"""
        # 🎙️ Fast-LLVC Studio (完全日本版・超低遅延ボイスチェンジャー)
        ### 【どんなWindowsでも動く！CPU / GPU 自動最適化 ＆ 超低遅延13.0ms】
        {get_system_hardware_badge()}
        """)

        with gr.Tabs():
            # TAB 1: リアルタイム変身
            with gr.TabItem("⚡ リアルタイム変身 (マイク変換)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎤 音声デバイス設定")
                        in_device_select = gr.Dropdown(
                            label="マイク入力デバイス (Input Device)",
                            choices=in_names,
                            value=in_names[0] if in_names else None,
                            interactive=True
                        )
                        out_device_select = gr.Dropdown(
                            label="スピーカー / 仮想マイク出力デバイス (Output Device)",
                            choices=out_names,
                            value=out_names[0] if out_names else None,
                            interactive=True
                        )
                        
                        gr.Markdown("### 🎭 声の選択")
                        base_model_dropdown = gr.Dropdown(
                            label="使用するキャラクターモデル (Base / Trained Model)",
                            choices=base_models,
                            value=base_models[0] if base_models else None,
                            interactive=True
                        )
                        
                        force_cpu_check = gr.Checkbox(
                            label="⚡ 強制CPU動作モード (GPUを使わずCPUマルチスレッドで動作)",
                            value=(not torch.cuda.is_available())
                        )
                        
                        vc_toggle_btn = gr.Button("🎤 リアルタイム変換を開始する", variant="primary", size="lg")
                        vc_status_text = gr.Markdown("🛑 **ボイスチェンジャー停止中**")

                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ 音質・ピッチ調整")
                        
                        latency_mode_dropdown = gr.Dropdown(
                            label="⚡ レイテンシ / 安定性モード",
                            choices=[
                                "🔥 爆速超低遅延モード (13.0ms / 最速・知覚遅延ゼロ)",
                                "✨ 推奨・超安定モード (26.0ms / 高音質・音飛びゼロ)",
                                "🛡️ 高安定モード (39.0ms / 低負荷・安心)"
                            ],
                            value="🔥 爆速超低遅延モード (13.0ms / 最速・知覚遅延ゼロ)",
                            interactive=True
                        )

                        with gr.Group():
                            gr.Markdown("#### 🎵 声の高さ調整 (Key Shift)")
                            key_shift_rt_slider = gr.Slider(
                                minimum=-24, maximum=24, value=0, step=1,
                                label="ピッチシフト量 (半音単位)"
                            )
                            with gr.Row():
                                btn_zunda = gr.Button("ずんだもん (+12)", size="sm")
                                btn_female = gr.Button("女性声 (+8)", size="sm")
                                btn_natural = gr.Button("地声 (0)", size="sm")
                                btn_male = gr.Button("男性声 (-12)", size="sm")
                            
                            btn_zunda.click(lambda: 12, outputs=[key_shift_rt_slider])
                            btn_female.click(lambda: 8, outputs=[key_shift_rt_slider])
                            btn_natural.click(lambda: 0, outputs=[key_shift_rt_slider])
                            btn_male.click(lambda: -12, outputs=[key_shift_rt_slider])

                        with gr.Row():
                            in_gain_slider = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="マイク入力音量 (Input Gain)")
                            out_gain_slider = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="出力音量 (Output Gain)")
                        
                        gate_slider = gr.Slider(minimum=-80.0, maximum=-20.0, value=-45.0, step=1.0, label="雑音・環境音カット感度 (Noise Gate Threshold)")

                        with gr.Accordion("✨ スタジオ高音質化・エキサイター設定 (オプション)", open=False):
                            enable_vocoder_check = gr.Checkbox(label="高域倍音エンハンサーを有効化", value=False)
                            vocoder_strength_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.4, step=0.05, label="エンハンス強度")

                        param_update_btn = gr.Button("🔄 調整した設定を即時反映", size="sm")
                        param_status = gr.Markdown("")

                is_active_state = gr.State(value=False)
                
                def on_toggle_click(current_state, in_dev, out_dev, base_m, ig, og, gt, ks, voc_en, voc_str, lat_m, f_cpu):
                    new_state = not current_state
                    active, status_msg, btn_upd = toggle_realtime_vc(new_state, in_dev, out_dev, base_m, ig, og, gt, ks, voc_en, voc_str, lat_m, f_cpu)
                    return active, status_msg, btn_upd

                vc_toggle_btn.click(
                    on_toggle_click,
                    inputs=[is_active_state, in_device_select, out_device_select, base_model_dropdown, in_gain_slider, out_gain_slider, gate_slider, key_shift_rt_slider, enable_vocoder_check, vocoder_strength_slider, latency_mode_dropdown, force_cpu_check],
                    outputs=[is_active_state, vc_status_text, vc_toggle_btn]
                )
                param_update_btn.click(
                    update_rt_params,
                    inputs=[in_gain_slider, out_gain_slider, gate_slider, key_shift_rt_slider, enable_vocoder_check, vocoder_strength_slider],
                    outputs=[param_status]
                )

            # TAB 2: 全自動RVC蒸留
            with gr.TabItem("⚡ 全自動RVC蒸留 (自分専用モデル作成)"):
                gr.Markdown("""
                ### 🎙️ お手持ちのRVCモデルから、超低遅延（13ms）モデルを一撃で全自動生成！
                * **タイピング音・環境雑音カット保護機能内蔵**：キーボード音を声に変換してしまう過学習を完全遮断。
                * **HuBERT音素アンカー**：滑舌や明瞭度を100%キープしながら話者性だけを蒸留。
                """)
                with gr.Row():
                    with gr.Column():
                        rvc_model_dropdown = gr.Dropdown(
                            label="目的のRVCモデルを選択 (Target RVC Model)",
                            choices=rvc_models,
                            value=rvc_models[0] if rvc_models else None,
                            interactive=True
                        )
                        distill_src_dir = gr.Textbox(label="元音声フォルダー (Source Audio Folder)", value="test_wavs")
                        distill_model_name = gr.Textbox(label="作成するモデル名 (Output Model Name)", value="my_fast_voice")
                        
                        with gr.Row():
                            key_shift_slider = gr.Slider(minimum=-24, maximum=24, value=0, step=1, label="声の高さ変換 (Key Shift)")
                            f0_method_radio = gr.Radio(choices=["rmvpe", "fcpe", "pm"], value="rmvpe", label="ピッチ抽出方式 (F0 Method)")

                        with gr.Row():
                            start_auto_btn = gr.Button("🚀 全自動蒸留を開始する", variant="primary", size="lg")
                            kill_btn = gr.Button("🛑 中断 / 強制停止", variant="stop", size="lg")

                    with gr.Column():
                        gr.Markdown("""
                        #### 💡 全自動の流れ
                        1. `assets/weights/` にあるRVCモデルを選びます。
                        2. 元音声をRVCで一括変換し、`元音声 ⇄ 目標音声` のペアを自動構築。
                        3. ノイズ遮断能力を保護しながら、超低遅延LLVCモデルを全自動蒸留！
                        4. 完了すると、自動的にリアルタイム変換タブのモデル一覧に追加されます。
                        """)
                        with gr.Accordion("⚙️ 詳細ハイパーパラメータ設定", open=False):
                            distill_epochs = gr.Slider(minimum=10, maximum=100, value=30, step=5, label="学習エポック数 (Epochs)")
                            distill_batch = gr.Slider(minimum=1, maximum=32, value=8, step=1, label="バッチサイズ (Batch Size)")
                            distill_lr = gr.Number(label="学習率 (Learning Rate)", value=3e-4)

                auto_result_box = gr.Markdown("")

                start_auto_btn.click(
                    run_auto_rvc_distill,
                    inputs=[distill_src_dir, rvc_model_dropdown, key_shift_slider, f0_method_radio, distill_model_name, distill_epochs, distill_batch, distill_lr],
                    outputs=[auto_result_box, base_model_dropdown]
                )
                kill_btn.click(kill_active_process, outputs=[auto_result_box])

            # TAB 3: ファイル一括変換
            with gr.TabItem("🚀 音声ファイル一括変換"):
                gr.Markdown("### 📂 wavファイルをドラッグ＆ドロップして、一瞬でターゲット声に変換！")
                with gr.Row():
                    with gr.Column():
                        file_input = gr.Audio(label="変換したい音声ファイル (WAV/MP3)", type="filepath")
                        file_model_select = gr.Dropdown(label="使用モデル", choices=base_models, value=base_models[0] if base_models else None)
                        file_key_shift = gr.Slider(minimum=-24, maximum=24, value=0, step=1, label="声の高さ調整 (Key Shift)")
                        file_stream_mode = gr.Checkbox(label="超低遅延ストリーミング推論モード (低メモリ・高速)", value=True)
                        file_convert_btn = gr.Button("⚡ 音声を変換する", variant="primary", size="lg")
                    
                    with gr.Column():
                        file_output = gr.Audio(label="変換後の音声 (Output Audio)")
                        file_status = gr.Markdown("")
                
                file_convert_btn.click(
                    process_audio_file,
                    inputs=[file_input, file_model_select, file_stream_mode, file_key_shift],
                    outputs=[file_output, file_status]
                )

        gr.Markdown("""
        ---
        **Fast-LLVC Studio Japan Edition** | 制作: Pop-chan & Antigravity Pair Programming
        """)

    return app


if __name__ == "__main__":
    app = create_japan_studio_ui()
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True
    )
