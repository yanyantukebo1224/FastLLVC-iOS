import os
import glob
import shutil
import argparse
import soundfile as sf
import numpy as np

def prepare_from_folders(src_dir: str, converted_dir: str, out_dir: str = "dataset/train"):
    """
    Pairs existing source audios with RVC converted audios into dataset/train/
    """
    os.makedirs(out_dir, exist_ok=True)
    src_files = sorted(glob.glob(os.path.join(src_dir, "**", "*.*"), recursive=True))
    src_files = [f for f in src_files if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
    
    conv_files = sorted(glob.glob(os.path.join(converted_dir, "**", "*.*"), recursive=True))
    conv_files = [f for f in conv_files if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
    
    print(f"[Pair] Found {len(src_files)} original audios and {len(conv_files)} converted audios.")
    
    count = 0
    for s_path in src_files:
        s_base = os.path.splitext(os.path.basename(s_path))[0]
        # Match with converted
        matched = None
        for c_path in conv_files:
            c_base = os.path.splitext(os.path.basename(c_path))[0]
            if s_base in c_base or c_base in s_base:
                matched = c_path
                break
        
        if matched:
            dst_orig = os.path.join(out_dir, f"{count:04d}_{s_base}_original.wav")
            dst_conv = os.path.join(out_dir, f"{count:04d}_{s_base}_converted.wav")
            
            # Read and resample to 16kHz mono
            from dataset_adapter import load_any_audio
            orig_wav = load_any_audio(s_path, 16000).numpy()
            conv_wav = load_any_audio(matched, 16000).numpy()
            
            sf.write(dst_orig, orig_wav, 16000)
            sf.write(dst_conv, conv_wav, 16000)
            count += 1
            print(f"  [Matched Pair {count}]: {os.path.basename(s_path)} <-> {os.path.basename(matched)}")
            
    print(f"\n🎉 Successfully prepared {count} pairs in '{out_dir}'!")


def main():
    parser = argparse.ArgumentParser(description="Prepare Paired Dataset for LLVC Fast-Distill")
    parser.add_argument("--src_dir", "-s", type=str, required=True, help="Original raw audio folder")
    parser.add_argument("--conv_dir", "-c", type=str, required=True, help="RVC converted audio folder")
    parser.add_argument("--out_dir", "-o", type=str, default="dataset/train", help="Output pair dataset folder")
    args = parser.parse_args()
    
    prepare_from_folders(args.src_dir, args.conv_dir, args.out_dir)

if __name__ == "__main__":
    main()
