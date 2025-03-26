import os
import csv
import argparse
from pathlib import Path
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

def prepare_vivos(csv_file, save_path, split="train", fix_length=True, output_folder=None):
    """
    Chuẩn bị dữ liệu VIVOS từ file CSV và tạo các file .csv cho train/test.
    
    Args:
        csv_file (str): Đường dẫn tới file CSV của VIVOS (e.g., data/vivos/mixture/data.csv).
        save_path (str): Thư mục để lưu file .csv đầu ra (e.g., ./data/vivos).
        split (str): Tập dữ liệu cần xử lý (train hoặc test).
        fix_length (bool): Nếu True, đảm bảo các file âm thanh trong cùng một mẫu có cùng độ dài.
        output_folder (str): Thư mục để lưu các file âm thanh đã được xử lý.
    """
    # Đảm bảo thư mục lưu file CSV tồn tại
    os.makedirs(save_path, exist_ok=True)

    # Xác định thư mục mixtures dựa trên split (train/test)
    if split == "train":
        base_path_mixtures = "../../../data/vivos/mixture/data"
    else:  # split == "test"
        base_path_mixtures = "../../../data/vivos/mixture/test/data"

    # Sources luôn lấy từ ./data/vivos/train/waves (cho cả train và test)
    base_path_sources = "../../../data/vivos/train/waves"

    # Nếu fix_length=True và output_folder được cung cấp, tạo thư mục đầu ra
    if fix_length and output_folder:
        processed_output = os.path.join(output_folder, f"processed_{split}")
        os.makedirs(processed_output, exist_ok=True)
        os.makedirs(os.path.join(processed_output, "mix"), exist_ok=True)
        os.makedirs(os.path.join(processed_output, "s1"), exist_ok=True)
        os.makedirs(os.path.join(processed_output, "s2"), exist_ok=True)
    else:
        processed_output = None

    # Kiểm tra xem file CSV và các thư mục có tồn tại không
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File CSV does not exist: {csv_file}")
    if not os.path.exists(base_path_mixtures):
        raise FileNotFoundError(f"Folder mixtures does not exist: {base_path_mixtures}")
    if not os.path.exists(base_path_sources):
        raise FileNotFoundError(f"Folder sources does not exist: {base_path_sources}")

    # Đọc file CSV
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        data = list(reader)

    # Tạo file CSV đầu ra cho tập tương ứng (train/test)
    output_csv = os.path.join(save_path, f"vivos_{split}.csv")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["ID", "duration", "mix_wav", "s1_wav", "s2_wav"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in tqdm(enumerate(data), desc=f"Data processing {split}", total=len(data)):
            # Điều chỉnh path mixture
            mix_path = os.path.join(base_path_mixtures, row["mixture"].split("/")[-1])
            # Sources luôn lấy từ ./data/vivos/train/waves
            s1_path = os.path.join(base_path_sources, row["source1"])
            s2_path = os.path.join(base_path_sources, row["source2"])
            duration = row["duration"]

            # Kiểm tra xem các file âm thanh có tồn tại không
            if not os.path.exists(mix_path):
                print(f"Warning: File mixture does not exist: {mix_path}")
                continue
            if not os.path.exists(s1_path):
                print(f"Warning: File source1 does not exist: {s1_path}")
                continue
            if not os.path.exists(s2_path):
                print(f"Warning: File source2 does not exist: {s2_path}")
                continue

            # Nếu fix_length=True, đảm bảo độ dài các file là như nhau
            if fix_length and processed_output:
                # Đọc các file âm thanh
                mix_wav, mix_sr = torchaudio.load(mix_path)
                s1_wav, s1_sr = torchaudio.load(s1_path)
                s2_wav, s2_sr = torchaudio.load(s2_path)

                # Tìm độ dài nhỏ nhất
                min_length = min(mix_wav.shape[1], s1_wav.shape[1], s2_wav.shape[1])

                # Cắt các file về cùng độ dài
                mix_wav = mix_wav[:, :min_length]
                s1_wav = s1_wav[:, :min_length]
                s2_wav = s2_wav[:, :min_length]

                # Lưu các file âm thanh đã xử lý
                mix_filename = f"mix_{split}_{i:05d}.wav"
                s1_filename = f"s1_{split}_{i:05d}.wav"
                s2_filename = f"s2_{split}_{i:05d}.wav"

                mix_output_path = os.path.join(processed_output, "mix", mix_filename)
                s1_output_path = os.path.join(processed_output, "s1", s1_filename)
                s2_output_path = os.path.join(processed_output, "s2", s2_filename)

                torchaudio.save(mix_output_path, mix_wav, mix_sr)
                torchaudio.save(s1_output_path, s1_wav, s1_sr)
                torchaudio.save(s2_output_path, s2_wav, s2_sr)

                # Cập nhật đường dẫn cho CSV
                mix_path = mix_output_path
                s1_path = s1_output_path
                s2_path = s2_output_path

                # Cập nhật duration
                duration = min_length / mix_sr

            writer.writerow({
                "ID": f"mix_{split}_{i:05d}",
                "duration": duration,
                "mix_wav": mix_path,
                "s1_wav": s1_path,
                "s2_wav": s2_path,
            })

    print(f"Created CSV file at: {output_csv}")
    
    if fix_length and processed_output:
        print(f"Handled and saved audio files at: {processed_output}")
    
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu VIVOS cho SpeechBrain.")
    parser.add_argument("--csv_file", type=str, default="../../../data/vivos/mixture/annotation.csv", 
                        help="Đường dẫn tới file CSV của VIVOS.")
    parser.add_argument("--save_path", type=str, default="../../../data/vivos", 
                        help="Thư mục lưu file .csv.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], 
                        help="Tập dữ liệu cần xử lý (train hoặc test).")
    parser.add_argument("--fix_length", action="store_true", 
                        help="Đảm bảo các file âm thanh trong một mẫu có cùng độ dài.")
    parser.add_argument("--output_folder", type=str, default="../../../data/vivos/processed", 
                        help="Thư mục lưu các file âm thanh đã xử lý.")
    args = parser.parse_args()

    prepare_vivos(args.csv_file, args.save_path, args.split, args.fix_length, args.output_folder)