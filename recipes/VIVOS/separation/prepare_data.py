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

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        data = list(reader)

    output_csv = os.path.join(save_path, f"vivos_{split}.csv")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["ID", "duration", "mix_wav", "s1_wav", "s2_wav"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in tqdm(enumerate(data), desc=f"Xử lý dữ liệu {split}", total=len(data)):
            mix_path = os.path.join(base_path_mixtures, row["mixture"].split("/")[-1])
            s1_path = os.path.join(base_path_sources, row["source1"])
            s2_path = os.path.join(base_path_sources, row["source2"])

            if fix_length and processed_output:
                # Load file gốc và file nguồn
                mix_wav_orig, mix_sr = torchaudio.load(mix_path)
                s1_wav, s1_sr = torchaudio.load(s1_path)
                s2_wav, s2_sr = torchaudio.load(s2_path)

                # print(mix_wav_orig.shape[1], mix_sr, mix_wav)

                # Đồng bộ tần số lấy mẫu
                if s1_sr != mix_sr:
                    s1_wav = torchaudio.transforms.Resample(s1_sr, mix_sr)(s1_wav)
                    s1_sr = mix_sr
                if s2_sr != mix_sr:
                    s2_wav = torchaudio.transforms.Resample(s2_sr, mix_sr)(s2_wav)
                    s2_sr = mix_sr

                # Debug: In độ dài trước khi xử lý
                # print(f"Trước xử lý - ID {i}: mix_orig: {mix_wav_orig.shape[1]} ({mix_wav_orig.shape[1]/mix_sr:.2f}s), "
                #     f"s1: {s1_wav.shape[1]} ({s1_wav.shape[1]/mix_sr:.2f}s), s2: {s2_wav.shape[1]} ({s2_wav.shape[1]/mix_sr:.2f}s)")

                # Tìm độ dài tối đa giữa mix_wav, s1_wav, và s2_wav
                max_length = max(mix_wav_orig.shape[1], s1_wav.shape[1], s2_wav.shape[1])
                
                # Đệm tất cả các file lên max_length
                if mix_wav_orig.shape[1] < max_length:
                    mix_wav_orig = F.pad(mix_wav_orig, (0, max_length - mix_wav_orig.shape[1]))
                    
                if s1_wav.shape[1] < max_length:
                    s1_wav = F.pad(s1_wav, (0, max_length - s1_wav.shape[1]))
                    
                if s2_wav.shape[1] < max_length:
                    s2_wav = F.pad(s2_wav, (0, max_length - s2_wav.shape[1]))

                # Debug: In độ dài sau khi đệm
                # print(f"Sau đệm - ID {i}: mix: {mix_wav_orig.shape[1]} ({mix_wav_orig.shape[1]/mix_sr:.2f}s), "
                #     f"s1: {s1_wav.shape[1]} ({s1_wav.shape[1]/mix_sr:.2f}s), s2: {s2_wav.shape[1]} ({s2_wav.shape[1]/mix_sr:.2f}s)")

                # Lấy mix_wav là tổng của s1_wav và s2_wav (tạo lại mix từ 2 nguồn)
                mix_wav = s1_wav + s2_wav

                # Cắt tất cả về cùng kích thước (đảm bảo)
                target_length = max_length
                mix_wav = mix_wav[:, :target_length]
                s1_wav = s1_wav[:, :target_length]
                s2_wav = s2_wav[:, :target_length]

                # Debug: In độ dài ngay trước khi lưu
                # print(f"Trước khi lưu - ID {i}: mix: {mix_wav.shape[1]} ({mix_wav.shape[1]/mix_sr:.2f}s), "
                #     f"s1: {s1_wav.shape[1]} ({s1_wav.shape[1]/mix_sr:.2f}s), s2: {s2_wav.shape[1]} ({s2_wav.shape[1]/mix_sr:.2f}s)")

                # Lưu các file
                mix_output_path = os.path.join(processed_output, "mix", f"mix_{split}_{i:05d}.wav")
                s1_output_path = os.path.join(processed_output, "s1", f"s1_{split}_{i:05d}.wav")
                s2_output_path = os.path.join(processed_output, "s2", f"s2_{split}_{i:05d}.wav")

                # Kiểm tra là mono hay stereo và đảm bảo lưu đúng định dạng
                if mix_wav.shape[0] > 1:
                    mix_wav = mix_wav[0:1]  # Lấy kênh đầu tiên nếu là stereo
                if s1_wav.shape[0] > 1:
                    s1_wav = s1_wav[0:1]
                if s2_wav.shape[0] > 1:
                    s2_wav = s2_wav[0:1]

                # Lưu với các thông số không thay đổi
                torchaudio.save(mix_output_path, mix_wav, mix_sr)
                torchaudio.save(s1_output_path, s1_wav, mix_sr)  # Đảm bảo cùng sample rate
                torchaudio.save(s2_output_path, s2_wav, mix_sr)  # Đảm bảo cùng sample rate

                # Kiểm tra lại file đã lưu
                mix_wav_saved, mix_sr_saved = torchaudio.load(mix_output_path)
                s1_wav_saved, s1_sr_saved = torchaudio.load(s1_output_path)
                s2_wav_saved, s2_sr_saved = torchaudio.load(s2_output_path)
                
                # print(f"Sau khi lưu - ID {i}: mix: {mix_wav_saved.shape[1]} ({mix_wav_saved.shape[1]/mix_sr_saved:.2f}s), "
                #       f"s1: {s1_wav_saved.shape[1]} ({s1_wav_saved.shape[1]/s1_sr_saved:.2f}s), "
                #       f"s2: {s2_wav_saved.shape[1]} ({s2_wav_saved.shape[1]/s2_sr_saved:.2f}s)")

                # Kiểm tra xem các file có cùng độ dài không
                if not (mix_wav_saved.shape[1] == s1_wav_saved.shape[1] == s2_wav_saved.shape[1]):
                    print(f"CẢNH BÁO: Các file ID {i} có độ dài khác nhau sau khi lưu!")

                # Cập nhật duration
                duration = target_length / mix_sr

                writer.writerow({
                    "ID": f"mix_{split}_{i:05d}",
                    "duration": duration,
                    "mix_wav": mix_output_path,
                    "s1_wav": s1_output_path,
                    "s2_wav": s2_output_path,
                })
            else:
                # Chỉ lưu thông tin vào CSV mà không xử lý file
                mix_wav, mix_sr = torchaudio.load(mix_path)
                duration = mix_wav.shape[1] / mix_sr
                
                writer.writerow({
                    "ID": f"mix_{split}_{i:05d}",
                    "duration": duration,
                    "mix_wav": mix_path,
                    "s1_wav": s1_path,
                    "s2_wav": s2_path,
                })

    print(f"Đã tạo file CSV tại: {output_csv}")
    if fix_length and processed_output:
        print(f"Đã xử lý và lưu các file âm thanh tại: {processed_output}")
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu VIVOS cho SpeechBrain.")
    parser.add_argument("--csv_file", type=str, default="../../../data/vivos/mixture/test/annotation.csv", 
                        help="Đường dẫn tới file CSV của VIVOS.")
    parser.add_argument("--save_path", type=str, default="../../../data/vivos", 
                        help="Thư mục lưu file .csv.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], 
                        help="Tập dữ liệu cần xử lý (train hoặc test).")
    parser.add_argument("--fix_length", action="store_true", 
                        help="Đảm bảo các file âm thanh trong một mẫu có cùng độ dài.")
    parser.add_argument("--output_folder", type=str, default="../../../data/vivos/processed", 
                        help="Thư mục lưu các file âm thanh đã xử lý.")
    args = parser.parse_args()

    prepare_vivos(args.csv_file, args.save_path, args.split, args.fix_length, args.output_folder)