import pandas as pd
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import os
import torch.nn.functional as F
from tqdm import tqdm
import argparse

# ─── CẤU HÌNH ĐƯỜNG DẪN ──────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--csv",      default="metrics.csv", help="Đường dẫn file CSV hiện tại")
_p.add_argument("--dir-hmi",  default="hmi",         help="Thư mục ảnh HMI")
_p.add_argument("--dir-deca", default="deca",        help="Thư mục ảnh DECA")
_p.add_argument("--dir-gt",   default="polyface_09042026", help="Thư mục Ground Truth")
ARGS = _p.parse_args()

# Thiết bị tăng tốc: Ưu tiên MPS (Apple Silicon M4) -> CUDA -> CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[INFO] Đang sử dụng thiết bị: {device}")

# ─── KHỞI TẠO MẠNG NHẬN DIỆN ─────────────────────────────────────────────
# Sử dụng InceptionResnetV1 đã được training trên tập VGGFace2 để nhận diện ID
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_csim(img_path, gt_path):
    """Tính toán Cosine Similarity giữa ảnh gen và ảnh Ground Truth."""
    if not img_path or not gt_path or not os.path.exists(img_path) or not os.path.exists(gt_path):
        return float("nan")
    
    try:
        # Load và tiền xử lý ảnh (FaceNet cần ảnh 160x160)
        def process_img(p):
            img = Image.open(p).convert('RGB').resize((160, 160))
            # Chuyển thành tensor và chuẩn hóa về [-1, 1]
            t = torch.tensor(list(img.getdata())).view(160, 160, 3).permute(2, 0, 1).float()
            return (t / 127.5 - 1.0).unsqueeze(0).to(device)

        with torch.no_grad():
            emb_gen = model(process_img(img_path))
            emb_gt  = model(process_img(gt_path))
            # Tính Cosine Similarity
            sim = F.cosine_similarity(emb_gen, emb_gt)
            return sim.item()
    except Exception as e:
        return float("nan")

def main():
    if not os.path.exists(ARGS.csv):
        print(f"[ERROR] Không tìm thấy file {ARGS.csv}")
        return

    # Đọc dữ liệu cũ
    df = pd.read_csv(ARGS.csv)
    print(f"[INFO] Đang xử lý {len(df)} đối tượng...")

    deca_csim_list = []
    ours_csim_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tính toán CSIM"):
        vid = row['vid']
        
        # 1. Xác định đường dẫn Ground Truth (ưu tiên C7)
        gt_path = None
        vid_gt_dir = os.path.join(ARGS.dir_gt, vid)
        if os.path.isdir(vid_gt_dir):
            for ext in ['.JPG', '.jpg', '.png']:
                cand = os.path.join(vid_gt_dir, f"{vid}_S002_L2_E01_C7{ext}")
                if os.path.exists(cand):
                    gt_path = cand
                    break
        
        # 2. Xác định đường dẫn ảnh gen
        hmi_path = os.path.join(ARGS.dir_hmi, f"{vid}.png") # hoặc .jpg tùy folder của bạn
        if not os.path.exists(hmi_path): hmi_path = hmi_path.replace(".png", ".jpg")
        
        deca_path = os.path.join(ARGS.dir_deca, f"{vid}.png")
        if not os.path.exists(deca_path): deca_path = deca_path.replace(".png", ".jpg")

        # 3. Tính toán
        deca_csim_list.append(get_csim(deca_path, gt_path))
        ours_csim_list.append(get_csim(hmi_path, gt_path))

    # 4. Ghép vào dataframe cũ
    df['deca_csim'] = deca_csim_list
    df['ours_csim'] = ours_csim_list
    
    # Tính delta
    df['delta_csim'] = df['ours_csim'] - df['deca_csim']

    # Lưu lại file (Lưu đè lên file cũ hoặc file mới)
    output_name = ARGS.csv # Bạn có thể đổi thành 'metrics_updated.csv' nếu muốn an toàn
    df.to_csv(output_name, index=False)
    
    print(f"\n[SUCCESS] Đã cập nhật CSIM vào {output_name}")
    print(f"CSIM Trung bình: DECA={df['deca_csim'].mean():.4f} | Ours={df['ours_csim'].mean():.4f}")

if __name__ == "__main__":
    main()