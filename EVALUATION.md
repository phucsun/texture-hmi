# Pipeline đánh giá — hướng dẫn chạy

Bộ script này sinh ra **Bảng 1**, **Bảng 2** và các số liệu còn thiếu trong `main.tex`, từ dữ liệu đã có sẵn trong repo.

---

## 1. Cài môi trường

```bash
python3 -m pip install -r requirements.txt
```

Lần chạy đầu sẽ tự tải trọng số của ba model (khoảng 300 MB): bộ detect landmark, LPIPS AlexNet, và FaceNet cho CSIM. Các lần sau dùng cache.

Kiểm tra nhanh:

```bash
python3 -c "import pyvista, trimesh, lpips, facenet_pytorch, face_alignment; print('OK')"
```

---

## 2. Chạy toàn bộ

```bash
python3 run_all.py
```

Một lệnh, chạy tuần tự 5 bước, **bỏ qua việc đã làm** nên ngắt giữa chừng rồi chạy lại được.

Thử nhanh trên 3 subject trước khi chạy đủ:

```bash
python3 run_all.py --limit 3
```

Khi đã có kết quả TESSERA:

```bash
python3 run_all.py --tessera-dir tessera_output
```

---

## 3. Chạy từng bước

### 3.1 Sinh texture baseline

```bash
python3 baselines.py
```

Đọc `results_C4_final/<subj>/<subj>.png` (texture baked của backbone), xác định vùng chưa được quan sát, rồi sinh ba baseline:

| File | Cách làm |
|---|---|
| `baselines/<vid>/mirroring.png` | lấy gương ngang qua trục đối xứng của atlas |
| `baselines/<vid>/telea.png` | `cv2.inpaint` thuật toán Telea |
| `baselines/<vid>/navier_stokes.png` | `cv2.inpaint` thuật toán Navier–Stokes |

Đồng thời ghi `baselines/coverage.csv` — tỉ lệ texel không được quan sát.

### 3.2 Render các góc nhìn

```bash
python3 render_views.py --methods backbone mirroring telea navier_stokes prior_transfer
```

Renderer **tất định**: camera dựng từ 68 landmark 3D của mesh, chiếu song song, khung hình chuẩn hoá theo khoảng cách liên đồng tử, không có chiếu sáng. Mọi phương pháp dùng **chung một mesh**, chỉ khác texture — nên mask bề mặt và landmark chiếu chỉ tính một lần cho mỗi (subject, view).

```
renders/<method>/<vid>_<view>.png    ảnh render
renders/_mask/<vid>_<view>.png       mask bề mặt
renders/_lmk/<vid>_<view>.json       68 landmark đã chiếu ra pixel
```

Kiểm tra căn chỉnh bằng mắt:

```bash
python3 render_views.py --limit 1 --methods backbone --debug-overlay --overwrite
```

Landmark xanh phải nằm đúng trên mắt, mũi, miệng ở **mọi** góc nhìn.

### 3.3 Đo metric

```bash
python3 eval_protocol.py --methods backbone mirroring telea navier_stokes prior_transfer
```

Sinh ra `results.csv`:

```
vid,view,method,psnr,ssim,lpips,csim,region,q_pred
```

### 3.4 Kiểm tra và sinh bảng LaTeX

```bash
python3 check_results.py results.csv --latex
```

In ra Bảng 1, Bảng 2, kiểm định Wilcoxon, tương quan $Q(\theta)$, danh sách failure case, và các dòng LaTeX dán thẳng vào `main.tex`.

---

## 4. Protocol đo — khớp với §4.1 của bài

**Đưa hai ảnh về cùng khung bằng cách dựng, không phải căn chỉnh hậu kỳ.**

1. **Ảnh render** — camera dựng từ landmark 3D của chính mesh đó, nên đầu nằm ở vị trí giống hệt nhau với mọi subject và mọi phương pháp.
2. **Ảnh tham chiếu** — detect 68 landmark 2D, rồi biến đổi similarity đưa về đúng vị trí của landmark đã chiếu từ mesh. Dùng 51 điểm trong (bỏ đường hàm 0–16, vì đường hàm đổi theo góc nhìn).
3. **Vùng đo** — giao của *mask bề mặt đã render* và *bao lồi landmark của ảnh tham chiếu*. Loại tóc, quần áo, nền — những thứ mesh không biểu diễn.

Metric tính **chỉ trong mask**: PSNR trên các pixel thuộc mask; SSIM lấy trung bình bản đồ SSIM trong mask; LPIPS và CSIM tính trên ảnh đã zero ngoài mask, cắt theo bbox.

Kết quả trên full crop (để phụ lục):

```bash
python3 eval_protocol.py --methods ... --region full_crop --out results_fullcrop.csv
```

---

## 5. Thêm dòng TESSERA

Khi thuật toán chạy xong, đặt texture theo một trong ba dạng:

```
tessera_output/<vid>/texture.png
tessera_output/<vid>/deca_texture.png
tessera_output/<vid>.png
```

rồi:

```bash
python3 render_views.py --methods tessera --tessera-dir tessera_output
python3 eval_protocol.py --methods backbone mirroring telea navier_stokes prior_transfer tessera
python3 check_results.py results.csv --latex
```

Để có mục **Predicted Quality** (§4.2), thêm cột `q_pred` vào `results.csv` cho các dòng `method=tessera` — giá trị $Q(\theta)$ mà TESSERA dự báo cho cặp (subject, view) đó. `check_results.py` sẽ tự tính tương quan Spearman.

---

## 6. Dòng DECA + albedo prior

Baseline này cần **albedo atlas** của DECA, thứ không có sẵn trong repo (`results_C4_final/` chỉ có texture baked). Phải chạy lại DECA với tuỳ chọn xuất albedo, đặt thành `<thư mục>/<vid>.png`, rồi:

```bash
python3 run_all.py --albedo-dir deca_albedo
```

Không có thì bỏ dòng đó và ghi rõ trong bài; sáu dòng còn lại vẫn đủ.

---

## 7. Chi phí

| Bước | Thời gian ước tính (74 subject) |
|---|---|
| `baselines.py` | vài phút |
| `render_views.py`, 6 phương pháp × 7 view | ~30–40 phút |
| `eval_protocol.py` | ~15–25 phút (nhanh hơn nhiều nếu có GPU) |

Phần chậm nhất của bước đo là detect landmark trên ảnh tham chiếu độ phân giải cao; kết quả được cache theo `(vid, view)` nên **dùng chung cho mọi phương pháp**.

---

## 8. Cấu hình cần rà lại

Trong `lib_face.py`:

```python
VIEWS = {
    "C1":  dict(yaw=-90.0, pitch=  0.0),
    "C4":  dict(yaw=-35.0, pitch=  0.0),   # camera đầu vào
    ...
}
```

Các góc này tôi **ước lượng bằng cách quan sát ảnh**, không phải lấy từ thông số rig chính thức của V-Face. Nếu có tài liệu rig thì sửa lại — nó ảnh hưởng trực tiếp tới Bảng 2 và tới lập luận về sự bất đối xứng giữa phía đã quan sát và phía bị che.

Hai hằng số khác:

```python
IMG_SIZE = 512      # cạnh ảnh render
IOD_FRAC = 0.26     # khoảng cách liên đồng tử chiếm bao nhiêu phần chiều cao ảnh
```

---

## 9. Về con số độ phủ

`baselines.py` báo tỉ lệ texel **bị để trống trong atlas baked** (khoảng 8–10%). Đây **không phải** con số 40–58% mà bài báo nêu — con số đó là tỉ lệ bề mặt **không nhìn thấy được từ camera lúc chụp**, một đại lượng khác. DECA lấp phần lớn atlas bằng nội dung nội suy thay vì để trống, nên hai cách đo cho kết quả rất khác nhau.

Muốn đo đúng đại lượng trong bài thì phải tính khả kiến theo hình học: với mỗi texel, lấy điểm 3D và pháp tuyến của nó rồi kiểm tra xem có hướng về phía camera lúc chụp hay không. Cần thêm một script riêng; ô `\PH{40--58\%}` nên để trống cho tới lúc đó.
