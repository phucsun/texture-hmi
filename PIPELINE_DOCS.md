# Tài liệu Pipeline: Chuyển Texture BFM → DECA

## Mục lục

1. [Bài toán](#1-bài-toán)
2. [Dữ liệu đầu vào](#2-dữ-liệu-đầu-vào)
3. [Công nghệ sử dụng](#3-công-nghệ-sử-dụng)
4. [V1 — align_only.py](#4-v1--align_onlypy)
5. [V2 — patch_uv.py](#5-v2--patch_uvpy)
6. [V3 — align_and_transfer (hard-gate)](#6-v3--align_and_transfer-phiên-bản-hard-gate)
7. [V4 — align_and_transfer (final)](#7-v4--align_and_transfer-phiên-bản-cuối)
8. [Tại sao V4 hoạt động được](#8-tại-sao-v4-hoạt-động-được)

---

## 1. Bài toán

### Mô tả

Có hai mô hình 3D khuôn mặt:

| Model | Vertices | Faces | UV Atlas | Ý nghĩa |
|---|---|---|---|---|
| **BFM** (Basel Face Model) | ~53,149 | ~106,466 | Có sẵn (1024×1024 PNG) | Model nguồn, đã có texture |
| **DECA** | ~5,023 | ~9,976 | Có sẵn nhưng trống | Model đích, cần tạo texture |

**Vấn đề cốt lõi:** Hai mesh có **topology hoàn toàn khác nhau** — số đỉnh khác nhau, thứ tự đỉnh khác nhau, cấu trúc UV atlas khác nhau. Không thể copy texture từ BFM sang DECA bằng cách đơn giản gán giá trị pixel, vì không có ánh xạ 1-1 giữa các đỉnh.

### Tại sao không thể copy trực tiếp

```
BFM texture atlas (1024×1024)
    │
    │  pixel (u=0.3, v=0.7) = màu da ở vùng trán
    │  → tọa độ UV này chỉ có nghĩa trong UV space của BFM
    │  → nếu dùng cùng (u,v) trong UV space của DECA
    │    → ra vị trí hoàn toàn khác trên khuôn mặt DECA
    ↓
DECA texture atlas (1024×1024) — trống
    → cần tìm: pixel (u',v') trong DECA tương ứng với điểm 3D nào
    → điểm 3D đó gần BFM triangle nào
    → lấy màu từ BFM triangle đó
```

Ngoài ra, hai mesh ban đầu **không nằm ở cùng vị trí trong không gian 3D** — BFM có thể dùng đơn vị mm còn DECA dùng đơn vị chuẩn hóa, hoặc khác convention trục tọa độ (Y-up vs Z-up). Phải căn chỉnh trước khi transfer.

---

## 2. Dữ liệu đầu vào

### File mesh

```
bfm_to_flame/
├── uv_idm.obj          # BFM mesh (nguồn) — 53k vertices, 106k faces, có UV
├── uv_idm.png          # BFM texture atlas 1024×1024 (RGB, màu da)
├── deca.obj            # DECA mesh (đích) — 5k vertices, 9k faces, có UV
├── deca_texture.png    # DECA texture atlas (trống hoặc placeholder)
└── head_template.obj   # Template đầu (dùng để tham chiếu)
```

### File landmarks

```
landmarks/
├── uv_idm.mat          # 68 landmarks BFM — file MATLAB (.mat), 121 MB
│                       # Có thể lưu dạng tọa độ 3D (68×3) hoặc vertex indices (68,)
└── deca.npy            # 68 landmarks DECA — NumPy dict, 31 KB
                        # Lưu dưới dạng face_idx + barycentric coords
                        # keys: full_lmk_faces_idx, full_lmk_bary_coords
```

**Tại sao có 68 landmarks?** Đây là chuẩn facial landmark detection (iBUG 300-W), bao gồm: đường viền mặt (17), lông mày (10), mắt (12), mũi (9), miệng (20). Đủ để xác định hình dạng và tư thế khuôn mặt.

**Tại sao DECA lưu landmarks dưới dạng barycentric?** Vì DECA là model tham số — vị trí các đỉnh thay đổi theo từng khuôn mặt cụ thể. Barycentric coordinates trên một triangle luôn cho ra vị trí đúng bất kể mesh deform thế nào.

```python
# Cách tính tọa độ 3D từ barycentric:
vi = F[face_idx]         # 3 đỉnh của triangle chứa landmark
P  = w0*V[vi[:,0]] + w1*V[vi[:,1]] + w2*V[vi[:,2]]  # nội suy
```

### Dữ liệu output

```
output/
├── bfm_aligned.obj         # BFM sau khi căn chỉnh sang DECA space
├── deca.obj                # DECA gốc (không đổi)
├── deca_texture.png        # Texture mới 1024×1024 (kết quả chính)
├── deca_textured.obj       # DECA với UV đầy đủ
├── deca_textured.mtl       # Material definition
└── FLAME_albedo_from_BFM.npz  # Albedo ở định dạng MPG FLAME
```

---

## 3. Công nghệ sử dụng

| Thư viện | Phiên bản | Vai trò trong pipeline |
|---|---|---|
| **NumPy** | ≥1.24 | Toàn bộ tính toán ma trận — SVD, vectorized ops, barycentric |
| **trimesh** | ≥4.0 | Load/export mesh OBJ/PLY, tính normals, proximity queries |
| **SciPy** | ≥1.10 | Đọc file MATLAB `.mat` (landmarks BFM) |
| **Open3D** | ≥0.17 | ICP registration (Point-to-Plane, Point-to-Point) |
| **OpenCV (cv2)** | ≥4.8 | Seam dilation, Gaussian blur, image post-processing |
| **PIL (Pillow)** | ≥10.0 | Load/save PNG texture |
| **Python stdlib** | — | `gc`, `io`, `os`, `time` |

### Kiến trúc tính toán

Tất cả tính toán nặng đều **vectorized hoàn toàn với NumPy** — không có vòng lặp Python trên pixel hay vertex. Điều này quan trọng vì:
- DECA có ~5k vertices → vertex transfer: 5k proximity queries × 3 ops = vectorizable
- Texture baking: 1024×1024 = ~1M pixels → **bắt buộc** phải vectorize hoặc batch

---

## 4. V1 — `align_only.py`

### Mục đích

Script đầu tiên được viết để **kiểm tra chất lượng alignment** trước khi làm texture transfer. Không chuyển texture, chỉ căn chỉnh BFM vào DECA space và export để xem trong browser.

### Pipeline

```
Input: BFM mesh + DECA mesh + landmarks (BFM .mat + DECA .npy)
  ↓
1. Load meshes + landmarks
2. Pre-center (trừ centroid)
3. Axis check (4 convention: XYZ, XZY, X-YZ, XY-Z)
4. Unit normalize (scale theo diagonal landmarks)
5. Procrustes (Umeyama SVD)
6. ICP 3-pass (nếu có Open3D)
7. Export:
   - output/bfm_aligned.obj   (BFM đã căn chỉnh)
   - output/deca.obj           (DECA gốc + UV)
   - output/deca_heatmap.obj   (DECA tô màu theo khoảng cách)
   - output/deca.mtl
```

### Heatmap distance visualization

```python
_, dists, _ = cp_fn(bfm_final, deca_verts)   # khoảng cách mỗi vertex DECA → BFM
max_d = np.percentile(dists, 95)              # clip ở p95 để outlier không chi phối
t = np.clip(dists / max_d, 0., 1.)           # normalize 0..1

# Jet colormap: xanh(0) → xanh lá → vàng → đỏ(1)
# Xanh = khít, Đỏ = lệch
```

Output heatmap cho phép **nhìn trực tiếp trong browser** xem vùng nào của DECA còn xa BFM sau alignment.

### Nhược điểm

| Vấn đề | Nguyên nhân |
|---|---|
| Không có texture output | Chỉ alignment, không bước transfer nào |
| Không dùng được trong production | Chỉ là inspection tool |
| Export bfm_aligned.obj bằng vòng lặp Python | Chậm với mesh lớn |
| Heatmap không lưu được màu chuẩn trong OBJ | Vertex color OBJ không được hỗ trợ rộng rãi |

---

## 5. V2 — `patch_uv.py`

### Mục đích

Một **quick fix** khi phát hiện `output/deca.obj` được export ra không có UV coordinates — browser viewer không render được texture. Script này copy UV từ source DECA OBJ (`bfm_to_flame/deca.obj`) vào output DECA OBJ.

### Pipeline

```
Input:
  bfm_to_flame/deca.obj   (source — có UV coords "vt" và face "f v/vt")
  output/deca.obj         (output — thiếu UV, chỉ có "f v")

Xử lý:
  1. Parse vt lines + face UV indices từ source
  2. Parse vertex lines + face vertex indices từ output
  3. Kiểm tra face count khớp nhau (topology phải giống hệt)
  4. Ghép lại: vertices từ output + UV coords từ source
  5. Ghi đè output/deca.obj với format "f v/vt"

Output:
  output/deca.obj   (đã có UV)
  output/deca.mtl   (material referencing deca_texture.png)
```

### Tại sao điều này hợp lệ

DECA mesh có topology cố định — số faces và thứ tự faces luôn giống nhau bất kể vertices nằm ở đâu trong không gian 3D. Do đó, UV indices từ source và vertex indices từ output **có thể ghép 1-1 theo index face**.

```python
# Điều kiện tiên quyết phải thỏa:
assert len(out_face_vi) == len(src_face_vti)
# → cùng số faces → UV mapping hợp lệ
```

### Nhược điểm

| Vấn đề | Nguyên nhân |
|---|---|
| Không transfer texture content | Chỉ patch UV, texture vẫn là BFM texture trong DECA UV space → sai màu |
| Giả định topology giống nhau | Nếu DECA bị reorder face khi export thì patch sai |
| Pure stdlib, không kiểm tra UV validity | Không detect được UV out-of-range hay degenerate |
| Chỉ là workaround | Không giải quyết bài toán gốc |

---

## 6. V3 — `align_and_transfer` (phiên bản hard-gate)

> **Lưu ý:** Phiên bản này không còn tồn tại dưới dạng file riêng. Nó là phiên bản trước của `align_and_transfer.py` hiện tại, được tái hiện từ các comments còn lại trong code.

### Bằng chứng trong code (V4 hiện tại)

```python
# Step 4 comment:
# "Strategy: remove ALL hard distance/normal gates from this step.
#  Every vertex receives a colour unconditionally via a 3-level fallback"

# Diagnostic line (vẫn còn trong V4 để so sánh):
print(f"  Would-cover at p95 threshold : {(all_dist <= p95).sum()}/{N} "
      f"({...}%)  ← old hard-gate behaviour")

# Baking step 8:
# "Occlusion — DISABLED for coverage diagnostics
#  Re-enable multi-ray block once vertex coverage consistently exceeds 4k/5118"
```

### Cơ chế của V3 (inferred)

```python
# V3 — cái đã bị xóa (reconstructed):

# Hard distance gate:
mask_dist = all_dist <= DIST_CAP         # reject nếu xa hơn 5cm
vertex_colors[~mask_dist] = 0           # pixel đen

# Hard normal gate:
dots = (deca_vn * bfm_face_n[all_tri]).sum(1)
mask_norm = dots > 0.0                  # reject nếu ngược hướng
vertex_colors[~mask_norm] = 0          # pixel đen

# Occlusion check (multi-ray):
# Ray cast từ DECA surface → kiểm tra có bị BFM che không
# → reject các vùng bị che
```

### Tại sao V3 fail

**Vấn đề coverage:** DECA và BFM có hình dạng hơi khác nhau dù đã align. Các vùng như tai, cổ, da đầu không có correspondence tốt trong BFM → khoảng cách lớn → bị hard gate reject → pixel đen.

```
DECA 5,023 vertices
  → ~4,000 verts pass distance gate (p95 ≈ 80%)
  → ~3,500 verts pass normal gate (thêm ~12% bị reject ở vùng occlusion)
  → "4k/5118" — bằng chứng còn trong comment: coverage chỉ đạt 4000/5118
  
Kết quả: ~20-30% texture bị đen
```

**Vấn đề occlusion:** Multi-ray occlusion check tốn kém tính toán và tạo ra false positives — vùng tai và gáy của DECA không tương ứng với BFM nhưng không phải "bị che", chỉ là không có dữ liệu.

---

## 7. V4 — `align_and_transfer` (phiên bản cuối)

### Tổng quan pipeline

```
Inputs:
  bfm_to_flame/uv_idm.obj   → BFM mesh (source)
  bfm_to_flame/uv_idm.png   → BFM texture 1024×1024
  bfm_to_flame/deca.obj     → DECA mesh (target)
  landmarks/uv_idm.mat      → 68 landmarks BFM
  landmarks/deca.npy        → 68 landmarks DECA

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: LOAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓ trimesh.load() → fix_normals() → fill_holes()
  ↓ PIL.Image.open() → numpy array (float32, /255)
  ↓ parse_obj_face_uvs() → bfm_face_uvs (F,3,2)
  ↓ load_landmarks_bfm() → lm_bfm (68,3)
  ↓ load_landmarks_deca() → lm_deca (68,3)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: PROCRUSTES ALIGNMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓ Pre-center (trừ centroid BFM)
  ↓ Axis check 4 candidates → chọn MSE thấp nhất
  ↓ Unit normalize (scale theo diagonal bbox landmarks)
  ↓ Umeyama SVD → (R 3×3, s scalar, t 3-vector)
  ↓ apply_transform() → bfm_verts_aligned

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3: ICP REFINEMENT (3 pass)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓ Pass 1: Point-to-Plane, threshold=0.05m → T1 (4×4)
  ↓ Pass 1.5: Point-to-Point, threshold=0.03m → T15 (ổn định translation)
  ↓ Pass 2: Point-to-Plane, threshold=0.02m → T2 (fine rotation)
  ↓ Compute adaptive DIST_CAP = max(0.03, 2 × surface_RMSE)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4: VERTEX-LEVEL COLOUR TRANSFER (100% coverage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓ ProximityQuery.on_surface(deca_verts) → cp, dist, tri (N,3)/(N,)/(N,)
  ↓ Soft weights: w_dist = 1-dist/p99, w_norm = (dot+0.8)/1.8
  ↓ vertex_confidence = w_dist × w_norm (N,) [dùng ở step baking]
  ↓ bary_uv_batch() → UV coords (N,2) trên closest BFM triangle
  ↓ Fallback degenerate: nearest-corner UV
  ↓ sample_texture() bilinear → vertex_colors (N,3)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5: PER-TEXEL BAKING (1024×1024)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓ rasterize_uv_triangles() → face_map (1024,1024), bary_map (1024,1024,3)
  ↓ For each batch of 10k texels:
      pts_3d = barycentric interpolation từ bary_map
      ProximityQuery.on_surface(pts_3d) → BFM closest point
      w_dist (quadratic falloff) × w_norm × w_occ(=1)
      Landmark boost ±5cm quanh 68 landmarks
      Smoothstep → gamma 2.2 → sigmoid blending
      Procedural inpainting cho non-face regions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6: SEAM DILATION + GAUSSIAN AA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓ Pass A: dilate 5×5 ×3 iter → fill UV seam gaps
  ↓ Pass B: dilate 9×9 ×5 iter → fill ear/neck/scalp background
  ↓ Pass C: GaussianBlur(11×11) → soft boundary blend

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 7: EXPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓ deca_texture.png (PIL → uint8)
  ↓ deca_textured.obj (StringIO buffer + np.savetxt)
  ↓ deca_textured.mtl
  ↓ FLAME_albedo_from_BFM.npz (MU = flatten texture)
  ↓ bfm_aligned.obj (để kiểm tra alignment trong browser)
```

---

### Step 1 — LOAD chi tiết

#### Parse OBJ UV (`parse_obj_face_uvs`)

OBJ format có thể tách biệt chỉ số vertex và chỉ số UV:

```
vt 0.241 0.847        ← UV coordinate #5
f  1/5  2/7  3/9     ← vertex 1 dùng UV 5, vertex 2 dùng UV 7, ...
```

Hàm này trả về `fuvs` shape `(F, 3, 2)` — mỗi trong F faces có 3 đỉnh, mỗi đỉnh có tọa độ (u, v). Đây là dữ liệu cốt lõi cho cả rasterization và sampling.

#### Load landmarks DECA

```python
data = np.load(path, allow_pickle=True).item()
face_idx = data['full_lmk_faces_idx']   # (68,) face index
bary     = data['full_lmk_bary_coords'] # (68,3) barycentric weights
P = w0*V[vi[:,0]] + w1*V[vi[:,1]] + w2*V[vi[:,2]]  # tọa độ 3D thực
```

---

### Step 2 — PROCRUSTES chi tiết

#### Tại sao cần 4 sub-bước

**Sub-bước 1 — Pre-centering:**

Nếu BFM nằm ở tọa độ (500, 300, 200) còn DECA ở (0.1, 0.2, 0.1), phép nhân ma trận sẽ bị overflow hoặc mất precision. Trừ centroid trước:

```python
bfm_verts_c = bfm_verts - lm_bfm.mean(0)   # về gốc tọa độ
```

**Sub-bước 2 — Axis check:**

BFM và DECA được train từ các framework khác nhau, có thể dùng convention trục tọa độ khác nhau:

| Convention | Thực tế | Ánh xạ |
|---|---|---|
| Y-up (OpenGL) | Y là chiều cao | XYZ giữ nguyên |
| Z-up (3ds Max) | Z là chiều cao | Cần hoán đổi Y↔Z → XZY |
| Y-down (ảnh) | Y tăng xuống dưới | Cần flip Y → X-YZ |

```python
_cands = {
    'XYZ' : lm_bfm_c.copy(),             # không đổi
    'XZY' : lm_bfm_c[:, [0, 2, 1]],     # Z-up → Y-up
    'X-YZ': lm_bfm_c * [1., -1., 1.],   # flip Y
    'XY-Z': lm_bfm_c * [1., 1., -1.],   # flip Z
}
# Chọn cái nào có MSE thấp nhất với DECA landmarks
_best = min({k: ((v-lm_deca_c)**2).mean() for k,v in _cands.items()})
```

**Sub-bước 3 — Unit normalize:**

Scale BFM sao cho diagonal bounding box của landmarks bằng với DECA. Tránh trường hợp Procrustes phải xử lý scale quá lớn/nhỏ (ví dụ BFM dùng mm, DECA dùng meter → factor 1000):

```python
pre = deca_diag / bfm_diag   # pre-scale factor
lm_bfm_c *= pre              # landmarks bây giờ cùng scale
bfm_verts_c *= pre           # apply lên cả mesh
```

**Sub-bước 4 — Umeyama Procrustes:**

Tìm (R, s, t) tối thiểu hóa `||tgt - s*R*src - t||²` trên 68 landmarks:

```python
K = tgt_centered.T @ src_centered / n   # cross-covariance (3×3)
U, Σ, Vt = SVD(K)

S = I  # xử lý reflection: nếu det(U)×det(Vt) < 0 → flip S[2,2]=-1
R = U @ S @ Vt                          # rotation matrix
s = (Σ·diag(S)).sum() / var_src        # scale
t = centroid_tgt - s * centroid_src @ R.T   # translation
```

Lý do flip `S[2,2]`: SVD có thể cho ra ma trận reflection (det = -1) thay vì rotation thuần (det = +1). Phép flip sửa điều này mà không ảnh hưởng đến optimality.

---

### Step 3 — ICP 3-PASS chi tiết

#### Tại sao Procrustes chưa đủ

Procrustes chỉ dùng 68 landmarks, không phải toàn bộ surface. Sau Procrustes, surface RMSE vẫn có thể là 5-10mm, đủ để gây color bleeding khi transfer.

#### Voxel downsampling

Trước ICP, mesh được downsample bằng voxel grid:

```python
src_down = src_pcd.voxel_down_sample(ICP_VOXEL)  # ICP_VOXEL = 0.002m = 2mm
```

Từ 53k points → vài nghìn points. ICP vẫn converge về cùng solution nhưng nhanh hơn ~100×.

#### 3-Pass strategy

| Pass | Method | Threshold | Mục đích |
|---|---|---|---|
| 1 | Point-to-Plane | 0.05m | Coarse — xử lý offset lớn còn lại sau Procrustes |
| 1.5 | Point-to-Point | 0.03m | Ổn định translation — P2Plane có thể drift translation |
| 2 | Point-to-Plane | 0.02m | Fine — khóa rotation với precision cao |

**Point-to-Plane vs Point-to-Point:**

- Point-to-Point: minimize `||p_src - p_tgt||²` → slow convergence, dễ bị local minima
- Point-to-Plane: minimize `(p_src - p_tgt) · n_tgt` → cho phép sliding dọc surface, converge nhanh hơn, đặc biệt tốt với surface phẳng (má, trán)

**Adaptive DIST_CAP sau ICP:**

```python
_, surf_d, _ = trimesh_closest_point(bfm_aligned, deca_verts_sample)
surf_rmse = sqrt(mean(surf_d²))
DIST_CAP = max(0.03, 2.0 * surf_rmse)   # dựa trên RMSE thực tế
```

Không hardcode ngưỡng — nếu alignment tốt (RMSE = 1mm) thì DIST_CAP = 2mm; nếu kém hơn (RMSE = 3cm) thì DIST_CAP = 6cm.

---

### Step 4 — VERTEX-LEVEL COLOUR TRANSFER chi tiết

#### Chiến lược 100% coverage (khác V3)

Thay vì reject vertices "xấu", V4 **cho tất cả vertices nhận màu** và lưu quality score vào `vertex_confidence` để dùng ở baking step:

```python
# Không còn: if dist > DIST_CAP: skip
# Thay bằng:
w_dist = clip(1 - dist/p99, 0, 1)           # soft weight, không bao giờ = -∞
w_norm = clip((dot + 0.8) / 1.8, 0, 1)     # cả dot=-0.8 cũng nhận weight nhỏ > 0
vertex_confidence = w_dist * w_norm          # [0,1], dùng ở baking
```

**Tại sao bỏ hard gate cho normal?**

Normal gate `dot > 0` reject tất cả vertices mà DECA normal ngược với BFM normal. Nhưng ở vùng tai/gáy của DECA, normals quay ra ngoài còn BFM bọc theo hướng khác → gate này loại nhiều vertices hợp lệ.

#### Barycentric UV

Sau khi biết DECA vertex nằm gần BFM triangle nào (`all_tri`), tính UV bằng:

```python
# Giải hệ Cramer để tìm barycentric coords (w0, w1, w2)
# sao cho: w0*p0 + w1*p1 + w2*p2 = query_point

e1, e2 = p1-p0, p2-p0
d11 = dot(e1,e1);  d12 = dot(e1,e2);  d22 = dot(e2,e2)
dp1 = dot(ep,e1);  dp2 = dot(ep,e2)
den = d11*d22 - d12²

bv = (d22*dp1 - d12*dp2) / den
bw = (d11*dp2 - d12*dp1) / den
bu = 1 - bv - bw

# UV interpolation:
uv = bu*uv0 + bv*uv1 + bw*uv2
```

**Fallback degenerate triangles:** Nếu `|den| < 1e-12` (tam giác suy biến — 3 điểm gần thẳng hàng) thì barycentric không xác định. Fallback: lấy UV của đỉnh gần query point nhất.

#### Bilinear texture sampling

```python
# UV (0,0) = bottom-left trong 3D, nhưng (0,0) = top-left trong image
py = (1 - v) * (H-1)   # flip Y

# Lấy 4 pixel xung quanh và nội suy:
color = img[y0,x0]*(1-wx)*(1-wy) + img[y0,x1]*wx*(1-wy)
      + img[y1,x0]*(1-wx)*wy     + img[y1,x1]*wx*wy
```

---

### Step 5 — PER-TEXEL BAKING chi tiết

#### Tại sao cần baking sau khi đã có vertex colors

Vertex colors gán màu cho **đỉnh** (5k điểm). Texture baking gán màu cho **pixel** (1M điểm). Mỗi pixel trong texture 1024×1024 có thể nằm **giữa** các đỉnh → cần nội suy chính xác hơn theo UV space.

#### Zero-loop UV Rasterization

Thay vì lặp qua từng face rồi từng pixel:

```python
# BAD (V3 hoặc naive approach):
for face in faces:
    for y in range(bb_y0, bb_y1):
        for x in range(bb_x0, bb_x1):
            if inside(face, x, y): ...   # 3 nested loops

# GOOD (V4 — vectorized):
# 1. Bounding boxes cho TẤT CẢ faces cùng lúc
bbx0, bbx1 = floor(min(px)), ceil(max(px))  # (F,) arrays

# 2. Enumerate candidates bằng np.repeat (ZERO loop):
lfi_flat = np.repeat(arange(M), pix_per_face)  # (n_cand,) face index
local = arange(n_cand) - cumsum(pix_per_face)[lfi_flat]
loc_row, loc_col = local // cols, local % cols

# 3. Edge-function test trên TOÀN BỘ flat array:
w0 = ((p2x-p1x)*(pty-p1y) - (p2y-p1y)*(ptx-p1x)) / area   # (n_cand,)
inside = (w0>=0) & (w1>=0) & (w2>=0)                         # (n_cand,) bool
```

Kết quả: `face_map[H,W]` và `bary_map[H,W,3]` — mỗi pixel biết thuộc face nào và barycentric coords là bao nhiêu.

#### Multi-weight blending (baking loop)

Với mỗi batch 10k texels:

```python
# 1. Reconstruct 3D position từ UV rasterization
pts_3d = w0*V[i0] + w1*V[i1] + w2*V[i2]     # (K,3) DECA surface points

# 2. Query BFM surface
cp_pts, cp_dists, cp_tri_ids = bfm_pq.on_surface(pts_3d)

# 3. Distance weight — quadratic falloff
w_dist = clip(1 - dist/BAKE_DIST_CAP, 0, 1)²   # quadratic vì linear có plateau ở gần 1

# 4. Normal weight — DECA face vs BFM face
dot = einsum('ij,ij->i', deca_fn_normalized, bfm_fn_normalized)  # cosine
w_norm = clip((dot + 0.8) / 1.8, 0, 1)

# 5. Sample BFM texture
uvs, uv_ok = bary_uv_batch(cp_pts, bfm_triangle_verts, bfm_uvs)
sampled_color = sample_texture(uvs, tex_np)     # bilinear

# 6. Smoothstep + cinematic gamma
w_smooth = w² × (3 - 2w)           # smoothstep(0,1,w) — không có kink ở biên
w_final  = w_smooth^2.2            # gamma 2.2 — đẩy confident verts lên, uncertain xuống

# 7. Landmark boost
lm_dists = sqrt(sum((pts_3d[:,None,:] - lm_deca[None,:,:])**2, axis=2))  # (K,68)
min_lm_d = min(lm_dists, axis=1)          # (K,) khoảng cách đến landmark gần nhất
lm_boost = clip(1 - min_lm_d/0.05, 0, 1) * 0.3   # trong 5cm: +0.3 boost

# 8. Sigmoid blending (anti-patchy)
blend = 1 / (1 + exp(-15 × (w - 0.4)))   # steep sigmoid tại w=0.4
color = blend × sampled + (1-blend) × mean_skin
```

**Tại sao 3 lớp weight function:**
- Smoothstep loại bỏ plateau artifacts khi weight gần 0 hoặc 1
- Gamma 2.2 tạo ra phân phối bi-modal (confident/non-confident) rõ ràng hơn
- Sigmoid blending đảm bảo transition mềm mại không có seam cứng

#### Procedural inpainting

Các texels không có BFM correspondence (tai, cổ, da đầu):

```python
# Base color: mean_skin + low-freq noise
noise = random.normal(0, 0.012, (n_nf, 3))   # σ=1.2% per channel
noise = convolve(noise, k8)                   # smooth 8-neighbor average

# Boundary blend: kéo màu từ face texture nearby (trong 3cm)
BLEND_R = 0.03
alpha = clip(1 - min_d_to_face/BLEND_R, 0, 1)
color_nf = alpha × edge_color + (1-alpha) × (mean_skin + noise)
```

---

### Step 6 — SEAM DILATION chi tiết

#### Vấn đề UV seam

UV atlas của mesh có "đường khâu" (seam) — cùng một edge 3D nhưng được map sang 2 vị trí khác nhau trong UV space. Khi rasterize, các pixel ở sát seam có thể không được fill → đường viền tối xuất hiện khi render.

```
UV space:
  ████████░░░░░░░░
          ↑
      Seam gap (pixel không thuộc triangle nào)
```

#### 3-pass fix

```python
# Pass A — fill seam gaps (tight, chỉ apply gần face region)
kernel_sm = ones((5,5))
dilated_sm = cv2.dilate(out_u8, kernel_sm, iterations=3)
seam_near = dilate(covered_mask, kernel_sm) & ~covered_mask  # vùng sát seam
out_u8[seam_near] = dilated_sm[seam_near]

# Pass B — flood background (ear/neck/scalp)
kernel_lg = ones((9,9))
dilated_bg = cv2.dilate(out_u8, kernel_lg, iterations=5)
still_empty = ~covered_mask & ~seam_near
out_u8[still_empty] = dilated_bg[still_empty]

# Pass C — soft boundary (blend face ↔ procedural)
conf_blur = GaussianBlur(covered_mask, (11,11), sigma=3.0)
blend_band = (conf_blur > 0.05) & (conf_blur < 0.95) & ~covered_mask
alpha = conf_blur[blend_band]
out_u8[blend_band] = alpha × out_u8[blend_band] + (1-alpha) × blurred_full[blend_band]
```

---

## 8. Tại sao V4 hoạt động được

### 1. Alignment 3 giai đoạn (Procrustes → ICP)

Procrustes cho kết quả nhanh nhưng chỉ tốt với landmarks. ICP sau đó tối ưu toàn bộ surface. Kết hợp cả hai đảm bảo surface RMSE < 2-3mm — đủ nhỏ để texture sampling cho màu đúng.

### 2. Bỏ hard gate → 100% coverage

Hard gate trong V3 tạo ra **binary decision**: hoặc vertex có màu, hoặc không. Vùng không có màu = màu đen trong texture. V4 thay bằng soft weight — mọi vertex đều có màu, chỉ là confidence khác nhau. Vùng kém confidence được blending với `mean_skin` (màu da trung bình) thay vì đen.

### 3. Hai lớp sampling (vertex + texel)

Vertex transfer nhanh, coverage 100%, nhưng chỉ có độ phân giải bằng số vertices (~5k).
Texel baking chậm hơn nhưng có độ phân giải 1M pixels, với barycentric nội suy giữa các vertices. Kết hợp: vertex transfer cung cấp `vertex_confidence` và `mean_skin` làm fallback cho texel baking.

### 4. Smoothstep + gamma tạo phân phối nhị phân sắc nét

Với weight function tuyến tính đơn giản, vùng confidence trung bình (0.4-0.6) sẽ là trung bình giữa sampled và mean_skin → màu nhạt không tự nhiên. Smoothstep + gamma đẩy distribution về hai cực:

```
w=0.3 → smoothstep=0.22 → ^2.2=0.04  (→ gần mean_skin)
w=0.5 → smoothstep=0.50 → ^2.2=0.22  (→ blended)
w=0.7 → smoothstep=0.78 → ^2.2=0.58  (→ gần sampled)
w=0.9 → smoothstep=0.97 → ^2.2=0.93  (→ gần sampled)
```

### 5. Landmark boost bảo vệ vùng chi tiết cao

Mắt, mũi, miệng là vùng landmark density cao và cũng là vùng người dùng quan sát kỹ nhất. Boost +0.3 cho các texel trong bán kính 5cm quanh landmarks đảm bảo những vùng này luôn dùng sampled color, không bao giờ fade sang mean_skin.

### 6. Sigmoid blending loại bỏ patchy artifacts

Linear blending `α*a + (1-α)*b` với α phân phối đều tạo ra gradient rõ ràng. Sigmoid blending `σ(k×(w-0.4))` với `k=15` tạo ra transition band hẹp — chỉ ~10% pixels nằm trong transition, 45% là sampled thuần, 45% là mean_skin thuần → không có vùng "nửa nạc nửa mỡ".

### 7. Seam dilation đảm bảo zero black pixels

Dù baking có procedural inpainting, vẫn có thể có pixel nào đó bị bỏ qua (precision rasterization). Dilation pass sau baking là safety net cuối cùng — flood-fill từ màu hiện có ra mọi pixel còn trống.

---

## Tóm tắt tiến hóa qua các phiên bản

```
V1 align_only.py
    Mục đích: inspection
    Nhược điểm: không có texture output

       ↓ cần xem texture trong browser

V2 patch_uv.py
    Mục đích: quick fix UV thiếu
    Nhược điểm: không transfer content, chỉ patch structure

       ↓ cần transfer thực sự

V3 align_and_transfer (hard-gate)
    Mục đích: full pipeline lần đầu
    Nhược điểm: coverage ~80%, đen ở tai/cổ/da đầu

       ↓ cần 100% coverage

V4 align_and_transfer (final) ← hiện tại
    Bỏ hard gate → soft weights → 100% coverage
    Procedural inpainting cho non-face regions
    Smoothstep + gamma + sigmoid cho natural blending
    Landmark boost bảo vệ vùng feature
    Seam dilation safety net
```
