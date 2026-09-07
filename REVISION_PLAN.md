# TESSERA — Tái thiết kế phần Method

> **Mục tiêu:** gom mọi cải tiến thành **một kỹ thuật có tên, có nguyên lý thống nhất**, đủ mạnh để trả lời *"limited novelty at the algorithmic level"* (R3) và *"largely built upon a combination of existing components"* (R5).
>
> **Ngày lập:** 2026-09-05 · **Đích:** SOICT · **Trọng tâm:** §3 Method · **Thời gian máy:** ưu tiên cho SOTA

---

## 0. Vì sao bản trước còn yếu

Bản trước liệt kê 5 cải tiến rời. Dù mỗi cái đều đúng, người đọc vẫn thấy **năm miếng vá**, không thấy **một ý tưởng**. Reviewer sẽ nói y hệt lần cũ.

Bản này gom tất cả về **một nguyên lý duy nhất**, và mọi thành phần đều là hệ quả của nguyên lý đó:

> **Mỗi nguồn bằng chứng đều tự đo được độ tin cậy của chính nó, từ một đại lượng dư mà nó vốn đã phải tính.**

Đây không phải khẩu hiệu. Nó là một quan sát kỹ thuật cụ thể:

| Nguồn bằng chứng | Đại lượng dư nó **vốn đã tính** | → Độ tin cậy tự đo |
|---|---|---|
| Vận chuyển liên tôpô | dư lượng khoảng cách bề mặt + lệch pháp tuyến | $\mathcal{R}_{\text{geo}}$ |
| Vận chuyển đối xứng | dư lượng bất đối xứng trên **vùng chồng lấn** | $\mathcal{R}_{\text{sym}}$ |
| Tiên nghiệm sinh | phương sai giữa các lần lấy mẫu | $\mathcal{R}_{\text{gen}}$ |

Ba nguồn hoàn toàn khác bản chất, nhưng đều **tự khai báo độ tin cậy trong cùng một thang $[0,1]$, đo bằng chính dư lượng của mình, không cần hằng số chỉnh tay, không cần huấn luyện thêm.** Vì đồng thang nên chúng **hợp nhất được bằng một phép giải duy nhất**.

Đó là ý tưởng của bài. Mọi thứ còn lại là hệ quả.

---

## 1. TESSERA

**T**exture **E**vidence **S**ynthesis via **S**elf-**E**stimated **R**eliability **A**rbitration

*(tessera: viên gạch nhỏ trong tranh khảm — đúng nghĩa đen của việc ghép một atlas hoàn chỉnh từ những mảnh bằng chứng có xuất xứ và độ tin cậy khác nhau)*

> **Tên thay thế nếu bạn thích nghĩa "phân xử" hơn nghĩa "khảm":** **ARBITER** — *Adaptive Reliability-Based Integration of Texture Evidence and Rendering*. Cả hai đều dùng được; TESSERA gợi hình hơn, ARBITER nói đúng cơ chế hơn. Chọn một rồi dùng nhất quán toàn bài.

### Phát biểu một câu

> TESSERA coi hoàn thiện texture là bài toán **phân xử giữa nhiều nhân chứng có độ khả tín khác nhau**, trong đó mỗi nhân chứng tự khai báo độ khả tín của mình bằng dư lượng của chính nó, và mọi lời khai được hợp nhất bằng một phép giải Poisson neo theo độ tin cậy trên bề mặt.

### Ba thành phần, một phép giải

```
   NHÂN CHỨNG 1                NHÂN CHỨNG 2               NHÂN CHỨNG 3
   Quan sát trực tiếp          Quan sát đối xứng          Tiên nghiệm sinh
   (pixel thật, nhìn thấy)     (pixel thật, nửa kia)      (diffusion đoán)
         │                            │                          │
    ┌────┴────┐                  ┌────┴────┐                ┌────┴────┐
    │   CTT   │                  │   IST   │                │  k mẫu  │
    │ vận chuyển                 │ vận chuyển               │ + phương│
    │ liên tôpô                  │ đối xứng                 │   sai   │
    └────┬────┘                  └────┬────┘                └────┬────┘
    dư: khoảng cách             dư: bất đối xứng           dư: phương sai
      bề mặt                      vùng chồng lấn             giữa các mẫu
         │                            │                          │
         └──────────────┬─────────────┴──────────────────────────┘
                        │
                 ┌──────┴──────┐
                 │    SERF     │   Trường độ tin cậy tự đo
                 │  R_geo · R_sym · R_gen  →  R(u,v) ∈ [0,1]
                 └──────┬──────┘
                        │
                 ┌──────┴──────┐
                 │    SAPF     │   Một phép giải Poisson duy nhất
                 │  trên đồ thị kề **bề mặt**, neo bởi R
                 └──────┬──────┘
                        │
              T_final  +  R_final     ← texture mang theo độ tin cậy của chính nó
                        │
                 ┌──────┴──────┐
                 │     VQP     │   Dự báo chất lượng Q(θ) ở mọi góc nhìn
                 └─────────────┘
```

**Bốn ký hiệu dùng xuyên suốt bài:**

| Viết tắt | Tên đầy đủ | Vai trò |
|---|---|---|
| **CTT** | Cross-Topology Transport | Vận chuyển bằng chứng qua khác biệt tôpô, ở mức texel |
| **IST** | Intrinsic Symmetry Transport | Vận chuyển bằng chứng thật từ nửa mặt đối diện |
| **SERF** | Self-Estimated Reliability Field | Trường độ tin cậy tự đo — **hạt nhân của bài** |
| **SAPF** | Surface-Anchored Poisson Fusion | Hợp nhất một lần trên hình học thật |
| **VQP** | View Quality Prediction | Năng lực tự đánh giá theo góc nhìn |

### Vì sao đây là **một** kỹ thuật, không phải năm miếng vá

Bỏ bất kỳ thành phần nào thì nguyên lý sụp:

- Bỏ **CTT** → không có gì để đo $\mathcal{R}_{\text{geo}}$, và bằng chứng chỉ còn ở độ phân giải lưới (5k mẫu cho 1M texel)
- Bỏ **IST** → mất hẳn một nhân chứng, và mất luôn phép đo bất đối xứng vốn cũng dùng để hiệu chỉnh
- Bỏ **SERF** → ba nhân chứng không còn đồng thang, không hợp nhất được, chỉ còn cách cắt cứng bằng mask
- Bỏ **SAPF** → mỗi nhân chứng thành một mảnh vá có đường biên nhìn thấy được
- Bỏ **VQP** → $\mathcal{R}$ thành biến nội bộ chết, đúng như `vertex_confidence` hiện nay

Đây là dấu hiệu của một kiến trúc thật: các bộ phận **cần nhau**, không phải xếp cạnh nhau.

---

## 2. Chi tiết từng thành phần

### 2.1 · CTT — Cross-Topology Transport

**Vấn đề trong kiến trúc hiện tại.** Code giải màu ở **mức vertex** rồi nội suy barycentric ra atlas $1024^2$:

```
Texel được phủ  : 1,021,378
Mẫu màu thật có :     5,023      ← nội suy 203×
```

Đo được trên V000190: `grad_mean` rơi từ **10.18** ($T_{IDM}$ nguồn) xuống **7.38** (đầu ra) — **mất 27% năng lượng gradient** ngay tại bước cốt lõi của bài. Bài quảng cáo atlas $1024^2$ nhưng thông tin thật chỉ tương đương $\sqrt{5023} \approx 71^2$. Nguồn gốc: [`align_and_transfer.py:714`](align_and_transfer.py#L714) — *"no BFM surface query"*.

**Thiết kế.** Thay vì truyền *màu* theo vertex, tính một **trường vận chuyển liên tục**

$$\Phi: \Omega_{UV}^{\text{tgt}} \to \Omega_{UV}^{\text{src}}$$

Với mỗi texel: lấy điểm 3D bằng barycentric (`face_map`/`bary_map` đã có), truy vấn điểm gần nhất trên bề mặt nguồn đã căn chỉnh, đọc UV nguồn tại đó.

**Vì sao nó là một *toán tử*, không chỉ là "truy vấn theo texel":**
- **Độc lập độ phân giải** — tính một lần, áp ở mọi mức
- **Đa kênh** — chuyển được màu, normal, mask, và **cả độ tin cậy**; đây là điều SERF cần
- **Tự đo dư lượng** — khoảng cách bề mặt và lệch pháp tuyến tại mỗi texel cho thẳng $\mathcal{R}_{\text{geo}}$, không cần thêm phép tính nào

$$\mathcal{R}_{\text{geo}}(u,v) = \underbrace{\rho\!\left(d(u,v)\right)}_{\text{dư lượng khoảng cách}} \cdot \underbrace{\psi\!\left(\langle \hat{n}_{\text{tgt}}, \hat{n}_{\text{src}} \rangle\right)}_{\text{nhất quán pháp tuyến}}$$

Đây là bản nâng cấp của $w_{\text{dist}} \cdot w_{\text{norm}}$ đang có — nhưng ở **mức texel** và **thực sự được dùng** (hiện tại `vertex_confidence` được tính rồi bỏ, [:721](align_and_transfer.py#L721)).

**Mức mới:** Trung bình. Là *nền* của TESSERA, không phải điểm bán hàng. Bài phải trình bày nó đúng vai trò đó.

---

### 2.2 · IST — Intrinsic Symmetry Transport

**Quan sát khởi nguồn.** Chụp ở $-35°$, má trái khuất — nhưng **má phải đã được chụp rõ**. Khuôn mặt gần đối xứng, nên vùng khuất có sẵn một nguồn bằng chứng **thật**. Kiến trúc hiện tại không dùng gì cả, giao toàn bộ cho diffusion hallucinate.

Đáng chú ý hơn: nhìn atlas $T_{IDM}$ thấy rõ model **tự nó** đang dựa nặng vào đối xứng (hai nửa gần như soi gương). Ta đang trả tiền cho một model sinh để làm việc mà hình học cho gần như miễn phí — và làm kém chính xác hơn.

**Thiết kế ba bước.**

1. **Ước lượng mặt phẳng đối xứng nội tại** $\mathcal{P}$ từ chính hình học FLAME (cực tiểu dư lượng giữa mesh và ảnh phản chiếu, khởi tạo từ landmark giữa 27–30, 8). Ước lượng từ hình học nên **vẫn đúng khi đầu nghiêng** — không phụ thuộc hệ toạ độ.
2. **Vận chuyển**: texel không quan sát được lấy bằng chứng từ texel đối ảnh $\sigma(u,v)$ nếu texel đó có quan sát trực tiếp.
3. **Hiệu chỉnh bất đối xứng**: ước lượng trường hiệu chỉnh tần số thấp từ **vùng chồng lấn** — nơi cả hai nửa cùng quan sát được — rồi áp cho vùng vận chuyển.

**Chỗ tinh tế nhất:** dư lượng trên vùng chồng lấn vừa dùng để hiệu chỉnh, vừa **chính là phép đo độ khả tín của nhân chứng này cho riêng đối tượng đó**:

$$\mathcal{R}_{\text{sym}} = \exp\!\left(-\frac{\|E_{\text{obs}} - \mathcal{C}(\sigma(E_{\text{obs}}))\|^2_{\text{overlap}}}{2\tau^2}\right)$$

Người mặt cân đối → dư lượng nhỏ → tin cậy cao. Người có nốt ruồi một bên, chiếu sáng lệch, hình học lệch → dư lượng lớn → hệ thống **tự động hạ tin cậy của chính tầng này**. Không hằng số chỉnh tay. Đây đúng là nguyên lý ở §0 được thể hiện.

**Mức mới:** **Cao trong bối cảnh này.** Đối xứng khuôn mặt là ý tưởng cổ điển; nhưng *ước lượng mặt đối xứng nội tại + hiệu chỉnh bất đối xứng học từ vùng chồng lấn + dùng chính dư lượng đó làm độ tin cậy tự đo* là thiết kế nguyên bản — và nó đặt bằng chứng thật **đứng trước** model sinh, đảo ngược hẳn kiến trúc hiện tại.

---

### 2.3 · SERF — Self-Estimated Reliability Field

**Hạt nhân của bài.** Thay tập nhị phân $\Omega_{in}$ (có/không có texture) bằng **trường liên tục** $\mathcal{R}: \Omega_{UV} \to [0,1]$.

$$\mathcal{R} = \max_k \; \mathcal{R}_k \cdot \mathcal{R}_{\text{obs}}, \qquad k \in \{\text{geo}, \text{sym}, \text{gen}\}$$

**Ba thành phần tự đo** (đã nêu ở 2.1, 2.2), cộng một hệ số quan sát chung:

$$\mathcal{R}_{\text{obs}} = \underbrace{V(u,v)}_{\text{khả kiến}} \cdot \underbrace{\cos\theta_{p}}_{\text{góc xiên}} \cdot \underbrace{s(u,v)}_{\text{mật độ lấy mẫu}}$$

Một texel nhìn thấy ở góc rất xiên thì **có** bằng chứng, nhưng bằng chứng đó tồi — chưa kiến trúc nào ở đây mô hình hoá điều này.

**$\mathcal{R}_{\text{gen}}$ — bất định của diffusion đo bằng đa mẫu.** Chạy UV-IDM $k$ lần với seed khác nhau; trung bình làm nội dung, **phương sai theo texel** làm bất định. Chỗ nào model dao động mạnh giữa các lần sinh chính là chỗ nó đang đoán. Rẻ, không cần huấn luyện, lấy được bất định từ một model sinh hộp đen.

**Nâng cấp lý thuyết.** Điều kiện phủ nhị phân của bài cũ trở thành **trường hợp suy biến** $\mathcal{R} \in \{0,1\}$. Bài mới **tổng quát hoá** chính đóng góp mà R3 đã khen (*"the problem formulation around UV-domain coverage is clear"*) thay vì lặp lại nó.

**Mức mới:** **Cao.**

**⚠️ Rủi ro phụ thuộc — cần xác nhận trước tiên.** Cần chạy lại được UV-IDM để lấy $k$ mẫu. Repo chỉ có output, không có model.
*Dự phòng nếu không chạy lại được:* ước lượng $\mathcal{R}_{\text{gen}}$ gián tiếp bằng độ lệch giữa nội dung sinh và bằng chứng đối xứng ở vùng cả hai cùng có mặt — yếu hơn nhưng vẫn giữ được nguyên lý tự đo, và **vẫn dùng đúng khung TESSERA**.

---

### 2.4 · SAPF — Surface-Anchored Poisson Fusion

**Hai lỗi hình học đang tồn tại.**

Pass A ($5\times5$, 3 iter) + Pass B ($9\times9$, 5 iter) lan tối đa **~26 px** trong mặt phẳng UV trên atlas $1024^2$. Nhưng **UV không phải mặt phẳng của bề mặt**: atlas FLAME cắt đầu thành nhiều island, hai texel kề nhau trong UV có thể nằm cách xa nhau trên đầu thật. Dilation vuông trong UV **lan màu xuyên qua ranh giới island** — chính là nguồn các vệt tối bẩn quanh tai và góc atlas nhìn bằng mắt thấy rõ. Pass C còn dùng Gaussian blur ($\sigma=1.5$) làm mờ thêm chi tiết ở đúng dải chuyển tiếp.

**Thiết kế.**

**(a) Kề nhau theo hình học, không theo UV.** Dựng đồ thị kề texel trong đó hai texel nối nhau khi kề nhau **trên bề mặt 3D**: kề trong cùng tam giác, **và** nối qua đường cắt UV bằng cạnh 3D chung giữa hai island. Mọi phép lan truyền từ đó tôn trọng hình học thật; dilation vuông bị loại bỏ hoàn toàn.

**(b) Hợp nhất trong miền gradient, neo bởi độ tin cậy.**

$$\min_{T} \; \sum_{(p,q) \in \mathcal{E}} \left\| (T_p - T_q) - g_{pq} \right\|^2 \;+\; \lambda \sum_{p} \mathcal{R}_p \left\| T_p - E_p \right\|^2$$

- $g_{pq}$: gradient mục tiêu, lấy từ nhân chứng đáng tin nhất tại cạnh đó
- Số hạng hai neo giá trị về bằng chứng $E_p$, **trọng số chính là $\mathcal{R}_p$**
- Texel tin cậy cao (pixel thật) gần như bị ghim; texel tin cậy thấp để lời giải trôi sao cho gradient liên tục

**Điểm mạnh về trình bày:** thay **ba pass hình thái học chỉnh tay** bằng **một bài toán tối ưu có cơ sở**. Ba nhân chứng hoà vào nhau không còn đường biên nhìn thấy, không cần bất kỳ mask cắt cứng nào. Đây là chỗ TESSERA trở thành *một* thuật toán thay vì một chuỗi bước.

Hệ thưa ~1M ẩn, giải bằng CG/AMG — chạy CPU được.

**Mức mới:** **Cao.** Poisson blending cổ điển, nhưng *đồ thị kề texel khâu qua đường cắt UV + neo bằng trường độ tin cậy tự đo* là thiết kế nguyên bản, và sửa một lỗi hình học thật.

---

### 2.5 · VQP — View Quality Prediction

**Vì đầu ra mang theo $\mathcal{R}$, hệ thống biết trước nó đáng tin ở góc nào — trước khi render.**

$$Q(\theta) = \frac{\displaystyle\int_{\text{Vis}(\theta)} \mathcal{R}\big(\varphi^{-1}(p)\big)\, \cos\theta_p \; dA}{\displaystyle\int_{\text{Vis}(\theta)} \cos\theta_p \; dA}$$

**Kiểm chứng:** $Q(\theta)$ có tương quan với PSNR/SSIM/LPIPS/CSIM đo thật ở 7 camera không? Báo cáo hệ số tương quan trên toàn tập.

**Vì sao đây là đòn kết:**
- **Năng lực tự đánh giá** — không method nào trong [a][b][c] có
- Ứng dụng trực tiếp: chọn góc an toàn, cảnh báo vùng không đáng tin, **chỉ ra nên chụp bổ sung ở góc nào**
- Biến mục Failure Analysis (R1.5 đòi) từ điểm yếu thành chỗ khoe sức mạnh: hệ thống **tự dự báo được** phần lớn ca hỏng
- Đóng vòng lý thuyết: $\mathcal{R}$ không phải biến nội bộ mà là **sản phẩm** của hệ thống

**Mức mới:** **Cao.**

---

### 2.6 · Thành phần phụ (làm nếu còn thời gian)

**Tinh chỉnh phi cứng có ràng buộc đối xứng.** ICP hiện chỉ tìm $\bm{\Xi} \in SE(3)$, nhưng sai số dư ở tai/viền hàm/chóp mũi đến từ **khác biệt hình học BFM↔FLAME**, không phép cứng nào khử được. Thêm biến dạng phi cứng nhẹ, ràng buộc landmark **và ràng buộc đối xứng $\mathcal{P}$ từ IST** — chỗ nối đẹp giữa hai thành phần, và giảm dư lượng CTT ở đúng vùng tệ nhất.

---

## 3. Bản đồ đóng góp → phản biện

| Thành phần | R1 | R3 | R5 |
|---|---|---|---|
| **Nguyên lý tự đo độ tin cậy** (§0) | — | **3.1 novelty** | **5.2 novelty** |
| CTT | 1.2 | 3.1 | 5.2 |
| IST | — | **3.1** | **5.2 · identity preservation** |
| SERF | — | **3.1** | 5.2 |
| SAPF | 1.5 artifact | **3.1** | 5.2 |
| VQP | 1.1 | 3.2 | **5.3** |
| Đánh giá 7 view | 1.3, 1.4 | 3.2 | **5.3** |
| SOTA + baseline mạnh | **1.2** | — | **5.4, 5.5** |
| Bỏ claim 4D | 1.1 | **3.3** | **5.1** |
| Failure analysis (qua VQP) | **1.5** | 3.2 | — |

---

## 4. Viết lại §3 Method

**Tiêu đề đề xuất:** *TESSERA: Self-Estimated Reliability Arbitration for Complete-Texture Free-Viewpoint Face Avatars from a Single Image*

**Bốn contribution:**

1. **Nguyên lý tự đo độ tin cậy** — chỉ ra rằng ba nguồn bằng chứng khác hẳn bản chất (hình học, đối xứng, sinh) đều tự khai báo được độ khả tín từ dư lượng vốn đã tính, trong cùng một thang; nhờ đó hợp nhất được bằng một phép giải duy nhất
2. **SERF** — nâng điều kiện phủ nhị phân thành trường độ tin cậy liên tục, bao trùm formalization cũ như trường hợp suy biến
3. **IST + SAPF** — vận chuyển đối xứng nội tại có hiệu chỉnh, hợp nhất Poisson trên đồ thị kề bề mặt khâu qua đường cắt UV; thay chuỗi heuristic hình thái học bằng một bài toán tối ưu
4. **VQP** — dự báo chất lượng theo góc nhìn, kiểm chứng bằng tương quan với metric đo thật trên 7 camera

**Cấu trúc §3 mới** — trình bày theo *nguyên lý*, không theo *thứ tự thực thi*:

```
3.1  Nguyên lý: bằng chứng và độ khả tín        ← đặt luận điểm trước
3.2  Ba nhân chứng và dư lượng tự đo của chúng
     3.2.1 CTT   3.2.2 IST   3.2.3 tiên nghiệm sinh + đa mẫu
3.3  SERF: hợp nhất về một thang chung
3.4  SAPF: phân xử bằng một phép giải
3.5  VQP: hệ quả — texture biết độ tin cậy của chính nó
3.6  Chi tiết cài đặt  ← 5 bước tuần tự xuống đây, đúng vai trò của chúng
```

Bố cục cũ đặt 5 bước tuần tự làm xương sống — đó chính là lý do bài đọc như pipeline. Bố cục mới đặt **nguyên lý** làm xương sống, các bước xuống mục cài đặt.

**Các sửa khác trong bài:**
- [ ] **Bỏ claim 4D toàn bài** (R3.3, R5.1) — tiêu đề, abstract, intro, §4.4. Giữ "animation-ready" như thuộc tính; `test/cheo1` làm demo định tính có ghi rõ giới hạn
- [ ] **Related Work** — thêm [a] NerFACE, [b] NHA, [c] FATE + định vị; thêm nhánh single-image full-head texture
- [ ] **§4 Setup** — mô tả đủ protocol đăng ký ảnh, renderer, camera pose, chiếu sáng, mask mặt (trả lời R1.3/R1.4)
- [ ] **Format** — xác nhận template SOICT (nhiều khả năng `acmart`; hiện đang `IEEEtran`)

---

## 5. Ngân sách tính toán

> Nguyên tắc: **mọi số trong bài đến từ một lần chạy thật.** Cách tiết kiệm là làm thí nghiệm rẻ đi và bỏ thí nghiệm không cần, không phải điền số.

### 5.1 Cắt giảm hợp lệ

| Kỹ thuật | Tiết kiệm | Ghi trong bài |
|---|---|---|
| **Ablation trên mẫu phân tầng** ~20–25 subject thay vì 74 | **~70%** | Ghi rõ $n$ và cách lấy mẫu. Đây là thông lệ chuẩn, reviewer chấp nhận |
| **Cache $\Phi$ (CTT)** — cùng cặp tôpô BFM↔FLAME, cấu trúc ổn định | lớn, đây là phần đắt nhất | chi tiết cài đặt |
| **$k = 4$ thay vì 8** cho đa mẫu | 50% chi phí diffusion | có ablation $k$ trên mẫu nhỏ để biện minh |
| **Bảng chính chỉ chạy đủ 74 subject × 7 view một lần** ở cấu hình cuối | — | đây là bảng bắt buộc phải đầy đủ |
| **Chỉ render lại cái đã đổi** | — | — |
| **Bỏ hẳn E7 (robustness pose đầu vào)** nếu thiếu giờ | 100% của mục đó | thà không có mục còn hơn có số không chạy |

### 5.2 Thứ tự ưu tiên thời gian máy

```
1. SOTA baselines            ← bạn đã chọn, đúng: R1.2 + R5.4/5.5 là chỗ mất phiếu nặng nhất
2. Bảng chính 74×7 view      ← bắt buộc đầy đủ
3. VQP correlation           ← dùng lại đúng dữ liệu của mục 2, gần như miễn phí
4. Ablation (mẫu 20-25)      ← rẻ vì đã cắt
5. Failure analysis          ← dùng lại dữ liệu mục 2, miễn phí
6. Robustness pose đầu vào   ← bỏ nếu thiếu giờ
```

Mục 3 và 5 gần như **không tốn thêm gì** vì dùng lại đầu ra của mục 2 — đó là chỗ thiết kế thí nghiệm khéo giúp tiết kiệm thật.

### 5.3 Chọn SOTA cho đúng — cảnh báo quan trọng

**[a] NerFACE, [b] NHA, [c] FATE đều cần video per-subject + tối ưu riêng từng người.** Dữ liệu của bạn là **ảnh tĩnh, 7 view/subject**. Đổ thời gian GPU vào ba method này nhiều khả năng **không ra kết quả dùng được**, vì thiếu đúng loại đầu vào chúng cần.

**Đề nghị chia hai nhánh:**

**Nhánh 1 — SOTA thật sự chạy được (ưu tiên thời gian GPU vào đây).** Tiêu chí chọn: đầu vào **một ảnh**, code + trọng số công khai, **không** cần huấn luyện per-subject. Hướng tìm: các method single-image full-head texture / albedo (dòng FitMe, AlbedoGAN, Relightify, PanoHead...). Cần khảo sát tính khả dụng thực tế trước khi cam kết giờ máy — **việc đầu tiên nên làm là chốt danh sách ứng viên và thử tải/chạy một cái**.

**Nhánh 2 — bảng định vị cho [a][b][c] (bắt buộc, không tốn GPU).** So sánh **điều kiện đầu vào / dữ liệu yêu cầu / thời gian tối ưu per-subject / đầu ra**, nêu rõ vì sao so trực tiếp là khác điều kiện. Bản cũ im lặng về ba method này — đó mới là lỗi. Nói tường minh thì reviewer chấp nhận; im lặng thì không.

**Baseline mạnh chạy CPU (rẻ, bắt buộc có):**
1. Naive UV resampling — panel giữa Fig. 2 hiện có, cần định lượng
2. **Symmetry mirroring thuần** — quan trọng: đây là bản thu gọn của chính IST, phải chứng minh phần hiệu chỉnh + phân xử mới là chỗ tạo giá trị
3. Classical inpainting (Telea, Navier–Stokes)
4. **UV-IDM + transfer = chính kiến trúc cũ của bạn** — nay thành baseline mạnh (CVPR 2024) mà TESSERA vượt qua

---

## 6. Hạ tầng tối thiểu

> Không phải mỹ phẩm: không có phần này thì không đo được TESSERA mạnh hơn ở đâu, và **VQP không tồn tại**.

| | Việc | Vì sao bắt buộc |
|---|---|---|
| **I1** | `render_views.py` — renderer offscreen xác định, 7 camera pose từ landmark 3D FLAME | Ảnh hiện tại tạo bằng **bấm chụp thủ công trong browser** ([`EvalApp.svelte:704`](src/EvalApp.svelte#L704)), không tái lập được và **không render được 7 góc**. Đã kiểm chứng: PyVista/VTK offscreen chạy tốt trên máy này; Open3D **không** chạy trên macOS (`EGL Headless is not supported`) — viết lớp render tách rời để đổi backend trên server |
| **I2** | Đo trong mask mặt | GT chứa tóc/áo/nền mà FLAME không có; phần lớn +10.07 dB đến từ lấp lỗ đen chứ không phải chất lượng texture. Đây đúng là điều R5 nghi ngờ |
| **I3** | CSIM vào bảng | [`compute_csim.py`](compute_csim.py) đã có sẵn nhưng chưa dùng; IST tác động thẳng vào identity |
| **I4** | `lib_align.py` | Procrustes/ICP/landmark đang lặp ở 3 file — sửa thuật toán sẽ phải sửa 3 nơi |
| **I5** | `requirements.txt` + khả chuyển server | Không env nào hiện có đủ; **không env nào có detector landmark 2D**. Đường dẫn qua CLI, backend render đổi bằng cờ, checkpoint theo subject |

**Camera rig V-Face** (74 subject có kết quả đều đủ 7 view):

| Camera | Yaw | Pitch | Vai trò |
|---|---|---|---|
| C1 | ≈ −90° | 0 | profile trái |
| **C4** | ≈ −35° | 0 | **ĐẦU VÀO** |
| C7 | 0° | 0 | frontal (GT hiện tại) |
| C10 | ≈ +35° | 0 | novel |
| C13 | ≈ +90° | 0 | profile phải |
| C17 | 0° | ≈ +30° | camera cao |
| C24 | 0° | ≈ −20° | camera thấp |

> Góc là ước lượng trực quan — chốt lại bằng đối chiếu landmark hoặc thông số rig chính thức.

---

## 7. Lộ trình

```
GIAI ĐOẠN 0 — Gỡ rủi ro
  ⚠️ Xác nhận chạy lại được UV-IDM (k mẫu)?          ← chặn R_gen của SERF
  ⚠️ Chốt danh sách SOTA single-image khả dụng       ← chặn ngân sách GPU
  I4 lib_align.py · I5 requirements + khả chuyển

GIAI ĐOẠN 1 — Hạ tầng đo
  I1 render_views.py (7 camera) · I2 mask mặt · I3 CSIM
  → Kiểm tra: dựng lại số cũ dưới protocol mới để có mốc so sánh

GIAI ĐOẠN 2 — TESSERA  (đường găng)
  CTT  → SERF → IST → SAPF        ← thứ tự bắt buộc, mỗi cái cần cái trước
  (phi cứng có ràng buộc đối xứng nếu còn giờ)

GIAI ĐOẠN 3 — Chứng minh
  Bảng chính 74×7 view  →  VQP correlation (miễn phí)
  Ablation (mẫu 20-25)  ·  SOTA + baseline  ·  Failure analysis (miễn phí)

GIAI ĐOẠN 4 — Viết bài
  §4 toàn bộ + Phụ lục B
```

**Đường găng:** `I1 → I2 → CTT → SERF → IST → SAPF → bảng chính → VQP → viết`

**Ba rủi ro:**
1. **Không chạy lại được UV-IDM** → mất $\mathcal{R}_{\text{gen}}$ đa mẫu; dùng dự phòng ở §2.3. *Xác nhận ngay.*
2. **Đổ giờ GPU vào [a][b][c] rồi không ra kết quả** vì sai loại đầu vào → chốt danh sách single-image trước khi cam kết máy.
3. **Số liệu giảm dưới protocol nghiêm ngặt** → đây là điều dự tính; TESSERA phải bù lại. Nếu không bù đủ, vẫn nộp bộ số nghiêm ngặt kèm phân tích trung thực — bộ số cũ không sống nổi qua vòng phản biện thứ hai.

---

## Phụ lục A — Chỗ bài viết đang tự làm yếu mình

> Sửa vì đang **mô tả kém hơn** kiến trúc thật.

**A1. Cơ chế phủ 100% bị giải thích sai theo hướng làm nó nghe yếu đi.** Bài viết *"the remaining 1% receive color via the seam dilation pass"* — nghe như phải nhờ hậu xử lý cứu. Thực tế code có **fallback 3 tầng ở mức vertex** (barycentric → nearest-corner cho tam giác suy biến → clamp UV), đảm bảo phủ **bằng thiết kế**. Trong TESSERA, tính chất này thuộc về CTT và phải được phát biểu như một bảo đảm, không phải một vá lỗi.

**A2. Thiết kế thí nghiệm có tính chất rất sạch mà bài không nói.** Baseline và method dùng **chung y hệt geometry, chỉ khác texture** — biến được cô lập hoàn hảo. Phải nêu rõ, đây là điểm mạnh thực nghiệm.

**A3. Các cơ chế robustness bị chôn vùi.** Tự dò quy ước trục (4 hoán vị chọn theo MSE), fallback tam giác suy biến, cap khoảng cách thích ứng, phát hiện pháp tuyến lật, guard chống scale cực đoan — khiến pipeline chạy được trên 74 subject không cần chỉnh tay. Với hội nghị thiên hệ thống như SOICT đây là điểm cộng.

**A4. Mô tả `d_cap` nói quá so với code.** Bài mô tả $d_{99} = \min(d_{99}^{\text{raw}}, d_{\text{cap}})$ nhưng code không có phép `min` ([:607](align_and_transfer.py#L607)); `BAKE_DIST_CAP` tính rồi không dùng ([:692](align_and_transfer.py#L692)). SERF thay hẳn phần này — viết lại theo thiết kế mới.

**A5. Claim `35×` speedup sẽ không còn đúng.** TESSERA chậm hơn có chủ ý. Thay bằng **bảng đánh đổi chất lượng–thời gian** theo $k$ và độ phân giải.

---

## Phụ lục B — Việc vặt, làm sau cùng

- [ ] `main.tex` thiếu `\author{}`
- [ ] 4 hình `overview/texture/align/qualitative.png` không tồn tại (dựng lại theo TESSERA)
- [ ] $R_u$ thật 256 (bài ghi 512); $K$ thật 5,118 (bài ghi 5,023)
- [ ] Sửa phạm vi claim `∀uv ∈ Ω_UV` (2.59% texel vẫn trống)
- [ ] Bỏ đoạn chọn frame theo yaw — không có code (`grep "yaw" *.py` không ra gì)
- [ ] Ghi backbone LPIPS (AlexNet), điều kiện chiếu sáng, xử lý nền
- [ ] Bảng ablation cũ không truy được nguồn (`ablation_output/` không tồn tại; [`run_ablation.py:103`](run_ablation.py#L103) so **atlas UV** với **ảnh chụp mặt**) — bị thay hoàn toàn
- [ ] Script Wilcoxon sinh p-value từ CSV
- [ ] `App.svelte` mang 2 phiên bản (dòng 1–301 là script cũ bị comment)
- [ ] `.gitignore` chưa loại dữ liệu nặng
- [ ] `output_batch/` mất `deca_textured.obj` ở cả 74 dir

---

## Phụ lục C — Số liệu tham chiếu

**`eval_report/metrics.csv` (n = 74):** `deca_psnr` 13.735 ± 1.713 · `ours_psnr` 23.804 ± 2.021 · `deca_ssim` 0.447 ± 0.031 · `ours_ssim` 0.602 ± 0.050 · `deca_lpips` 0.513 ± 0.040 · `ours_lpips` 0.350 ± 0.087 · 74/74 cải thiện cả 3 metric.

**Dữ liệu:** `results_C4_final/` 76 dir (có $T_{partial}$ **đã đúng UV space FLAME**) · `uv-idm-output/` 131 atlas 256² · `output_batch/` 74 dir · `polyface_09042026/` 121 subject, **114 đủ 7 view** · giao với tập có kết quả = **74** · `test/cheo1` 11 frame.

**Không có trong repo:** model UV-IDM (*rủi ro #1*) · video ngoài `cheo1` · FLAME decoder · `ablation_output/` · 4 hình của bài.
