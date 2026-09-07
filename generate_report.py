import csv
import math
import os

# Cấu hình đường dẫn file
CSV_PATH = "eval_report/metrics.csv"
HTML_PATH = "eval_report/report.html"

def fmt(v, decimals=4):
    if math.isnan(v): return "—"
    return f"{v:.{decimals}f}"

def delta_style(v, higher_better=True):
    if math.isnan(v): return "color:#888"
    if (v > 0) == higher_better:
        return "color:green;font-weight:bold"
    return "color:red"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[LỖI] Không tìm thấy file {CSV_PATH}")
        return

    # Đọc dữ liệu từ metrics.csv
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Khởi tạo các biến để tính tổng (Aggregate)
    sum_deca_psnr, sum_deca_ssim, sum_deca_lpips = 0.0, 0.0, 0.0
    sum_ours_psnr, sum_ours_ssim, sum_ours_lpips = 0.0, 0.0, 0.0
    valid_count = 0

    table_rows = ""

    for r in rows:
        vid = r["vid"]
        # Mặc định GT là có sẵn (✓)
        gt_tag = "✓" 
        
        def parse_val(key):
            try: return float(r[key])
            except: return float("nan")

        dp, ds, dl = parse_val("deca_psnr"), parse_val("deca_ssim"), parse_val("deca_lpips")
        op, os_, ol = parse_val("ours_psnr"), parse_val("ours_ssim"), parse_val("ours_lpips")
        
        # Tính delta
        dP = op - dp
        dS = os_ - ds
        dL = ol - dl

        # Cộng dồn để tính trung bình
        if not math.isnan(dp) and not math.isnan(op):
            sum_deca_psnr += dp
            sum_deca_ssim += ds
            sum_deca_lpips += dl
            sum_ours_psnr += op
            sum_ours_ssim += os_
            sum_ours_lpips += ol
            valid_count += 1

        # Tạo từng dòng của bảng Per-Subject Details
        table_rows += f"""
        <tr>
          <td><a href="subjects/{vid}.jpg">{vid}</a></td><td>{gt_tag}</td>
          <td>{fmt(dp, 2)}</td><td>{fmt(ds, 4)}</td><td>{fmt(dl, 4)}</td>
          <td>{fmt(op, 2)}</td><td>{fmt(os_, 4)}</td><td>{fmt(ol, 4)}</td>
          <td style="{delta_style(dP, True)}">{fmt(dP, 2)}</td>
          <td style="{delta_style(dS, True)}">{fmt(dS, 4)}</td>
          <td style="{delta_style(dL, False)}">{fmt(dL, 4)}</td>
        </tr>"""

    # Tính toán Aggregate Results
    if valid_count > 0:
        agg_deca_psnr = sum_deca_psnr / valid_count
        agg_deca_ssim = sum_deca_ssim / valid_count
        agg_deca_lpips = sum_deca_lpips / valid_count
        
        agg_ours_psnr = sum_ours_psnr / valid_count
        agg_ours_ssim = sum_ours_ssim / valid_count
        agg_ours_lpips = sum_ours_lpips / valid_count
        
        agg_delta_psnr = agg_ours_psnr - agg_deca_psnr
        agg_delta_ssim = agg_ours_ssim - agg_deca_ssim
        agg_delta_lpips = agg_ours_lpips - agg_deca_lpips
    else:
        agg_deca_psnr = agg_deca_ssim = agg_deca_lpips = float("nan")
        agg_ours_psnr = agg_ours_ssim = agg_ours_lpips = float("nan")
        agg_delta_psnr = agg_delta_ssim = agg_delta_lpips = float("nan")

    # Template HTML (Đã loại bỏ phần hiển thị thư viện ảnh ở cuối)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>2D Texture Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; text-align: right; }}
    th {{ background: #f2f2f2; text-align: center; }}
    .agg {{ background: #e8f4e8; font-weight: bold; }}
  </style>
</head>
<body>
<h1>2D Evaluation: DECA vs Ours (HMI)</h1>
<h2>Aggregate Results (n={valid_count})</h2>
<table>
  <tr><th>Method</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th></tr>
  <tr><td>DECA</td><td>{fmt(agg_deca_psnr, 2)}</td><td>{fmt(agg_deca_ssim, 4)}</td><td>{fmt(agg_deca_lpips, 4)}</td></tr>
  <tr class="agg"><td>Ours (HMI)</td><td>{fmt(agg_ours_psnr, 2)}</td><td>{fmt(agg_ours_ssim, 4)}</td><td>{fmt(agg_ours_lpips, 4)}</td></tr>
  <tr><td>Δ (Ours − DECA)</td><td style="{delta_style(agg_delta_psnr, True)}">{fmt(agg_delta_psnr, 2)}</td><td style="{delta_style(agg_delta_ssim, True)}">{fmt(agg_delta_ssim, 4)}</td><td style="{delta_style(agg_delta_lpips, False)}">{fmt(agg_delta_lpips, 4)}</td></tr>
</table>
<h2>Per-Subject Details</h2>
<table>
  <tr><th>Subject</th><th>GT</th><th colspan="3">DECA</th><th colspan="3">Ours</th><th colspan="3">Δ (Ours−DECA)</th></tr>
  <tr><th></th><th></th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>ΔPSNR</th><th>ΔSSIM</th><th>ΔLPIPS</th></tr>
  {table_rows}
</table>
</body>
</html>"""

    # Ghi ra file HTML
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"Đã cập nhật và tạo thành công báo cáo tại: {HTML_PATH}")

if __name__ == "__main__":
    main()