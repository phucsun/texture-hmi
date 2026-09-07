"""
Generate UV Texture Progression figure for paper Section 3.2/3.3.
Layout: [T_baked partial] → [T_IDM after inpainting] → [T_complete after transfer]
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as FancyArrowPatch
from PIL import Image

SUBJECT = "V000190"
SUBJECT_FULL = "V000190_v000190_s002_l2_e01_c4"
SIZE = 512  # display size for each panel

paths = {
    "T_baked":   f"output_batch/{SUBJECT}/deca_texture.png",
    "T_IDM":     f"uv-idm-output/output_vkist_uv/{SUBJECT}_S002_L2_E01_C4.png",
    "T_complete": f"results_C4_final/{SUBJECT_FULL}/{SUBJECT_FULL}.png",
}

labels = [
    r"$\mathbf{T}_\mathrm{baked}$" + "\n(partial UV, front-facing only)",
    r"$\mathbf{T}_\mathrm{IDM}$" + "\n(after inpainting)",
    r"$\mathbf{T}_\mathrm{complete}$" + "\n(after texture transfer)",
]

imgs = []
for key in ["T_baked", "T_IDM", "T_complete"]:
    im = Image.open(paths[key]).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    imgs.append(np.array(im))

# ── layout ──────────────────────────────────────────────────────────────────
fig_w = 3 * SIZE / 100 + 2.0   # inches: 3 panels + arrow gaps
fig_h = SIZE / 100 + 1.0

fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h),
                         gridspec_kw={"wspace": 0.05})

for ax, img, lbl in zip(axes, imgs, labels):
    ax.imshow(img)
    ax.set_title(lbl, fontsize=9, pad=6)
    ax.axis("off")

# arrows via annotate on the figure
for x_start, x_end in [(0.345, 0.375), (0.670, 0.700)]:
    axes[0].annotate(
        "",
        xy=(x_end, 0.52), xycoords="figure fraction",
        xytext=(x_start, 0.52), textcoords="figure fraction",
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.8),
    )

out_path = "uv_texture_progression.pdf"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
out_png = "uv_texture_progression.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}  and  {out_png}")
plt.close()
