# Reviewer Comments

## Reviewer #1

### Questions

#### 1. Summary of the paper

* The authors proposed a method that performs 4D face avatar reconstruction from a monocular video.

#### 2. Strong points

* The paper demonstrated quantitative and qualitative improvements w.r.t. the baseline.

#### 3. Weak points and comments

* Since your method is 4D, evidence supporting the temporal consistency of the proposed method is lacking. Including both qualitative examples and quantitative metrics for temporal stability would strengthen the paper and claims.
* A quantitative comparison with the current state of the art is missing. The only comparison is against a baseline using partial textures, which is relatively weak and does not convincingly demonstrate the advantages of the proposed approach.
* On top of that, the quantitative evaluation methodology is not entirely clear. In particular, how is pixel-level alignment between the rendered images and the ground truth ensured?
* Based on the examples shown in Figure 4, the rendered faces appear to be imperfectly aligned with the ground-truth images, which could significantly affect pixel-wise evaluation metrics.
* Finally, the paper would benefit from a more thorough discussion of the remaining artifacts, failure cases and the primary limitations of the proposed method, which will help to identify directions for future improvement.

#### 5. Overall score

* **Weak Reject**

---

## Reviewer #3

### Questions

#### 1. Summary of the paper

* The paper proposes an end-to-end pipeline for complete-texture 4D face avatar reconstruction from a monocular video. Starting from a monocular 3D face reconstruction method, it takes one near-frontal identity frame, bakes a partial UV texture, completes missing regions using a UV-space diffusion inpainting model, and then transfers the completed texture to an animation mesh with a geometry-aware cross-topology transfer pipeline based on Procrustes alignment, multi-pass ICP, vertex color transfer, vectorized UV baking, and seam blending.
* The final output is a 4D avatar with a single completed texture reused across expression-varying frames for free-viewpoint rendering.
* The method is evaluated on 74 subjects from V-Face, reporting improvements over a partial-texture baseline in PSNR, SSIM, and LPIPS, together with an ablation study and runtime analysis.

#### 2. Strong points

* The paper addresses a real and practical limitation of monocular face reconstruction: standard methods recover geometry reasonably well but produce incomplete, view-baked textures that are unsuitable for free-view rendering. The problem formulation around UV-domain coverage is clear and motivates the pipeline well.

* The method is systematically engineered: the combination of UV inpainting and cross-topology transfer is technically coherent, and the cross-topology module is described in sufficient detail, including alignment, confidence weighting, rasterization, and seam handling.

* The paper provides solid empirical gains over the baseline on all 74 evaluated subjects, with a large average PSNR improvement (+10.07 dB), SSIM increase, LPIPS reduction, and supporting ablations that show the contribution of ICP, soft-confidence weighting, and blending.

#### 3. Weak points and comments

* The main weakness is limited novelty at the algorithmic level. Much of the pipeline combines existing components—monocular face reconstruction, UV diffusion inpainting, ICP-based alignment, and rasterization—into an application pipeline. The contribution is therefore stronger in integration/engineering than in fundamentally new vision methodology.

* The evaluation is somewhat narrow. The experiments use a controlled subset of V-Face with neutral expression, studio lighting, and a specific camera-pose setup (C4 to C7), so it remains unclear how well the method generalizes to wilder videos, stronger occlusions, accessories, larger pose changes, or more diverse illumination.

* The 4D claim is somewhat limited: the output is a static completed texture reused across time, while dynamic appearance changes such as wrinkles, specularities, and expression-dependent texture changes are explicitly not modeled. Thus, the work is closer to animated textured avatar reconstruction than truly dynamic 4D appearance modeling.

#### 5. Overall score

* **Weak Accept**

---

## Reviewer #5

### Questions

#### 1. Summary of the paper

* This paper proposes a pipeline for constructing a fully textured facial avatar from a monocular video. The method combines monocular face reconstruction, UV-space diffusion inpainting, geometric registration using Procrustes alignment and ICP, cross-topology texture transfer, and vectorized UV rasterization.
* A completed texture is generated once and then applied to a sequence of meshes with different facial expressions, resulting in what the authors describe as a “4D avatar.”
* Experiments on 74 subjects from the V-FACE dataset show substantial improvements over a baseline using incomplete textures.
* The proposed idea is practically relevant, and the paper is generally easy to follow. However, in its current form, the paper does not provide sufficient experimental evidence to support its main claims regarding 4D reconstruction, identity preservation, and free-viewpoint rendering.

#### 2. Strong points

* The problem is practically meaningful and easy to understand.
* The proposed method forms a complete processing pipeline. It goes beyond texture inpainting by also addressing topology mismatch, geometric alignment, texture rebaking, and seam filling.
* The paper includes both a baseline comparison and an ablation study.

#### 3. Weak points and comments

* The paper presents the method as a 4D facial avatar reconstruction system, but the experiments mainly evaluate it as a 3D avatar reconstruction task.
* The method is largely built upon a combination of existing components.
* Only one novel view is evaluated, using C4 as input and C7 as ground truth. More viewpoints with different angles should be included.
* The paper lacks performance comparisons with relevant 4D facial avatar methods [a][b][c].
* The current baseline is relatively weak, which may make the reported improvements appear larger than they actually are.

### References

* [a] Gafni, Guy, et al. "Dynamic neural radiance fields for monocular 4d facial avatar reconstruction." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2021.
* [b] Grassal, Philip-William, et al. "Neural head avatars from monocular RGB videos." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2022.
* [c] Zhang, Jiawei, et al. "Fate: Full-head Gaussian avatar with textural editing from monocular video." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2025.

#### 5. Overall score

* **Weak Reject**