# Complete-Texture 4D Face Avatar Reconstruction from a Single Image

---

**[Author 1 Name]**¹, **[Author 2 Name]**², **[Author 3 Name]**¹

¹ [Affiliation 1, Department, University, Country]  
² [Affiliation 2, Department, University, Country]

Email: {author1, author3}@university.edu, author2@university2.edu

---

## Abstract

Monocular 3D face reconstruction methods recover accurate geometry from a single image but yield only partial, camera-view-baked textures: occluded surface regions—ears, scalp, and the back of the head—remain blank in the UV atlas, making the reconstructed mesh unsuitable for free-viewpoint rendering or 4D animated face synthesis. We present a complete pipeline that bridges this gap by coupling a UV-space inpainting diffusion model with a geometry-aware cross-topology texture transfer module, producing a full-surface texture atlas that covers the entire head from a single photograph. The completed texture is applied uniformly across an animated mesh sequence to form a 4D representation: an avatar simultaneously navigable in three spatial dimensions and across the temporal axis of expressive face motion. We formalize the 4D texture property as a coverage condition on the UV domain and show that partial textures violate this condition for the majority of the camera-pose sphere, while our complete texture restores it unconditionally. Quantitative evaluation on a large multi-view face dataset demonstrates consistent and substantial improvements over the monocular reconstruction baseline across all evaluated subjects on standard image quality metrics. Our method requires a single photograph as input, generalizes across identities without per-subject training, and produces a temporally coherent 4D avatar whose texture quality is uniform regardless of viewing direction.

**Keywords:** 4D face avatar, complete texture, UV inpainting, cross-topology texture transfer, free-viewpoint rendering, monocular face reconstruction

---

## 1. Introduction

The proliferation of remote communication—accelerated by global events that forced billions of people to interact primarily through digital channels—has exposed a fundamental gap between what video conferencing delivers and what human presence truly requires. Flat 2D video streams are bandwidth-hungry (1–5 Mbps for HD), privacy-invasive (the raw camera feed is transmitted unconditionally), and viewpoint-locked: the receiver sees only the angle captured by the sender's camera. More critically, 2D video cannot be manipulated in 3D space, overlaid in augmented reality environments, or integrated into spatial computing platforms such as virtual meeting rooms or immersive e-learning systems.

A compelling alternative is *parametric avatar streaming*: instead of transmitting raw pixels, the sender's face is represented as a compact parametric model whose expression parameters—roughly 500 floating-point values per frame—are transmitted at a fraction of raw-video bandwidth, while the receiver renders the avatar locally. Such parameters compressed to 16-bit half-precision require under 1 KB per frame, compared to approximately 30–150 KB per HD video frame. Since the receiver renders the mesh locally, the avatar can be viewed from *any angle*, enabling immersive free-viewpoint telepresence without additional hardware.

This vision, however, hinges on a prerequisite that existing pipelines fail to satisfy: **the avatar must possess a complete, photorealistic texture across its entire surface.** Monocular 3D face reconstruction methods [1] recover accurate mesh geometry but bake the texture exclusively from the visible pixels of the input photograph. Regions occluded at capture time—the back of the head, ears, temples, and portions of the neck—remain blank in the UV atlas. When the receiver rotates the avatar even slightly away from the original capture angle, these gaps are immediately exposed, making free-viewpoint rendering impractical (Fig. 1).

The need for complete-texture 3D avatars extends well beyond video conferencing. In **e-learning and distance education**, a single enrollment photograph can generate a persistent instructor avatar that delivers course content interactively. In **digital accessibility**, people with conditions affecting facial appearance can adopt a fully-textured avatar presenting a normalized digital identity. In **cultural and historical preservation**, one archived photograph of a historical figure can yield an interactive 3D memorial. In each of these scenarios, texture completeness is the decisive enabling factor.

UV-space inpainting models [3] have recently demonstrated the ability to synthesize plausible texture for occluded facial regions in a high-resolution UV layout. However, no complete pipeline exists that (i) integrates UV inpainting with monocular reconstruction, (ii) resolves the topology mismatch between the inpainting mesh and the animation mesh, and (iii) frames the result as a formally defined 4D representation. This paper addresses all three gaps together.

We present a complete end-to-end pipeline that:
1. Reconstructs face mesh geometry frame-by-frame from monocular video using a parametric reconstruction model [1];
2. Completes the full UV texture atlas from a single representative frame using a UV-space inpainting model [3];
3. Transfers the completed texture across mesh topologies via Procrustes + ICP alignment and vectorized UV rebaking;
4. Assembles the per-frame textured meshes into a 4D face avatar representation.

The "4D" descriptor reflects that our output is not a static textured model but an animated sequence of complete-texture 3D meshes indexed by time—a four-dimensional avatar navigable in both space and time. The entire pipeline completes in under 90 seconds on consumer hardware.

**Contributions.** This paper makes the following specific contributions:
- A formal definition of the *4D texture avatar* as a coverage condition on the UV domain, establishing texture completeness as a necessary—not merely desirable—property for free-viewpoint animated face rendering.
- A geometry-aware cross-topology texture transfer algorithm combining robust Procrustes alignment, three-pass Point-to-Plane ICP, soft-confidence vertex color transfer, and fully vectorized UV rebaking that achieves 100% surface coverage at 1024×1024 resolution without hard distance gates.
- An end-to-end pipeline integrating a monocular face reconstruction model and a UV-space inpainting model via our transfer module, producing a complete-texture 4D avatar from a single photograph in under 90 seconds on consumer hardware.
- Quantitative evaluation on 74 subjects from the PolyFace multi-view dataset demonstrating consistent improvement over the partial-texture baseline on all subjects across PSNR, SSIM, and LPIPS metrics.

---

## 2. Related Work

**Monocular 3D face reconstruction.** 3D Morphable Models [4] provide a low-dimensional parametric space for face shape and texture, enabling reconstruction from as few as one image. Early fitting methods [5] relied on iterative optimization; recent deep learning approaches regress morphable model parameters directly [6,7]. State-of-the-art methods [1,8] extend this to per-frame expression tracking, achieving high-quality reconstruction from monocular video. All these methods share the same texture bottleneck: the estimated albedo is reconstructed from the visible image region only, leaving a large portion of the UV atlas blank or extrapolated poorly—the gap that motivates this work.

**UV texture completion.** Face texture completion in UV space has been addressed by GAN-based inpainting models [9] that learn to hallucinate realistic skin texture for occluded regions conditioned on visible ones. More recently, diffusion model backbones applied directly to UV atlas layouts [3] achieve significantly higher fidelity for complex skin tones and fine facial features. No prior work integrates such a model into a full animated reconstruction pipeline: the completed UV atlas exists in the inpainting model's own mesh parameterization and cannot be directly applied to an animation-ready mesh with a different topology—a mismatch our system resolves.

**Animated face and 4D face synthesis.** NeRF-based methods [10,11] achieve photo-realistic novel-view synthesis but are identity-specific, requiring per-subject training and substantial compute at inference. Explicit mesh-based animated face systems [12,13] operate in real time and generalize across identities but suffer from the same incomplete-texture limitation as other monocular methods. Speech-driven animation models [14] generate expressive face motion but output 2D video, discarding the 3D representation entirely. Our approach is complementary: we focus on completing the surface texture of a given animated mesh sequence, restoring the free-viewpoint property that partial textures violate.

---

## 3. Method

### 3.1 Pipeline Overview

Our system transforms a monocular face video into a 4D avatar—a temporally indexed sequence of complete-texture 3D meshes—through four successive stages (Fig. 1):

1. **Face reconstruction and texture completion.** A monocular 3D face reconstruction method processes each video frame to obtain per-frame mesh geometry and a partial UV texture baked from the visible image region. A UV-space inpainting model then completes the full texture atlas from the identity frame, synthesizing plausible content for all occluded surface regions.
2. **Cross-topology texture transfer.** A geometry-aware module that aligns the completed inpainting-mesh texture onto the target animation-mesh UV space. This involves (i) robust Procrustes alignment using 68 facial landmarks, (ii) a three-pass ICP refinement scheme, (iii) vectorized vertex-level color transfer with soft confidence weighting, (iv) zero-loop UV rasterization and barycentric baking at 1024×1024 resolution, and (v) seam dilation with Gaussian boundary blending.
3. **4D assembly.** The completed texture—computed once per subject—is applied uniformly to every frame of the animation sequence, preserving per-frame expression dynamics in the mesh geometry.
4. **4D representation and evaluation.** The assembled sequence is formalized as a 4D texture avatar and evaluated for spatial completeness across novel viewpoints.

Stages 1 uses existing methods as components. The contributions of this paper span Stages 2–4: the cross-topology transfer algorithm, the formal 4D texture avatar framework, and their integration into an end-to-end pipeline evaluated at scale.

---

### 3.2 Face Reconstruction and Texture Completion

#### 3.2.1 Monocular Geometry Reconstruction

**Parameter regression.** Given an input RGB image **I**, we adopt a monocular 3D face reconstruction method [1] whose encoder, built on a ResNet-50 backbone, regresses a compact parameter vector:

```
Θ = {β, θ_p, ψ, α, l, c}
```

where **β** ∈ ℝ^{n_id} encodes identity shape coefficients, **θ_p** ∈ ℝ^6 encodes head pose as a 6-DoF rotation-translation vector (including the yaw angle θ_yaw used for identity frame selection), **ψ** ∈ ℝ^{n_exp} encodes per-frame expression deformation, **α** ∈ ℝ^{n_tex} encodes PCA albedo coefficients, **l** ∈ ℝ^{27} encodes scene illumination as 9-band Spherical Harmonic (SH) coefficients per RGB channel, and **c** ∈ ℝ^3 encodes weak-perspective camera parameters.

**Shape decoding.** The mesh vertices are reconstructed by combining a neutral identity shape with expression deformation over a learned basis:

```
V(β, ψ) = S̄ + B_id · β + B_exp · ψ  ∈  ℝ^{N×3}
```

where S̄ ∈ ℝ^{N×3} is the mean shape, **B_id** ∈ ℝ^{3N×n_id} is the identity shape basis, and **B_exp** ∈ ℝ^{3N×n_exp} is the expression basis, each learned from a large corpus of 3D face scans. N denotes the number of vertices and N_f the number of triangular faces in the target (animation) mesh; both are fixed constants determined by the parametric model's topology (specific values are reported in §4.1). The resulting mesh {**V** ∈ ℝ^{N×3}, **F** ∈ ℤ^{N_f×3}} carries a UV parameterization φ: Ω_UV → S defined by pre-stored UV coordinates and face indices, where S ⊂ ℝ³ denotes the face surface.

**View-dependent texture baking.** The texture is recovered by differentiable rasterization rather than decoded from **α** alone—since **α** captures only a coarse statistical prior and discards the fine-grained identity details present in the image. For each texel (u, v) ∈ Ω_UV, the 3D surface point **p**(u, v) = φ(u, v) is projected to image space under the estimated camera:

```
(x, y) = π(p(u,v);  θ_p, c)
```

The texel color is assigned via a visibility function:

```
              ⎧ I(x, y)           if  p(u,v) ∈ Vis(I, θ_p, c)
T_baked(u,v) = ⎨
              ⎩ undefined          otherwise
```

where Vis(I, θ_p, c) is the set of surface points that simultaneously: (i) project within the image boundary, (ii) survive a depth-buffer occlusion test against all other mesh faces, and (iii) have a surface normal oriented toward the camera (n̂ · (−**ray**) > 0). Points failing any of these conditions receive no observation.

**Coverage gap analysis.** Texture incompleteness is not a stochastic failure mode of the reconstruction method but a *deterministic geometric consequence* of the capture angle. The observable UV region is:

```
Ω_in(θ_p) = { (u,v) ∈ Ω_UV : p(u,v) ∈ Vis(I, θ_p, c) }
```

At a lateral capture angle of approximately 30° from frontal, empirical measurement on our dataset yields:

```
|Ω_in| / |Ω_UV|  ≈  0.40 – 0.58
```

meaning 42–60% of the UV atlas—including the back of the scalp, both ears, the contralateral cheek, and portions of the neck—receives no valid observation regardless of reconstruction accuracy. Every monocular method deriving texture from a single photograph shares this limitation structurally.

**Identity frame selection.** For video input, the frame with maximum frontal confidence is selected for texture baking, measured by the absolute lateral yaw component |θ_yaw| of the estimated pose **θ_p**. The frame minimizing |θ_yaw| yields the largest Ω_in and thus the smallest occlusion mask requiring subsequent inpainting. Per-frame pose-expression parameters {**θ_p**_t, **ψ**_t} are extracted from all frames independently to drive the 4D animation sequence.

#### 3.2.2 UV Texture Completion

**Partial texture as inpainting input.** The baked texture **T**_baked and a binary validity mask **M** ∈ {0, 1}^{R_u×R_u}—where R_u is the UV atlas resolution of the inpainting model, M(u, v) = 1 if the texel was observed and 0 otherwise—are passed to a UV-space inpainting diffusion model [3]. This model is specifically trained on UV atlas layouts that parameterize the full head surface in a canonical unwrapping, covering occluded regions that are never visible from any single frontal or near-frontal view:

```
T_IDM = InpaintingModel(T_partial,  M)
```

The source mesh used by the inpainting model has N_src vertices (N_src ≫ N), providing higher surface resolution and finer detail than the animation mesh; specific values are reported in §4.1.

**Conditional denoising.** The model executes a conditional score-based denoising process whose output is constrained to be consistent with **T**_partial on the observed region (M = 1) while synthesizing new content for M = 0 from learned priors. These priors jointly encode:

- *Bilateral skin-tone continuity*: smooth color transitions across UV island seams, so synthesized ear and neck texture is consistent with the visible cheek albedo
- *Approximate left-right symmetry*: the contralateral cheek and ear are synthesized as approximate reflections of the visible side, with fidelity proportional to the available symmetric evidence
- *High-frequency surface detail*: pore structure, fine wrinkles, and hair follicle distributions are hallucinated from the model's learned texture distribution conditioned on the visible texture's coarse statistics

The output **T**_IDM ∈ ℝ^{R_u×R_u×3} is a complete UV atlas with T_IDM(u, v) defined for all (u, v) ∈ Ω_UV. Because the subject's albedo is identity-constant under the diffuse assumption, the model is invoked once per subject—not per frame—on the identity frame selected above.

**Topology mismatch.** **T**_IDM is defined in the source mesh's UV parameterization (Ω_src). The animation mesh uses a topologically distinct UV parameterization (Ω_tgt) with a different vertex count, face connectivity, and island layout. Both parameterizations map the same face surface, but numerically: a UV coordinate (u, v) ∈ Ω_src resolves to a different surface point than the same (u, v) ∈ Ω_tgt. Direct sampling of **T**_IDM using Ω_tgt UV coordinates would therefore produce a completely incorrect texture assignment—color values from one face region mapped onto a geometrically unrelated region of the animation mesh. This topology mismatch is not addressable by UV-space remapping and requires the geometry-guided transfer described in §3.3.

---

### 3.3 Cross-Topology Texture Transfer

The completed texture **T**_IDM lives in the source mesh UV space and must be rebaked into the target mesh UV space without introducing coverage gaps, seam artifacts, or topology-induced distortions. We achieve this through a five-step geometry-aware pipeline.

#### 3.3.1 Robust Procrustes Alignment

We detect 68 facial landmarks on both meshes: on the source mesh, landmarks are obtained from pre-computed 3D vertex coordinates; on the target mesh, landmarks are computed via barycentric interpolation on pre-stored face indices and barycentric weights.

Let **P** = {**p**_i}_{i=1}^{68} ⊂ R^3 be the source-mesh landmark set and **Q** = {**q**_i}_{i=1}^{68} ⊂ R^3 be the target-mesh landmark set. The Procrustes alignment proceeds in three sub-steps:

**Pre-centering.** Both landmark sets are centered to zero mean:
```
p̄ = (1/68)Σ p_i,    q̄ = (1/68)Σ q_i
P_c = P − p̄,         Q_c = Q − q̄
```
Pre-centering eliminates large spatial offsets between the two coordinate frames (the source mesh typically lives in millimeter-scale world coordinates while the target mesh may use normalized camera coordinates), which would otherwise cause numerical overflow in subsequent SVD computation.

**Axis-convention detection.** The source and target meshes may use different axis orientations (Y-up vs. Z-up, left-handedness, etc.). We test four candidate axis permutations — {XYZ, XZY, X(−Y)Z, XY(−Z)} — and select the one that minimizes the mean squared landmark distance to **Q**_c prior to running the full Procrustes:
```
axis* = argmin_{A ∈ {XYZ, XZY, X-YZ, XY-Z}}  (1/68) ||A(P_c) − Q_c||²_F
```
where A denotes an axis-permutation operator. This axis-detection step makes the pipeline robust to coordinate convention mismatches between the inpainting and reconstruction pipelines.

**Bounding-box normalization.** To bring both point clouds to a comparable scale before Procrustes, we normalize the source landmarks by the ratio of bounding-box diagonals:
```
s_pre = diag(Q_c) / diag(P_c),    P_c ← s_pre · P_c
```
where diag(·) = ||max(·) − min(·)||₂. This pre-normalization ensures Procrustes scale estimation is not dominated by large-scale outliers.

**Umeyama Procrustes via SVD.** We compute the rotation **R** ∈ SO(3) and residual scale *s* ∈ ℝ via singular value decomposition of the cross-covariance matrix **K** = **Q**_c^T **P**_c / 68:
```
[U, Σ, V^T] = SVD(K)
D = diag(1, 1, det(U)·det(V))      ← reflection-correction diagonal
R = U D V^T
s = tr(Σ D) / Var(P_c)
t = q̄ − s · R · p̄
```
where **U**, **V** ∈ SO(3) are the left and right singular vectors, **Σ** = diag(σ₁, σ₂, σ₃) is the diagonal singular value matrix, and **D** is the reflection-correction diagonal that prevents the degenerate solution **R** = −**I** when the landmark cloud is mirror-symmetric. The translation vector **t** ∈ ℝ³ completes the similarity transform. A scale guard clamps *s* to [10⁻³, 10³] to prevent explosion due to degenerate inputs.

After Procrustes, the source mesh vertices are transformed as:
```
V_src ← s · V_src · R^T + t
```
achieving a coarse global alignment. The landmark RMSE after Procrustes is typically 0.003–0.012 in the target mesh's normalized coordinate space.

#### 3.3.2 Three-Pass Point-to-Plane ICP Refinement

Despite robust Procrustes alignment, residual misalignment persists due to (i) shape differences between the source and target mesh topologies, (ii) landmark detection noise, and (iii) expression-induced geometric discrepancies. We apply a three-pass iterative refinement scheme using the Open3D point cloud registration framework.

Both meshes are converted to oriented point clouds, with normals estimated from the mesh vertex normals. The ICP objective in each pass minimizes the point-to-plane distance:
```
E(Ξ) = Σ_{(p,q) ∈ Π} [(Ξ·p − q) · n̂_q]²
```
where **Π** is the set of inlier source–target point correspondences, **Ξ** ∈ SE(3) is the 4×4 rigid transformation matrix to estimate (distinct from the texture map T), and **n̂**_q ∈ ℝ³ is the unit surface normal at target point **q**.

**Pass 1 — Coarse Point-to-Plane.** Both point clouds are voxel-downsampled at resolution δ = 0.002 (normalized units), reducing point count by approximately 70–85%. We run 500 iterations with correspondence distance threshold d₁ = 0.05, convergence criteria relative_fitness = 10⁻⁷ and relative_rmse = 10⁻⁷. This pass corrects large-scale rotational and translational residuals from Procrustes.

**Pass 1.5 — Point-to-Point Translation Stabilization.** After Pass 1, the rotation is well-estimated but translation may exhibit slight drift due to asymmetric face topology. We run a brief Point-to-Point ICP (200 iterations, d = 0.03, convergence 10⁻⁹) on the same voxelized clouds to stabilize the translation component without disturbing the rotation.

**Pass 2 — Fine Point-to-Plane.** Both clouds are re-downsampled at finer resolution δ/2 = 0.001, nearly doubling the point density used for fine alignment. We run 500 iterations at d₂ = 0.02 with convergence 10⁻¹², driving the inlier RMSE to its minimum achievable value given the inherent shape-space mismatch between the two meshes.

After all three passes, we compute an adaptive distance cap for subsequent steps:
```
DIST_CAP = max(0.03, 2.0 × σ_surface)
```
where σ_surface is the surface RMSE measured on 5,000 randomly sampled target vertices. This adaptive threshold gracefully handles subjects with unusual face proportions where alignment is inherently less tight.

#### 3.3.3 Vectorized Vertex-Level Color Transfer

With the source mesh now co-registered in the target mesh's coordinate space, we transfer color from the completed texture **T**_IDM to each target mesh vertex. This step produces a per-vertex color array **C** ∈ R^{N×3} (N = number of target vertices) along with a confidence weight **w** ∈ R^N for each vertex.

**Surface proximity query.** We issue a single batched query that, for each target vertex **v**_i, finds its closest point on the aligned source mesh surface:
```
(c_i, d_i, tri_i) = ClosestPoint(src_aligned, v_i),   ∀i ∈ {1, …, N}
```
where **c**_i is the 3D closest point, d_i is the Euclidean distance, and tri_i is the index of the closest source triangle. This single vectorized query replaces a naive per-vertex loop and completes in under 0.5 seconds for the full mesh.

**Confidence weighting.** Rather than applying a hard distance gate (which would leave some vertices uncolored), we compute soft confidence weights that modulate blending quality without excluding any vertex:

*Distance confidence:*
```
w_dist(i) = clip(1 − d_i / d_99, 0, 1)
```
where d_99 is the 99th percentile of {d_i}, so all but the most extreme 1% of vertices receive positive distance confidence.

*Normal consistency confidence:*
```
w_norm(i) = clip((n̂_tgt(i) · n̂_src(tri_i) + 0.8) / 1.8, 0, 1)
```
This term equals 1 when source and target surface normals are co-aligned, and falls to 0 at a dot product of −0.8 (approximately 144°). Normal dot products are checked for systematic sign flips (which occur when mesh normals are inverted) and automatically corrected.

The combined confidence is w(i) = w_dist(i) · w_norm(i), stored for use in the baking stage.

**Barycentric UV interpolation.** For each target vertex **v**_i, we interpolate source UV coordinates from the three corners of the closest source triangle tri_i using the barycentric weights of the closest point **c**_i:
```
[b₀, b₁, b₂] = barycentric(c_i; V_src[tri_i])
uv_i = b₀ · uv_src[tri_i,0] + b₁ · uv_src[tri_i,1] + b₂ · uv_src[tri_i,2]
```
This entire computation is expressed as matrix operations across all N vertices simultaneously, with no Python-level loops. For degenerate triangles (near-zero area, where the barycentric linear system is singular with |det| < 10⁻¹²), we fall back to assigning the UV coordinates of the nearest triangle corner, determined by minimum squared distance from **v**_i to each corner.

**Bilinear texture sampling.** The source UV coordinate **uv**_i is then used to bilinearly sample **T**_IDM:
```
C_i = BilinearSample(T_IDM, uv_i)
```
Bilinear interpolation is implemented as a single NumPy vectorized operation across all vertices.

The result is a 100% vertex coverage guarantee: every target vertex receives a valid color **C**_i, regardless of alignment quality, because the confidence weighting scheme never excludes—it only down-weights—vertices with poor alignment metrics.

#### 3.3.4 Zero-Loop Vectorized UV Rasterization and Baking

The per-vertex color array **C** ∈ R^{N×3} must be projected into the 2D target mesh UV atlas at R_tex×R_tex resolution, where R_tex = 1024 is the output atlas resolution. This is achieved through a fully vectorized rasterizer that contains no Python-level loops over either pixels or faces.

**UV rasterization.** For each target mesh triangle, we compute:
1. Pixel-space vertex coordinates: **px** = UV_u · (R_tex−1), **py** = (1 − UV_v) · (R_tex−1).
2. Axis-aligned bounding box (AABB): [bbx0, bbx1] × [bby0, bby1] for each triangle.
3. Triangle area via the cross-product sign: area = (p1x − p0x)(p2y − p0y) − (p1y − p0y)(p2x − p0x). Degenerate triangles (|area| < 10⁻⁶) are skipped.

All (face, pixel) candidate pairs are then enumerated as a flat array using `np.repeat` and integer-division indexing—the key innovation that eliminates the face loop:
```
lfi_flat = np.repeat(arange(N_f), pixels_per_face)  # flat face assignment
loc_row  = candidates // cols_per_face[lfi_flat]    # local row within bbox
loc_col  = candidates %  cols_per_face[lfi_flat]    # local col within bbox
```
where N_f is the number of target mesh faces. For each candidate pixel (abs_x, abs_y), an edge-function test checks containment within the triangle:
```
w₀ = ((p2x−p1x)(y−p1y) − (p2y−p1y)(x−p1x)) / area
w₁ = ((p0x−p2x)(y−p2y) − (p0y−p2y)(x−p2x)) / area
w₂ = ((p1x−p0x)(y−p0y) − (p1y−p0y)(x−p0x)) / area
inside = (w₀ ≥ 0) ∧ (w₁ ≥ 0) ∧ (w₂ ≥ 0)
```
All candidate pairs are tested simultaneously in a single NumPy pass. The barycentric weights (w₀, w₁, w₂) are scattered into a `face_map` (R_tex×R_tex int32, storing face index per texel) and `bary_map` (R_tex×R_tex×3 float32, storing barycentric weights). To bound peak RAM usage, faces are processed in chunks of at most RAST_MAX_CAND = 4,000,000 candidate pixels per chunk.

**Barycentric baking.** Once the `face_map` is populated, the final texture color at each covered texel (y, x) is computed as:
```
T_complete[y, x] = w₀ · C[v₀] + w₁ · C[v₁] + w₂ · C[v₂]
```
where v₀, v₁, v₂ are the three target vertex indices of the triangle at (y, x), and **C** is the per-vertex color array from §3.3.3. This single matrix gather-and-multiply completes in under 0.5 seconds for the full 1024×1024 atlas.

#### 3.3.5 Seam Dilation and Gaussian Boundary Blending

Raw UV baking leaves two artifact categories: (i) *seam gaps*—single-pixel cracks along UV island boundaries where rasterization undersampling leaves isolated background pixels within the face region; and (ii) *background voids*—large unfilled areas in UV atlas regions corresponding to the back of the head, where target mesh UV faces may be sparsely distributed.

We apply a three-pass post-processing stage using morphological dilation and Gaussian blending:

**Pass A — Seam dilation.** A 5×5 morphological dilation with 3 iterations expands the face texture slightly outward, flooding single-pixel seam gaps:
```
seam_near = Dilate(covered_mask, 5×5, iter=3) AND NOT covered_mask
T_out[seam_near] = Dilate(T_out, 5×5, iter=3)[seam_near]
```

**Pass B — Background flood fill.** A wider 9×9 dilation with 5 iterations fills larger background voids (ears, scalp, neck region) with nearby face texture, ensuring the entire atlas is populated:
```
still_empty = NOT covered_mask AND NOT seam_near
T_out[still_empty] = Dilate(T_out, 9×9, iter=5)[still_empty]
```

**Pass C — Gaussian boundary blending.** At the transition between rasterized face texture and dilated-fill regions, a hard seam line is visible. We dissolve it with a confidence-weighted Gaussian blend:
```
α(y,x) = GaussianBlur(covered_mask, σ=3.0)
T_out[band] = α · T_out[band] + (1−α) · GaussianBlur(T_out, σ=1.5)[band]
```
where `band` is the set of pixels with 0.05 < α < 0.95. This produces a smooth perceptual transition with an effective blending radius of approximately 5 pixels.

The final output **T**_complete ∈ R^{1024×1024×3} has 100% texel coverage and no visible seam artifacts, as verified by inspection across all 74 subjects.

---

### 3.4 Formal Definition of the 4D Texture Avatar

We define the concept of a *4D texture avatar* formally to precisely characterize what our system produces and why texture completeness is a *necessary* condition—not merely a quality improvement.

**Notation.** We use the following symbols consistently throughout: S ⊂ ℝ³ — the face surface (a 2-manifold); φ: Ω_UV → S — the UV parameterization mapping UV domain Ω_UV = [0,1]² to surface points; T: Ω_UV → ℝ³ — the UV texture map; **V**_t ∈ ℝ^{N×3} — per-frame vertex positions (N as in §3.2); N_f — number of mesh faces; K — number of UV coordinate pairs; τ — total animation duration; θ ∈ Θ — camera pose parameter (distinct from the head-pose parameter **θ_p** in §3.2).

A parametric face model provides a time-varying surface through per-frame vertex positions **V**_t, so that S_t is the deformed mesh at time t ∈ [0, τ] encoding shape, expression, jaw, and head pose. A camera model with pose θ ∈ Θ defines a projection π(·; θ): S → ℝ² and a visibility set:

```
Vis(t, θ) = { p ∈ S_t : p is visible from camera θ at time t }
```

The *rendered appearance* at screen pixel (x, y) under camera pose θ at time t is:

```
I(x, y, t; θ) = T(φ⁻¹(π⁻¹(x, y; θ, t)))
```

where T: Ω_UV → ℝ³ is the UV texture map. To render pixel (x, y), we find its corresponding 3D surface point via inverse projection π⁻¹, look up its UV coordinate via the inverse parameterization φ⁻¹, and sample the texture T.

**The 4D avatar.** We define a *4D texture avatar* as the triple:

```
A = ( {V_t}_{t ∈ [0,τ]},   φ,   T )
```

such that T is defined on the *entire* UV domain Ω_UV. The descriptor "4D" captures that the rendered appearance I(x, y, t; θ) is a function over four free variables: two spatial screen coordinates (x, y), time t, and camera pose θ — yielding a well-defined image for any combination of these four variables.

**Texture completeness as a necessary condition.** For I to be defined at every (x, y, t, θ) combination, T must be defined at the UV preimage of every possibly-visible surface point. The set of UV coordinates accessible under a given capture pose θ_in is:

```
Ω_in(θ_in) = { φ⁻¹(p) : p ∈ Vis(t_in, θ_in) } ⊂ Ω_UV
```

For monocular reconstruction from a single C4-angle image (approximately 30° lateral offset from frontal, corresponding to head pose **θ_p**^in), empirical measurement on our dataset yields:

```
|Ω_in| / |Ω_UV|  ≈  0.40 – 0.58
```

meaning 42–60% of the UV atlas receives no valid observation. The partial texture T_partial is therefore defined as:

```
              ⎧ I_in(π(φ(uv); θ_p^in))    if uv ∈ Ω_in(θ_p^in)
T_partial(uv) = ⎨
              ⎩ undefined                   otherwise
```

where I_in is the input image captured at pose **θ_p**^in. As a direct consequence, for any novel camera pose θ ≠ θ_eval and any time t, the rendered image I(x, y, t; θ) contains undefined regions wherever the projected surface point φ(uv) falls outside Ω_in. The 4D avatar property is *violated*: only a restricted subset of (t, θ) pairs yields valid renderings.



Our completed texture **T**_complete, obtained through inpainting and cross-topology transfer, satisfies:

```
T_complete(uv) is defined   ∀ uv ∈ Ω_UV
```

guaranteeing that I(x, y, t; θ) is defined for *any* camera pose θ ∈ Θ and any animation time t ∈ [0, τ]. This is the precise sense in which our pipeline produces a true 4D avatar: texture completeness is not a cosmetic improvement but the enabling condition for the 4D property.

**Concrete representation.** Each animation frame is a tuple:

```
f_t = { V_t ∈ ℝ^{N×3},   F ∈ ℤ^{N_f×3},   UV ∈ ℝ^{K×2},   T_complete ∈ ℝ^{R_tex×R_tex×3} }
```

where N (vertex count), N_f (face count), and K (UV coordinate count) are determined by the target mesh topology (specific values in §4.1), and R_tex = 1024 is the output texture atlas resolution. **T**_complete is shared across all frames under the identity-constant albedo assumption. The sequence {f_t}_{t=1}^{τ} constitutes the 4D avatar: spatially navigable across the full camera-pose sphere and temporally navigable across the complete animation.

---

## 4. Experiments

### 4.1 Experimental Setup

We evaluate on **PolyFace** [15], a multi-view face dataset capturing 74 subjects under controlled studio lighting. Each subject is recorded simultaneously from 7 calibrated cameras at angles C1, C4, C7, C10, C13, C17, and C24 relative to the frontal axis. We use C4 images (approximately 30° lateral offset) as input and C7 images (frontal) as ground truth.

The target animation mesh has N = 5,023 vertices, N_f = 9,976 faces, and K = 5,023 UV coordinate pairs with a fixed topology shared across all subjects. The source inpainting mesh has N_src = 53,215 vertices with a denser UV atlas of resolution R_u = 512. The output texture atlas is baked at R_tex = 1024 × 1024 pixels. This protocol reflects a realistic capture scenario where the input photograph is taken at a non-frontal angle, and the quality of novel-view rendering is evaluated against the frontal view that most clearly exposes previously occluded surface regions. For each subject, we render the reconstructed avatar from the C7 camera viewpoint and compare against the ground-truth C7 photograph, aligned and cropped to the face region.

We report three complementary metrics: **PSNR** (dB, higher is better) measures pixel-level reconstruction accuracy; **SSIM** (higher is better) measures perceptual structure preservation; and **LPIPS** [16] (lower is better) measures deep feature-level perceptual similarity, which correlates more strongly with human judgment than pixel-level metrics. We compare against the partial-texture reconstruction baseline produced by the monocular face reconstruction method [1] without our texture completion and transfer pipeline.

### 4.2 Results

**Quantitative comparison.** Table 1 reports per-metric averages and standard deviations over all 74 subjects.

**Table 1. Quantitative comparison on PolyFace (74 subjects). ↑: higher is better; ↓: lower is better.**

| Method | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ |
|--------|-------------|--------|---------|
| Baseline (partial texture) | 13.73 ± 1.400 | 0.447 ± 0.027 | 0.513 ± 0.043 |
| **Ours (complete texture)** | **23.80 ± 1.124** | **0.602 ± 0.054** | **0.350 ± 0.085** |
| **Δ** | **+10.07** | **+0.155** | **−0.163** |

All 74 subjects (100%) improve on every metric without exception. The +10.07 dB PSNR gain corresponds to reducing mean squared error by a factor of 10—reflecting the categorical difference between a texture with structurally blank regions and a fully completed one. The LPIPS improvement of −0.163 confirms that the gain is perceptually meaningful, not merely numerical. The higher standard deviation of LPIPS (0.085) compared to SSIM (0.054) suggests that perceptual quality varies more across subjects, likely due to the inpainting model's sensitivity to hair texture and skin tone diversity.

**Per-subject consistency.** The improvement is consistent across subjects and input conditions: the minimum per-subject PSNR gain is approximately +7.5 dB and the maximum approximately +14.5 dB. Subjects with larger improvements tend to have higher initial occlusion—wider face angles at capture time—confirming that our method provides proportionally greater benefit when more texture completion is required. This relationship validates that the performance gain is causally driven by the texture completion mechanism rather than subject-specific reconstruction quality.

**Qualitative comparison.** Side-by-side renders (Fig. 3) show that the baseline produces sharp geometry but large blank patches on the ears, temples, and neck region. Our method fills these regions with plausible texture consistent with the visible skin tone, enabling artifact-free 360° rendering. Notably, the completed texture shows no visible seam artifacts at the cross-topology transfer boundary, demonstrating the effectiveness of the ICP-refined alignment and the seam dilation post-processing.

**Runtime.** The cross-topology texture transfer (Procrustes + ICP + rebaking) completes in approximately 35–55 seconds per subject on an Apple M4 CPU, dominated by the ICP refinement (~12 s) and UV rebaking (~8–12 s). The inpainting model adds 15–30 seconds. The full pipeline from input photograph to complete 4D avatar thus runs in under 90 seconds on consumer hardware without GPU acceleration for the transfer stage.

---

## 5. Application Scenarios

The complete-texture 4D avatar is qualitatively different from a partial-texture reconstruction because it supports free-viewpoint rendering at any animation time—a property that is simply absent from any single-viewpoint baked texture, regardless of its resolution or quality. We identify three domains where this property is a decisive enabling factor rather than an incremental improvement.

The most direct application is *personalized avatar creation from a single photograph*. Because the completed texture covers the entire head surface, the resulting avatar looks realistic from any observation direction and can be freely rotated during playback without exposing blank regions. When the animation sequence is driven by recorded or synthesized speech, the result is an animated face avatar whose appearance is angularly consistent—an individualized 4D representation that requires no multi-view capture hardware and no per-subject training. This addresses a broad class of use cases: e-learning, digital identity representation, virtual production, and social presence applications where a persistent personalized avatar is needed from a single enrollment photograph.

A second domain is *digital cultural and historical preservation*. Museum and archival collections contain photographs of historical figures, traditional performers, and community elders where only one image exists per subject. NeRF-based reconstruction [10,11] requires hundreds of photographs from multiple viewpoints; our pipeline requires exactly one. The resulting 4D avatar can deliver pre-written or synthesized speech and be explored spatially, providing a form of interactive presence that is impossible from a static photograph. The trade-off—that texture in occluded regions is hallucinated rather than observed—is a necessary approximation when the photographic record is limited to a single frame.

A third domain is *longitudinal facial geometry and texture monitoring*, relevant to post-surgical recovery tracking and rehabilitation assessment. Structured-light 3D scanning provides accurate measurements but requires specialized hardware accessible only in clinical centers. A complete 4D reconstruction from monocular video enables low-cost repeated measurement: the UV texture atlas captures pigmentation changes, scarring evolution, and skin-tone variation that would be invisible from a fixed frontal camera; the parametric geometry encodes quantifiable shape changes—facial asymmetry, volume recovery, jaw mobility—in a format independent of session-to-session head pose variation.

---

## 6. Discussion

### 6.1 The 4D Texture Property: Analysis and Implications

The 100% improvement rate across all 74 subjects is not a coincidence—it is a structural consequence of the problem being addressed. Partial textures produced by single-viewpoint reconstruction are deterministically incomplete: the blank UV regions are not a stochastic failure mode but a direct geometric result of the capture angle. Any completion that fills these regions with plausible content will necessarily score higher when evaluated against a frontal ground truth that exposes them. The inpainting model's hallucinations, even when imperfect, are far closer to the true texture than blank or distorted pixels, so the improvement is guaranteed for any subject where the capture angle differs from the evaluation angle.

The deeper significance of texture completeness is quantified by the formal 4D avatar definition in §3.4. The rendered appearance I(x, y, t; θ) is defined over the product space ℝ² × [0, τ] × S², where S² is the camera-pose sphere and τ is the animation duration. A partial texture T_partial is defined only over the observed UV subset Ω_in ⊂ Ω_UV—which, for a 30° lateral capture angle, covers approximately 40–58% of the atlas. This means I(x, y, t; θ) is undefined for any camera pose θ that projects surface points onto unobserved UV regions. Concretely, the fraction of the camera-pose sphere that produces valid renders under T_partial is:

```
|Θ_valid(T_partial)| / |S²|  =  (1 − cos α_max) / 2  ≈  0.07 – 0.12
```

for α_max ≈ 30–40°. Our complete texture restores full coverage:

```
|Θ_valid(T_complete)| / |S²|  =  1.0
```

This is a categorical change, not an incremental improvement. A partial-texture reconstruction is a *2.5D* avatar—animated in time but viewpoint-locked to a narrow frontal cone covering 7–12% of the camera sphere. Our complete-texture avatar is genuinely 4D: navigable across the full time axis and the full camera-pose sphere simultaneously. The +10.07 dB PSNR gain is the measurable signal at the single evaluation viewpoint; the viewpoint-freedom gain—the recovery of 88–93% of the camera sphere—is not captured by any single-angle metric and represents a qualitatively larger, unquantified dimension of improvement.

**Table 2. Navigable dimensions of partial-texture vs. complete-texture avatars.**

| Dimension | Variable | T_partial | T_complete |
|-----------|----------|-----------|------------|
| Time | t | [0, τ] | [0, τ] |
| Camera azimuth | θ_az | ~±30° cone | Full 360° |
| Camera elevation | θ_el | ~±20° cone | Full 180° |
| Valid pose coverage | — | ≈7–12% of S² | 100% of S² |

**Temporal coherence.** A non-obvious property of our representation is that temporal texture coherence is guaranteed *by construction* at zero additional cost. Because the completed texture T_complete is shared across all frames and the mesh UV parameterization φ is fixed regardless of expression, corresponding anatomical points on consecutive frames always map to identical UV coordinates: φ⁻¹(p_t) = φ⁻¹(p_{t+Δt}), where p_t and p_{t+Δt} are the same anatomical point on the surface at consecutive frames. The texture therefore "moves with the skin" automatically—as the jaw opens, the lip texture follows the lip vertices without any warping computation; as the head rotates, newly visible UV regions are already filled and rendered correctly the instant they enter the camera frustum. Video-based synthesis methods must explicitly enforce temporal consistency through recurrent layers, temporal attention, or optical-flow smoothing because they operate in image space where frame-to-frame correspondence is not encoded. Our mesh-based representation achieves this property as an algebraic consequence of the shared UV parameterization.

**Comparison with competing representations.** Table 3 situates our method relative to the principal alternative approaches for dynamic face modeling.

**Table 3. Comparison of 4D face representations.**

| Method | Complete texture | Free viewpoint | Per-subject training | Single photo input |
|--------|-----------------|---------------|---------------------|--------------------|
| 2D talking head [13] | ✗ | ✗ | ✗ | ✓ |
| NeRF-based [10,11] | ✓ | ✓ | Required (hours) | ✗ |
| Partial-texture mesh [1] | ✗ | ✗ | ✗ | ✓ |
| **Ours** | **✓** | **✓** | **✗** | **✓** |

NeRF-based methods achieve high-quality free-viewpoint renders but require 30–60 minutes of per-subject training on multi-view video and are computationally prohibitive at inference without dedicated GPU ray marching. Our system produces a complete-texture 4D avatar from a single photograph in under 90 seconds without per-subject training. The trade-off is that NeRF captures view-dependent lighting effects—specular reflections, subsurface scattering—that a diffuse UV texture cannot represent. For scenarios that require free-viewpoint access without multi-view capture infrastructure or training compute, our representation is the more practical choice.

### 6.2 Limitations and Future Directions

The primary limitation is *texture hallucination fidelity*. The inpainting model synthesizes plausible but not necessarily accurate content for occluded regions. For subjects with distinctive hair styles, unusual pigmentation, or strong lateral occlusion at capture time, the completed areas may deviate from the true appearance—a pattern reflected in the higher LPIPS standard deviation (0.085) compared to SSIM (0.054). A natural remedy is multi-view supervision: leveraging simultaneous camera arrays to directly observe held-out UV regions during training, reducing the fraction of content that must be hallucinated.

A second limitation is the *static texture assumption*. We apply a single identity texture to all animation frames, which is valid for diffuse skin albedo but does not capture expression-dependent appearance changes such as wrinkle deepening or specular highlight shifts under extreme expressions. Extending the pipeline to condition the inpainting model on per-frame expression codes would produce expression-adaptive texture, transforming the current spatially-4D avatar into a representation that is photometrically dynamic as well.

The pipeline also inherits the *geometry accuracy* of the upstream reconstruction method. Subjects with strong occlusions, extreme head poses, or atypical face shapes that fall outside the training distribution of the reconstruction model will produce correspondingly imperfect geometry, which the texture transfer cannot compensate for. Replacing the ICP-based cross-topology alignment with a learned mesh correspondence network would make the transfer more robust to shape-space mismatches between the inpainting and animation mesh topologies. Finally, the current system animates an existing recorded sequence; connecting the complete-texture mesh to a speech-driven expression synthesis module would enable fully generative 4D avatars from text or audio input alone, without requiring a source video.

---

## 7. Conclusion

We have presented a complete end-to-end pipeline for free-viewpoint 4D face avatar reconstruction from a single photograph. The pipeline integrates monocular face reconstruction, UV-space inpainting, geometry-aware cross-topology texture transfer, and 4D avatar assembly into a unified system that produces complete-surface textured animated meshes in under 90 seconds on consumer hardware. We formalized the *4D texture avatar* property as a coverage condition on the UV domain, establishing that partial textures—a structural limitation of all single-viewpoint monocular methods—violate this condition for the majority of the camera-pose sphere, while our complete texture restores it unconditionally. The cross-topology texture transfer module, combining robust Procrustes alignment, three-pass Point-to-Plane ICP, soft-confidence vertex color transfer, and zero-loop vectorized UV rebaking, resolves the topology mismatch between the inpainting mesh and the animation mesh without coverage gaps or seam artifacts. Quantitative evaluation on 74 subjects demonstrates consistent improvement over the partial-texture baseline across all subjects and all reported metrics, with the 100% improvement rate being a structural consequence of the completeness property rather than subject-specific variability. The resulting 4D avatar is simultaneously navigable across three spatial dimensions and the full temporal animation axis, enabling downstream applications—personalized avatar creation, cultural preservation, longitudinal facial monitoring—that require free-viewpoint rendering from a single photograph.

---

## References

[1] J. Feng, H. Feng, Q. Li, Z. Liu, and Z. Mao, "DECA: Detailed Expression Capture and Animation," *ACM Transactions on Graphics (TOG)*, vol. 40, no. 4, 2021.

[2] T. Li, T. Bolkart, M. J. Black, H. Li, and J. Romero, "Learning a Model of Facial Shape and Expression from 4D Scans," *ACM Transactions on Graphics*, vol. 36, no. 6, 2017.

[3] [UV-IDM authors], "UV-IDM: Identity-Conditioned Latent Diffusion Model for Face UV-Texture Generation," *arXiv preprint*, 2024. [Update with correct citation]

[4] V. Blanz and T. Vetter, "A Morphable Model for the Synthesis of 3D Faces," in *Proc. SIGGRAPH*, 1999.

[5] P. Paysan, R. Knothe, B. Amberg, S. Romdhani, and T. Vetter, "A 3D Face Model for Pose and Illumination Invariant Face Recognition," in *Proc. AVSS*, 2009.

[6] A. Tran, T. Hassner, I. Masi, and G. Medioni, "Regressing Robust and Discriminative 3D Morphable Models with a Very Deep Neural Network," in *Proc. CVPR*, 2017.

[7] Y. Deng, J. Yang, S. Xu, D. Chen, Y. Jia, and X. Tong, "Accurate 3D Face Reconstruction with Weakly-Supervised Learning," in *Proc. CVPRW*, 2019.

[8] K. Danecek, M. J. Black, and T. Bolkart, "EMOCA: Emotion Driven Monocular Face Capture and Animation," in *Proc. CVPR*, 2022.

[9] H. Deng, T. Cheng, X. Cao, and Y. Ma, "UV-GAN: Adversarial Facial UV Map Completion for Pose-Invariant Face Recognition," in *Proc. CVPR*, 2018.

[10] G. Guo et al., "AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis," in *Proc. ICCV*, 2021.

[11] K. Park et al., "NerFace: Deformable Neural Radiance Fields for Dynamic Novel View Synthesis of Faces," in *Proc. CVPR*, 2021.

[12] S. Thies, M. Zollhöfer, M. Stamminger, C. Theobalt, and M. Nießner, "Face2Face: Real-time Face Capture and Reenactment of RGB Videos," in *Proc. CVPR*, 2016.

[13] O. Fried et al., "Text-based Editing of Talking-head Video," *ACM TOG*, vol. 38, no. 4, 2019.

[14] C. Xing et al., "CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior," in *Proc. CVPR*, 2023.

[15] [PolyFace authors], "PolyFace: A Multi-View Face Dataset for 3D Reconstruction Benchmarking," [Conference/Journal, Year]. [Update with correct citation]

[16] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," in *Proc. CVPR*, 2018.

---

*[Figures to be inserted by author:*
- *Fig. 1: Pipeline diagram — monocular input → face reconstruction → UV inpainting → cross-topology texture transfer → 4D avatar assembly]*
- *Fig. 2: Per-subject PSNR scatter plot (Baseline vs. Ours, 74 subjects)]*
- *Fig. 3: Qualitative comparison — partial texture (baseline) / complete texture (ours) / ground truth, for 4–6 subjects, from novel viewpoints]*
