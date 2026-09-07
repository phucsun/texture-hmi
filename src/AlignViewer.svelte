<script lang="ts">
  import {
    Engine, Scene, ArcRotateCamera, HemisphericLight, DirectionalLight,
    Vector3, Mesh, Color3, Color4, StandardMaterial, VertexData, VertexBuffer,
  } from "@babylonjs/core";
  import { onMount, onDestroy } from "svelte";

  let canvas: HTMLCanvasElement;
  let engine: Engine;
  let scene: Scene;

  let isLoading = false;
  let status = "";
  let errorMsg = "";

  let srcMesh: Mesh | null = null;
  let tgtMesh: Mesh | null = null;

  // pre-parsed vertex arrays for each step (OBJ space)
  let stepVerts: Float32Array[] = [];
  let srcIdxOBJ: Int32Array = new Int32Array(0);

  let steps: { name: string }[] = [];
  let currentStep = 0;
  $: totalSteps = steps.length;

  const C_SRC = new Color3(0.35, 0.60, 1.00);
  const C_TGT = new Color3(1.00, 0.56, 0.27);

  // ── OBJ parser ─────────────────────────────────────────────────────────────
  function parseOBJ(text: string): { pos: Float32Array; idx: Int32Array } {
    const verts: number[] = [];
    const faces: number[] = [];
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      if (line.startsWith("v ")) {
        const t = line.split(/\s+/);
        verts.push(+t[1], +t[2], +t[3]);
      } else if (line.startsWith("f ")) {
        const t = line.split(/\s+/).slice(1);
        const ids = t.map(s => parseInt(s.split("/")[0]) - 1);
        for (let i = 1; i < ids.length - 1; i++)
          faces.push(ids[0], ids[i], ids[i + 1]);
      }
    }
    return { pos: new Float32Array(verts), idx: new Int32Array(faces) };
  }

  // ── OBJ (RHS) → BabylonJS (LHS): flip Z, reverse winding ─────────────────
  function objToBabylon(pos: Float32Array): Float32Array {
    const out = new Float32Array(pos.length);
    for (let i = 0; i < pos.length; i += 3) {
      out[i] = pos[i]; out[i + 1] = pos[i + 1]; out[i + 2] = -pos[i + 2];
    }
    return out;
  }

  function reverseWinding(idx: Int32Array): Int32Array {
    const out = new Int32Array(idx.length);
    for (let i = 0; i < idx.length; i += 3) {
      out[i] = idx[i]; out[i + 1] = idx[i + 2]; out[i + 2] = idx[i + 1];
    }
    return out;
  }

  function makeMesh(name: string, posOBJ: Float32Array, idxOBJ: Int32Array,
                    color: Color3, alpha = 0.75, updatable = false): Mesh {
    const pos = objToBabylon(posOBJ);
    const idx = reverseWinding(idxOBJ);
    const normals: number[] = [];
    VertexData.ComputeNormals(pos as unknown as number[], idx as unknown as number[], normals);

    const mesh = new Mesh(name, scene);
    const vd = new VertexData();
    vd.positions = pos;
    vd.indices   = idx;
    vd.normals   = new Float32Array(normals);
    vd.applyToMesh(mesh, updatable);

    const mat = new StandardMaterial(name + "_mat", scene);
    mat.diffuseColor  = color;
    mat.emissiveColor = color.scale(0.20);
    mat.alpha = alpha;
    mat.backFaceCulling = false;
    mesh.material = mat;
    return mesh;
  }

  function updateSrcMesh(posOBJ: Float32Array) {
    if (!srcMesh) return;
    const pos = objToBabylon(posOBJ);
    const idx = reverseWinding(srcIdxOBJ);
    const normals: number[] = [];
    VertexData.ComputeNormals(pos as unknown as number[], idx as unknown as number[], normals);
    srcMesh.updateVerticesData(VertexBuffer.PositionKind, pos, false, false);
    srcMesh.updateVerticesData(VertexBuffer.NormalKind, new Float32Array(normals), false, false);
  }

  // ── load folder ────────────────────────────────────────────────────────────
  // Expected layout (output of align_only.py):
  //   step_0_raw.obj, step_1_axis_scale.obj, step_2_procrustes.obj,
  //   step_3_icp1.obj, step_4_icp15.obj, step_5_icp2.obj
  //   deca.obj  (target, stays static)
  async function loadFolder() {
    try {
      const dirHandle = await (window as any).showDirectoryPicker();
      isLoading = true; errorMsg = ""; status = "Scanning folder…";

      const fileMap: Record<string, File> = {};
      for await (const entry of dirHandle.values()) {
        if (entry.kind !== "file") continue;
        const f = await entry.getFile();
        fileMap[f.name.toLowerCase()] = f;
      }

      // Collect step files
      const stepFiles: Array<{ name: string; file: File }> = [];
      for (const [fname, file] of Object.entries(fileMap)) {
        const m = fname.match(/^step_(\d+)_(.+)\.obj$/);
        if (m) stepFiles.push({ name: `Step ${m[1]} — ${m[2].replace(/_/g, " ")}`, file });
      }
      if (stepFiles.length === 0)
        throw new Error("Không tìm thấy file step_N_*.obj. Chạy align_only.py trước.");
      stepFiles.sort((a, b) => a.file.name.localeCompare(b.file.name));

      const tgtFile = fileMap["deca.obj"];
      if (!tgtFile) throw new Error("Không tìm thấy deca.obj trong folder.");

      // Parse all step OBJs (they all share the same topology)
      status = "Parsing step files…";
      const parsed = await Promise.all(stepFiles.map(s => s.file.text().then(parseOBJ)));
      const tgtParsed = await tgtFile.text().then(parseOBJ);

      // All step meshes share the topology of the first step
      srcIdxOBJ = parsed[0].idx;
      stepVerts = parsed.map(p => p.pos);
      steps = stepFiles.map(s => ({ name: s.name }));

      // Build meshes
      status = "Building scene…";
      srcMesh?.dispose(); tgtMesh?.dispose();
      srcMesh = makeMesh("src", stepVerts[0], srcIdxOBJ, C_SRC, 0.72, true);
      tgtMesh = makeMesh("tgt", tgtParsed.pos, tgtParsed.idx, C_TGT, 0.60, false);
      currentStep = 0;

      // Focus camera on target mesh
      tgtMesh.computeWorldMatrix(true);
      const bb  = tgtMesh.getBoundingInfo().boundingBox;
      const sz  = bb.maximumWorld.subtract(bb.minimumWorld);
      const cam = scene.activeCamera as ArcRotateCamera;
      cam.setTarget(bb.centerWorld);
      cam.radius = Math.max(sz.x, sz.y, sz.z) * 2.8;

    } catch (e: any) {
      errorMsg = e.message ?? String(e);
    } finally {
      isLoading = false; status = "";
      engine?.resize();
    }
  }

  // ── navigation ──────────────────────────────────────────────────────────────
  function goTo(i: number) {
    if (!srcMesh || i < 0 || i >= totalSteps) return;
    currentStep = i;
    updateSrcMesh(stepVerts[i]);
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "ArrowLeft")  goTo(currentStep - 1);
    if (e.key === "ArrowRight") goTo(currentStep + 1);
  }

  // ── scene ───────────────────────────────────────────────────────────────────
  onMount(() => {
    engine = new Engine(canvas, true);
    scene  = new Scene(engine);
    scene.clearColor = new Color4(0.08, 0.08, 0.10, 1);

    const cam = new ArcRotateCamera("cam", -Math.PI / 2, Math.PI / 2.5, 1, Vector3.Zero(), scene);
    cam.attachControl(canvas, true);
    cam.minZ = 0.001;
    cam.wheelPrecision = 80;

    new HemisphericLight("h", new Vector3(0, 1, 0),  scene).intensity = 0.7;
    new DirectionalLight("d", new Vector3(-1,-2,-1), scene).intensity = 0.4;

    engine.runRenderLoop(() => scene.render());
    window.addEventListener("resize",  () => engine.resize());
    window.addEventListener("keydown", onKey);
  });

  onDestroy(() => {
    engine?.dispose();
    window.removeEventListener("keydown", onKey);
  });
</script>

<main class="w-full h-screen bg-[#0f0f12] relative overflow-hidden">

  {#if isLoading}
    <div class="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/80 text-white gap-3">
      <div class="w-8 h-8 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
      <div class="text-blue-300 font-mono text-sm">{status}</div>
    </div>
  {/if}

  {#if errorMsg}
    <div class="absolute top-4 left-1/2 -translate-x-1/2 z-50 bg-red-900/90 text-red-200 px-4 py-2 rounded-lg text-sm max-w-md text-center">
      ⚠ {errorMsg}
    </div>
  {/if}

  <button
    class="absolute top-3 right-3 z-20 bg-white/90 hover:bg-blue-50 px-4 py-2 rounded-lg font-semibold shadow text-sm"
    on:click={loadFolder} disabled={isLoading}
  >📂 Load output/ folder</button>

  {#if steps.length}
    <div class="absolute top-3 left-3 z-20 flex gap-4 bg-black/50 backdrop-blur-sm rounded-lg px-3 py-1.5 text-sm text-white">
      <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full" style="background:#5999FF"></span>Source (BFM)</span>
      <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full" style="background:#FF8F45"></span>Target (DECA)</span>
    </div>
  {/if}

  <canvas bind:this={canvas} class="w-full h-full touch-none outline-none" />

  {#if !steps.length && !isLoading}
    <div class="absolute inset-0 flex items-center justify-center pointer-events-none">
      <div class="text-center text-white/40">
        <div class="text-5xl mb-3">🦴</div>
        <div class="font-medium">Alignment Viewer</div>
        <div class="text-sm mt-1 opacity-70">Chạy <code>python align_only.py</code> rồi load folder output/</div>
      </div>
    </div>
  {/if}

  {#if steps.length}
    <div class="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 w-[38rem] max-w-[94vw]">
      <div class="bg-black/70 backdrop-blur-md rounded-2xl border border-white/10 px-5 py-4 flex flex-col gap-3">

        <div class="flex items-center justify-center gap-2">
          {#each steps as _, i}
            <button
              class="rounded-full transition-all {i === currentStep ? 'w-3.5 h-3.5 bg-blue-400' : 'w-2.5 h-2.5 bg-white/30 hover:bg-white/60'}"
              on:click={() => goTo(i)}
            ></button>
          {/each}
        </div>

        <div class="text-center">
          <div class="text-white font-semibold">{steps[currentStep].name}</div>
        </div>

        <div class="flex items-center justify-between">
          <button
            class="px-4 py-2 rounded-xl text-sm font-medium {currentStep === 0 ? 'text-white/20 cursor-not-allowed' : 'text-white bg-white/10 hover:bg-white/20'}"
            on:click={() => goTo(currentStep - 1)} disabled={currentStep === 0}
          >← Prev</button>

          <span class="text-white/40 text-xs font-mono">{currentStep + 1}/{totalSteps} · ← → keys</span>

          <button
            class="px-4 py-2 rounded-xl text-sm font-medium {currentStep === totalSteps-1 ? 'text-white/20 cursor-not-allowed' : 'text-white bg-white/10 hover:bg-white/20'}"
            on:click={() => goTo(currentStep + 1)} disabled={currentStep === totalSteps - 1}
          >Next →</button>
        </div>

      </div>
    </div>
  {/if}
</main>
