<script lang="ts">
  /**
   * EvalApp — Batch 3D face texture capture tool
   *
   * Loads a folder of DECA/FLAME OBJ subjects, auto-rotates each face to
   * frontal view using the 68-pt DECA landmarks (deca_landmarks.json),
   * and captures a cropped 512×512 PNG named after the subject VID.
   *
   * Browser compat:
   * Chrome/Edge  → showDirectoryPicker()
   * Safari/FF    → <input webkitdirectory> fallback
   */
  import {
    Engine, Scene, ArcRotateCamera, HemisphericLight,
    Vector3, Matrix, Quaternion, Color3, Color4, Mesh,
    SceneLoader, StandardMaterial, Texture, // <-- ĐÃ THÊM Texture
    VertexBuffer, Tools,
  } from "@babylonjs/core";
  import "@babylonjs/loaders/OBJ";
  import { onMount, onDestroy } from "svelte";
  import LM_DATA from "./assets/deca_landmarks.json";

  // ── Landmark constants (68-pt model, 0-indexed) ───────────────────────────
  const LM_FACE_IDX: number[] = LM_DATA.face_idx;
  const LM_BARY:     number[][] = LM_DATA.bary;
  const IDX_NOSE_TIP   = 30;
  const IDX_LEFT_EYE   = [36, 37, 38, 39, 40, 41] as const;
  const IDX_RIGHT_EYE  = [42, 43, 44, 45, 46, 47] as const;

  // ── Types ─────────────────────────────────────────────────────────────────
  interface Subject {
    vid:           string;
    subFolderName: string;
    fileMap:       Map<string, File>;   // lowercase filename → File
  }

  // ── State ─────────────────────────────────────────────────────────────────
  let canvas: HTMLCanvasElement;
  let inputEl: HTMLInputElement;          // hidden <input webkitdirectory>
  let engine: Engine;
  let scene: Scene;
  let camera: ArcRotateCamera;

  let subjects: Subject[] = [];
  let currentIdx = 0;
  $: current = subjects[currentIdx] as Subject | undefined;

  let inputFolderName  = "";
  let isLoading        = false;
  let loadStatus       = "";
  let loadError        = "";

  let outputDirHandle: FileSystemDirectoryHandle | null = null;
  let outputFolderName = "";

  let capturedCount = 0;
  let batchRunning  = false;
  let batchMsg      = "";

  const blobUrls: string[] = [];

  // ── BabylonJS init ────────────────────────────────────────────────────────
  onMount(() => {
    engine = new Engine(canvas, true, {
      preserveDrawingBuffer: true,
      stencil: true,
      antialias: true,
    });
    scene = new Scene(engine);
    scene.clearColor = new Color4(0.13, 0.13, 0.13, 1);
    scene.imageProcessingConfiguration.toneMappingEnabled = false;

    camera = new ArcRotateCamera("cam", Math.PI / 2, Math.PI / 2.5, 1, Vector3.Zero(), scene);
    camera.minZ = 0.001;
    camera.wheelPrecision = 300;
    camera.lowerRadiusLimit = 0.01;
    camera.attachControl(canvas, true);

    new HemisphericLight("light", new Vector3(0, 1, 0), scene);
    engine.runRenderLoop(() => scene.render());
    window.addEventListener("resize", () => engine.resize());
  });

  onDestroy(() => { engine?.dispose(); revokeBlobUrls(); });

  function revokeBlobUrls() { blobUrls.forEach(u => URL.revokeObjectURL(u)); blobUrls.length = 0; }
  function clearScene() {
    scene.meshes.slice().forEach(m => m.dispose());
    scene.materials.slice().forEach(m => m.dispose());
    scene.textures.slice().forEach(t => t.dispose());
    revokeBlobUrls();
  }

  // ── Build subjects from FileList (webkitdirectory) ────────────────────────
  function processFileList(files: FileList) {
    const dirTree = new Map<string, Map<string, File>>();
    let rootName = "";

    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      const parts = f.webkitRelativePath.split("/");
      // parts: [rootFolder, subFolder, ...fileName]
      if (parts.length < 3) continue;
      if (!rootName) rootName = parts[0];
      const subDir   = parts[1];
      const basename = parts[parts.length - 1].toLowerCase();
      if (!dirTree.has(subDir)) dirTree.set(subDir, new Map());
      dirTree.get(subDir)!.set(basename, f);
    }

    inputFolderName = rootName;
    const found: Subject[] = [];

    for (const [subFolderName, fileMap] of dirTree) {
      const hasObj = [...fileMap.keys()].some(k => k.endsWith(".obj") && !k.includes("_detail"));
      if (!hasObj) continue;
      const vidMatch = subFolderName.match(/^(V\d+)/);
      found.push({
        vid: vidMatch ? vidMatch[1] : subFolderName,
        subFolderName,
        fileMap,
      });
    }

    found.sort((a, b) => a.vid.localeCompare(b.vid));
    return found;
  }

  // ── Build subjects from FileSystemDirectoryHandle (Chrome/Edge) ───────────
  async function processDirectoryHandle(dirHandle: FileSystemDirectoryHandle): Promise<Subject[]> {
    const found: Subject[] = [];

    for await (const [name, entry] of dirHandle.entries()) {
      if (entry.kind !== "directory") continue;
      const subDir   = entry as FileSystemDirectoryHandle;
      const fileMap  = new Map<string, File>();
      let   hasObj   = false;

      for await (const [fname, fentry] of subDir.entries()) {
        if (fentry.kind !== "file") continue;
        const f  = await (fentry as FileSystemFileHandle).getFile();
        const fl = fname.toLowerCase();
        fileMap.set(fl, f);
        if (fl.endsWith(".obj") && !fl.includes("_detail")) hasObj = true;
      }
      if (!hasObj) continue;

      const vidMatch = name.match(/^(V\d+)/);
      found.push({
        vid: vidMatch ? vidMatch[1] : name,
        subFolderName: name,
        fileMap,
      });
    }

    found.sort((a, b) => a.vid.localeCompare(b.vid));
    return found;
  }

  // ── Pick input folder (with Safari fallback) ──────────────────────────────
  async function pickInputFolder() {
    loadError  = "";
    loadStatus = "";
    subjects   = [];
    capturedCount = 0;
    batchMsg   = "";

    if (typeof window.showDirectoryPicker === "function") {
      // Chrome / Edge
      try {
        const dirHandle = await window.showDirectoryPicker({ mode: "read" });
        inputFolderName = dirHandle.name;
        isLoading       = true;
        loadStatus      = "Scanning…";
        subjects        = await processDirectoryHandle(dirHandle);
        await afterFolderLoaded();
      } catch (e: unknown) {
        if ((e as DOMException)?.name !== "AbortError")
          loadError = `Error: ${(e as Error).message}`;
        isLoading = false;
      }
    } else {
      // Safari / Firefox — trigger hidden <input webkitdirectory>
      inputEl.click();
    }
  }

  // Called by the hidden input's change event (Safari path)
  async function onInputChange() {
    const files = inputEl.files;
    if (!files || files.length === 0) return;
    isLoading  = true;
    loadStatus = "Scanning…";
    loadError  = "";
    subjects   = processFileList(files);
    await afterFolderLoaded();
    // Reset so the same folder can be re-picked
    inputEl.value = "";
  }

  async function afterFolderLoaded() {
    if (subjects.length === 0) {
      loadError = "No OBJ subjects found in this folder";
      isLoading = false;
      return;
    }
    loadStatus = `Found ${subjects.length} subjects`;
    currentIdx = 0;
    await loadSubject(0);
  }

  // ── Pick output folder ────────────────────────────────────────────────────
  async function pickOutputFolder() {
    if (typeof window.showDirectoryPicker !== "function") {
      alert("Output folder selection requires Chrome/Edge.\nFiles will be downloaded to your Downloads folder.");
      return;
    }
    try {
      outputDirHandle  = await window.showDirectoryPicker({ mode: "readwrite" });
      outputFolderName = outputDirHandle.name;
    } catch (e: unknown) {
      if ((e as DOMException)?.name !== "AbortError")
        alert(`Cannot open output folder: ${(e as Error).message}`);
    }
  }

  // ── Load one subject ──────────────────────────────────────────────────────
  async function loadSubject(idx: number) {
    if (idx < 0 || idx >= subjects.length) return;
    const subj = subjects[idx];
    isLoading  = true;
    loadStatus = `Loading ${subj.vid}…`;
    loadError  = "";
    clearScene();

    try {
      const { fileMap } = subj;

      // Find main OBJ
      let objName: string | undefined;
      for (const k of fileMap.keys()) {
        if (k.endsWith(".obj") && !k.includes("_detail")) { objName = k; break; }
      }
      if (!objName) throw new Error("No OBJ found");

      // Read OBJ text
      let objText = await fileMap.get(objName)!.text();

      // Find MTL: referenced one first, fallback to any .mtl in folder
      const mtlRefMatch = objText.match(/^mtllib\s+(.+)$/m);
      const refKey      = mtlRefMatch?.[1]?.trim().toLowerCase();
      let mtlKey        = (refKey && fileMap.has(refKey)) ? refKey : undefined;
      if (!mtlKey) {
        for (const k of fileMap.keys()) { if (k.endsWith(".mtl")) { mtlKey = k; break; } }
      }

      // Cờ lưu URL của texture để ép bằng tay nếu cần
      let finalTexUrl: string | undefined;

      // Patch MTL texture paths → blob URLs
      if (mtlKey && fileMap.has(mtlKey)) {
        let mtlText = await fileMap.get(mtlKey)!.text();
        mtlText = mtlText.replace(
          /(map_Kd|map_Ka|map_Ks|map_Bump|bump|disp)\s+(\S+)/gi,
          (_, key, rawPath) => {
            const base    = rawPath.replace(/\\/g, "/").split("/").pop()!.toLowerCase();
            let texFile = fileMap.get(base);
            
            // Nếu sai đuôi .jpg/.png
            if (!texFile) {
              const nameWithoutExt = base.includes('.') ? base.substring(0, base.lastIndexOf('.')) : base;
              texFile = fileMap.get(`${nameWithoutExt}.jpg`) || fileMap.get(`${nameWithoutExt}.png`);
            }

            if (!texFile) return `${key} ${rawPath}`;
            const url = URL.createObjectURL(texFile);
            blobUrls.push(url);
            
            // Bắt lấy URL ảnh để dùng làm phương án dự phòng
            if (key.toLowerCase() === 'map_kd') finalTexUrl = url;
            return `${key} ${url}`;
          }
        );
        const mtlUrl = URL.createObjectURL(new Blob([mtlText], { type: "text/plain" }));
        blobUrls.push(mtlUrl);
        objText = mtlRefMatch
          ? objText.replace(/^mtllib\s+.+$/m, `mtllib ${mtlUrl}`)
          : `mtllib ${mtlUrl}\n` + objText;
      }
      
      // DỰ PHÒNG MẠNH: Nếu obj hỏng không có MTL, tự lùng file ảnh trong thư mục
      if (!finalTexUrl) {
        for (const [k, f] of fileMap.entries()) {
          if ((k.endsWith(".png") || k.endsWith(".jpg")) && 
              (k.includes("texture") || k.includes("albedo") || k.includes("map") || k.replace(/\.[^/.]+$/, "") === objName.replace(/\.[^/.]+$/, ""))) {
            finalTexUrl = URL.createObjectURL(f);
            blobUrls.push(finalTexUrl);
            break;
          }
        }
      }

      // ── Compute 3D landmarks BEFORE handing to BabylonJS ─────────────────
      // Parse vertices and face vertex-indices from OBJ text
      const lm3d = computeLandmarksFromObjText(objText);

      const objUrl = URL.createObjectURL(new Blob([objText], { type: "text/plain" }));
      blobUrls.push(objUrl);

      const result = await SceneLoader.ImportMeshAsync("", "", objUrl, scene, undefined, ".obj");
      if (result.meshes.length === 0) throw new Error("No meshes in OBJ");

      // Fix materials & ÉP CHỦ ĐỘNG TEXTURE LÊN MESH
      result.meshes.forEach(m => {
        if (!(m instanceof Mesh)) return;
        if (m.isVerticesDataPresent(VertexBuffer.ColorKind))
          m.removeVerticesData(VertexBuffer.ColorKind);
          
        // Bỏ qua lỗi thiếu lệnh "usemtl" trong OBJ bằng cách ép texture bằng tay
        if (finalTexUrl) {
            let missingTex = true;
            if (m.material instanceof StandardMaterial && m.material.diffuseTexture) missingTex = false;
            
            if (missingTex) {
                const mat = new StandardMaterial(m.name + "_mat", scene);
                const tex = new Texture(finalTexUrl, scene);
                mat.diffuseTexture = tex;
                mat.emissiveTexture = tex;
                mat.emissiveColor = new Color3(1, 1, 1);
                mat.disableLighting = true;
                mat.backFaceCulling = false;
                m.material = mat;
            }
        } 
        
        // Cấu hình lại nếu material tự nhận thành công
        if (m.material instanceof StandardMaterial) {
          m.material.backFaceCulling = false;
          m.material.disableLighting = true;
          const dt = m.material.diffuseTexture;
          if (dt) { m.material.emissiveTexture = dt; m.material.emissiveColor = new Color3(1, 1, 1); }
        }
      });

      await new Promise(r => requestAnimationFrame(r));
      await new Promise(r => requestAnimationFrame(r));

      if (lm3d) {
        autoFrontalFromLandmarks(lm3d);
      } else {
        autoFrontalFallback();   // heuristic if landmark parse failed
      }

      loadStatus = "";
    } catch (e: unknown) {
      console.error(e);
      loadError = `${subj.vid}: ${(e as Error).message ?? e}`;
    } finally {
      isLoading = false;
    }
  }

  // ── Parse OBJ text → 3D landmark positions ───────────────────────────────
  function computeLandmarksFromObjText(objText: string): number[][] | null {
    try {
      const verts: number[][] = [];
      const faces: number[][]  = [];

      for (const line of objText.split("\n")) {
        if (line.startsWith("v ")) {
          const p = line.trim().split(/\s+/);
          verts.push([parseFloat(p[1]), parseFloat(p[2]), parseFloat(p[3])]);
        } else if (line.startsWith("f ")) {
          const tokens = line.trim().split(/\s+/).slice(1, 4);
          faces.push(tokens.map(t => parseInt(t.split("/")[0]) - 1)); // 1-indexed → 0-indexed
        }
      }

      if (verts.length === 0 || faces.length === 0) return null;

      // Compute 3D position for each of the 68 landmarks
      const lm3d: number[][] = [];
      for (let k = 0; k < LM_FACE_IDX.length; k++) {
        const fi = LM_FACE_IDX[k];
        if (fi >= faces.length) return null;
        const [v0, v1, v2] = faces[fi];
        if (v0 === undefined || v1 === undefined || v2 === undefined) return null;
        const [b0, b1, b2] = LM_BARY[k];
        const P0 = verts[v0], P1 = verts[v1], P2 = verts[v2];
        lm3d.push([
          b0 * P0[0] + b1 * P1[0] + b2 * P2[0],
          b0 * P0[1] + b1 * P1[1] + b2 * P2[1],
          b0 * P0[2] + b1 * P1[2] + b2 * P2[2],
        ]);
      }
      return lm3d;
    } catch {
      return null;
    }
  }

  // ── Frontal alignment: ROTATE THE MESH so face points toward +Z ───────────
  //
  //   Strategy: compute the face's 3 orthogonal axes from landmarks, then build
  //   the rotation matrix R that maps  faceRight→+X, faceUp→+Y, faceNorm→+Z.
  //   Apply R to every mesh (pivoting around the face centre), and place the
  //   camera at a fixed canonical position: target=faceCenter, α=π/2, β=π/2
  //   (i.e. camera at faceCenter + r·Ẑ, looking in −Ẑ direction).
  //
  function autoFrontalFromLandmarks(lm: number[][]) {
    const noseTip    = lm[IDX_NOSE_TIP];                        // 30
    const leftEye    = meanPts(IDX_LEFT_EYE.map(i  => lm[i])); // 36-41
    const rightEye   = meanPts(IDX_RIGHT_EYE.map(i => lm[i])); // 42-47
    const eyeCenter  = meanPts([leftEye, rightEye]);
    const chin       = lm[8];   // bottom of jaw
    const noseBridge = lm[27];  // bridge between eyes
    // Interior landmarks only (skip jaw outline 0–16) for a stable pivot
    const faceCenter = meanPts(lm.slice(17));

    // ── 1. Face outward normal ────────────────────────────────────────────
    // cross(rightEye→leftEye, eyeCenter→noseTip), then sign-check
    const eyeVec  = [leftEye[0]-rightEye[0], leftEye[1]-rightEye[1], leftEye[2]-rightEye[2]];
    const noseVec = [noseTip[0]-eyeCenter[0], noseTip[1]-eyeCenter[1], noseTip[2]-eyeCenter[2]];
    let faceNorm = normalize3(cross3(eyeVec, noseVec));
    const noseOut = [noseTip[0]-faceCenter[0], noseTip[1]-faceCenter[1], noseTip[2]-faceCenter[2]];
    if (dot3(faceNorm, noseOut) < 0) faceNorm = [-faceNorm[0], -faceNorm[1], -faceNorm[2]];

    // ── 2. Face "up" direction ────────────────────────────────────────────
    // chin→noseBridge raw, projected out the face-normal component
    const rawUp = [noseBridge[0]-chin[0], noseBridge[1]-chin[1], noseBridge[2]-chin[2]];
    const dotUN = dot3(rawUp, faceNorm);
    const faceUp = normalize3([
      rawUp[0] - dotUN * faceNorm[0],
      rawUp[1] - dotUN * faceNorm[1],
      rawUp[2] - dotUN * faceNorm[2],
    ]);

    // ── 3. Face "right" direction ─────────────────────────────────────────
    const faceRight = normalize3(cross3(faceUp, faceNorm));

    // ── 4. Rotation matrix R: faceRight→+X, faceUp→+Y, faceNorm→+Z ───────
    // BabylonJS uses row-major matrices with row-vector multiplication: v'=v·M
    // For v·M to map faceNorm→(0,0,1), the columns of M must be [fR, fU, fN].
    // In row-major storage each row i = (fR[i], fU[i], fN[i]):
    const rotMat = Matrix.FromValues(
      faceRight[0], faceUp[0], faceNorm[0], 0,   // row 0
      faceRight[1], faceUp[1], faceNorm[1], 0,   // row 1
      faceRight[2], faceUp[2], faceNorm[2], 0,   // row 2
      0,            0,          0,           1,   // row 3
    );
    const qFrontal = new Quaternion();
    Quaternion.FromRotationMatrixToRef(rotMat, qFrontal);

    // ── 5. Apply rotation to every mesh, pivoting around faceCenter ───────
    // world_vertex = local_vertex · rotMat + mesh.position
    // We want:  world_vertex = (local_vertex - C) · rotMat + C
    //         = local_vertex · rotMat + (C - C · rotMat)
    // So: mesh.position = C − (C · rotMat)
    const C  = new Vector3(faceCenter[0], faceCenter[1], faceCenter[2]);
    const RC = Vector3.Zero();
    Vector3.TransformCoordinatesToRef(C, rotMat, RC);   // RC = C · rotMat
    const meshPos = C.subtract(RC);                     // pivot correction

    scene.meshes.forEach(m => {
      if (!(m instanceof Mesh)) return;
      m.rotationQuaternion = qFrontal.clone();
      m.position           = meshPos.clone();
    });

    // ── 6. Fix camera at canonical frontal position ───────────────────────
    const xs = lm.map(p => p[0]), ys = lm.map(p => p[1]);
    const extent = Math.max(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys));

    camera.upVector = new Vector3(0, 1, 0);  // world Y is now face-up
    camera.setTarget(C.clone());
    camera.alpha  = Math.PI / 2;  // pos = target + r·(0,0,1)  ← +Z
    camera.beta   = Math.PI / 2;
    camera.radius = extent * 1.6;

    console.log(
      `[frontal] norm=(${faceNorm[0].toFixed(3)},${faceNorm[1].toFixed(3)},${faceNorm[2].toFixed(3)}) ` +
      `up=(${faceUp[0].toFixed(3)},${faceUp[1].toFixed(3)},${faceUp[2].toFixed(3)}) ` +
      `right=(${faceRight[0].toFixed(3)},${faceRight[1].toFixed(3)},${faceRight[2].toFixed(3)})`
    );
  }

  // ── Fallback: top-Z heuristic (resets any mesh rotation) ─────────────────
  function autoFrontalFallback() {
    // Reset any rotation that was applied by autoFrontalFromLandmarks
    scene.meshes.forEach(m => {
      if (!(m instanceof Mesh)) return;
      m.rotationQuaternion = null;
      m.position = Vector3.Zero();
    });
    camera.upVector = new Vector3(0, 1, 0);
    const pos: Array<[number, number, number]> = [];
    scene.meshes.forEach(m => {
      if (!(m instanceof Mesh)) return;
      const vd = m.getVerticesData(VertexBuffer.PositionKind);
      if (!vd) return;
      for (let i = 0; i < vd.length; i += 3) pos.push([vd[i], vd[i+1], vd[i+2]]);
    });
    if (!pos.length) return;

    const n = pos.length;
    const cx = pos.reduce((s, v) => s + v[0], 0) / n;
    const cy = pos.reduce((s, v) => s + v[1], 0) / n;
    const cz = pos.reduce((s, v) => s + v[2], 0) / n;

    pos.sort((a, b) => b[2] - a[2]);
    const fn = Math.max(1, Math.floor(n * 0.15));
    const front = pos.slice(0, fn);
    const fx = front.reduce((s, v) => s + v[0], 0) / fn;
    const fy = front.reduce((s, v) => s + v[1], 0) / fn;
    const fz = front.reduce((s, v) => s + v[2], 0) / fn;

    const [dx, dy, dz] = normalize3([fx - cx, fy - cy, fz - cz]);
    const xs = pos.map(v => v[0]), ys = pos.map(v => v[1]), zs = pos.map(v => v[2]);
    const extent = Math.max(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys), Math.max(...zs)-Math.min(...zs));

    camera.setTarget(new Vector3(cx, cy, cz));
    camera.alpha  = Math.atan2(dz, dx);
    camera.beta   = Math.acos(Math.max(-1, Math.min(1, dy)));
    camera.radius = extent * 1.8;
  }

  // ── Vector math helpers ───────────────────────────────────────────────────
  function meanPts(pts: number[][]): number[] {
    const n = pts.length;
    return [
      pts.reduce((s, p) => s + p[0], 0) / n,
      pts.reduce((s, p) => s + p[1], 0) / n,
      pts.reduce((s, p) => s + p[2], 0) / n,
    ];
  }
  function normalize3(v: number[]): number[] {
    const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) || 1;
    return [v[0]/len, v[1]/len, v[2]/len];
  }
  function cross3(a: number[], b: number[]): number[] {
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  }
  function dot3(a: number[], b: number[]): number {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  }

  // ── Navigation ────────────────────────────────────────────────────────────
  async function prev() {
    if (currentIdx > 0) { currentIdx--; await loadSubject(currentIdx); }
  }
  async function next() {
    if (currentIdx < subjects.length - 1) { currentIdx++; await loadSubject(currentIdx); }
  }
  async function jumpTo(e: Event) {
    currentIdx = parseInt((e.target as HTMLSelectElement).value);
    await loadSubject(currentIdx);
  }

  // ── Capture: render at CURRENT camera → crop to mesh screen bounds → 512×512
  async function capture(): Promise<void> {
    if (!current || isLoading) return;

    // Force one render so camera state is fully applied
    scene.render();
    await new Promise(r => requestAnimationFrame(r));

    const CAP = 1024;
    const rawDataUrl = await Tools.CreateScreenshotUsingRenderTargetAsync(
      engine, camera, { width: CAP, height: CAP },
    );

    // Project mesh bounding box onto screen space to get precise crop region
    const screenBounds = getMeshScreenBounds(CAP, CAP);
    const croppedUrl   = screenBounds
      ? await cropToRect(rawDataUrl, screenBounds, 512)
      : await cropByPixels(rawDataUrl, 512);

    const filename = `${current.vid}.png`;
    await saveFile(croppedUrl, filename);
    capturedCount++;
    loadStatus = `✓ ${filename}`;
    setTimeout(() => { if (loadStatus.startsWith("✓")) loadStatus = ""; }, 1800);
  }

  /** Project ALL mesh vertices to screen space → tight pixel bounding rect */
  function getMeshScreenBounds(W: number, H: number): { x0: number; y0: number; x1: number; y1: number } | null {
    const transformMatrix = scene.getTransformMatrix();
    const viewport = camera.viewport.toGlobal(W, H);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    let found = false;

    const tmpWP = new Vector3();
    const tmpSP = new Vector3();

    scene.meshes.forEach(mesh => {
      if (!(mesh instanceof Mesh)) return;
      const vd = mesh.getVerticesData(VertexBuffer.PositionKind);
      if (!vd) return;
      const worldMat = mesh.getWorldMatrix();
      // Sample every vertex (DECA has ~5k verts — fast enough)
      for (let i = 0; i + 2 < vd.length; i += 3) {
        tmpWP.set(vd[i], vd[i+1], vd[i+2]);
        Vector3.TransformCoordinatesToRef(tmpWP, worldMat, tmpWP);
        Vector3.ProjectToRef(tmpWP, Matrix.Identity(), transformMatrix, viewport, tmpSP);
        if (tmpSP.z < 0 || tmpSP.z > 1) continue; // outside depth range
        found = true;
        if (tmpSP.x < x0) x0 = tmpSP.x;
        if (tmpSP.x > x1) x1 = tmpSP.x;
        if (tmpSP.y < y0) y0 = tmpSP.y;
        if (tmpSP.y > y1) y1 = tmpSP.y;
      }
    });

    if (!found || x1 <= x0 || y1 <= y0) return null;
    const pad = Math.ceil(Math.max(x1 - x0, y1 - y0) * 0.04);
    return {
      x0: Math.max(0, Math.floor(x0) - pad),
      y0: Math.max(0, Math.floor(y0) - pad),
      x1: Math.min(W, Math.ceil(x1)  + pad),
      y1: Math.min(H, Math.ceil(y1)  + pad),
    };
  }

  /** Crop to the projected mesh bounds (square), output outSize×outSize */
  async function cropToRect(
    dataUrl: string,
    bounds: { x0: number; y0: number; x1: number; y1: number },
    outSize: number,
    W = 1024,
    H = 1024,
  ): Promise<string> {
    return new Promise(resolve => {
      const img = new Image();
      img.onload = () => {
        // Make crop square around the centre of the bounds
        const bw   = bounds.x1 - bounds.x0;
        const bh   = bounds.y1 - bounds.y0;
        const side = Math.max(bw, bh);
        const midX = (bounds.x0 + bounds.x1) / 2;
        const midY = (bounds.y0 + bounds.y1) / 2;
        // Clamp so we don't go outside the image
        const sx = Math.max(0, Math.min(W - side, midX - side / 2));
        const sy = Math.max(0, Math.min(H - side, midY - side / 2));
        const out = document.createElement("canvas");
        out.width = out.height = outSize;
        out.getContext("2d")!.drawImage(img, sx, sy, side, side, 0, 0, outSize, outSize);
        resolve(out.toDataURL("image/png"));
      };
      img.src = dataUrl;
    });
  }

  /** Fallback: find non-background pixels to determine crop region */
  async function cropByPixels(dataUrl: string, outSize: number): Promise<string> {
    return new Promise(resolve => {
      const img = new Image();
      img.onload = () => {
        const S = img.width;
        const tmp = document.createElement("canvas");
        tmp.width = tmp.height = S;
        const ctx = tmp.getContext("2d")!;
        ctx.drawImage(img, 0, 0);
        const data = ctx.getImageData(0, 0, S, S).data;
        const BG = 33, THR = 20;
        let x0 = S, x1 = 0, y0 = S, y1 = 0;
        for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
          const i = (y * S + x) * 4;
          if (Math.abs(data[i]-BG) > THR || Math.abs(data[i+1]-BG) > THR || Math.abs(data[i+2]-BG) > THR) {
            if (x < x0) x0 = x; if (x > x1) x1 = x;
            if (y < y0) y0 = y; if (y > y1) y1 = y;
          }
        }
        if (x1 <= x0 || y1 <= y0) { resolve(dataUrl); return; }
        const pad = Math.floor(S * 0.04);
        x0 = Math.max(0, x0-pad); x1 = Math.min(S-1, x1+pad);
        y0 = Math.max(0, y0-pad); y1 = Math.min(S-1, y1+pad);
        const side = Math.max(x1-x0, y1-y0);
        const midX = (x0+x1)/2, midY = (y0+y1)/2;
        const sx = Math.max(0, midX-side/2), sy = Math.max(0, midY-side/2);
        const out = document.createElement("canvas");
        out.width = out.height = outSize;
        out.getContext("2d")!.drawImage(tmp, sx, sy, side, side, 0, 0, outSize, outSize);
        resolve(out.toDataURL("image/png"));
      };
      img.src = dataUrl;
    });
  }

  async function saveFile(dataUrl: string, filename: string) {
    if (outputDirHandle) {
      try {
        const fh = await outputDirHandle.getFileHandle(filename, { create: true });
        const wr = await fh.createWritable();
        await wr.write(await (await fetch(dataUrl)).blob());
        await wr.close();
        return;
      } catch { /* fallthrough */ }
    }
    const a = document.createElement("a");
    a.href = dataUrl; a.download = filename; a.click();
  }

  // ── Capture all subjects ──────────────────────────────────────────────────
  async function captureAll() {
    if (!outputDirHandle) {
      if (!confirm("No output folder set — files will be downloaded one by one.\nContinue?")) return;
    }
    batchRunning = true; capturedCount = 0;
    for (let i = 0; i < subjects.length; i++) {
      currentIdx = i;
      batchMsg   = `${i + 1}/${subjects.length}  —  ${subjects[i].vid}`;
      await loadSubject(i);
      await new Promise(r => setTimeout(r, 800));
      await capture();
      await new Promise(r => setTimeout(r, 100));
    }
    batchMsg = `✓ Done — ${subjects.length} files saved`;
    batchRunning = false;
  }
</script>

<main class="flex flex-col h-screen bg-gray-950 text-white overflow-hidden select-none">

  <input
    bind:this={inputEl}
    type="file"
    style="display:none"
    webkitdirectory
    multiple
    on:change={onInputChange}
  />

  <header class="flex items-center gap-2 px-3 py-2 bg-gray-800 border-b border-gray-700 flex-shrink-0 text-sm flex-wrap">

    <button on:click={pickInputFolder} disabled={isLoading || batchRunning}
            class="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded font-semibold disabled:opacity-40 whitespace-nowrap">
      📂 Load Folder
    </button>

    {#if inputFolderName}
      <span class="text-blue-300 font-mono text-xs truncate max-w-[200px]" title={inputFolderName}>
        {inputFolderName}
      </span>
    {/if}

    <div class="w-px h-5 bg-gray-600" />

    {#if subjects.length > 0}
      <button on:click={prev} disabled={currentIdx === 0 || isLoading || batchRunning}
              class="px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded disabled:opacity-40">◀</button>

      <select value={currentIdx} on:change={jumpTo} disabled={isLoading || batchRunning}
              class="bg-gray-700 border border-gray-500 rounded px-2 py-1 font-mono text-xs max-w-[240px]">
        {#each subjects as s, i}
          <option value={i}>{s.vid}  ·  {s.subFolderName}</option>
        {/each}
      </select>

      <button on:click={next} disabled={currentIdx === subjects.length - 1 || isLoading || batchRunning}
              class="px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded disabled:opacity-40">▶</button>

      <span class="text-gray-400 text-xs">{currentIdx + 1}/{subjects.length}</span>
    {/if}

    <div class="flex-1" />

    <button on:click={pickOutputFolder} disabled={batchRunning}
            class="px-3 py-1.5 rounded text-xs disabled:opacity-40 whitespace-nowrap
                   {outputFolderName ? 'bg-green-800 hover:bg-green-700 text-green-200' : 'bg-gray-600 hover:bg-gray-500'}">
      💾 {outputFolderName || "Set Output Folder"}
    </button>

    <div class="w-px h-5 bg-gray-600" />

    <button on:click={capture} disabled={!current || isLoading || batchRunning}
            class="px-3 py-1.5 bg-green-600 hover:bg-green-500 rounded font-semibold disabled:opacity-40 whitespace-nowrap">
      📸 Capture
    </button>

    <button on:click={captureAll} disabled={subjects.length === 0 || isLoading || batchRunning}
            class="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 rounded font-semibold disabled:opacity-40 whitespace-nowrap">
      📸 Capture ALL ({subjects.length})
    </button>
  </header>

  <div class="flex-1 relative min-h-0">
    <canvas bind:this={canvas} class="w-full h-full touch-none outline-none" />

    {#if isLoading}
      <div class="absolute inset-0 flex flex-col items-center justify-center bg-black/60 gap-3 pointer-events-none">
        <div class="w-10 h-10 border-4 border-blue-400 border-t-transparent rounded-full animate-spin" />
        <div class="text-blue-300 font-mono text-sm">{loadStatus}</div>
      </div>
    {/if}

    {#if loadError}
      <div class="absolute top-3 left-3 bg-red-900/90 border border-red-500 rounded px-3 py-2 text-red-200 text-xs max-w-md">
        ⚠ {loadError}
      </div>
    {/if}

    {#if current && !isLoading}
      <div class="absolute top-3 right-3 bg-black/70 backdrop-blur rounded px-3 py-2 font-mono text-xs">
        <div class="text-blue-300 font-bold text-sm">{current.vid}</div>
        <div class="text-gray-400 truncate max-w-[200px]">{current.subFolderName}</div>
        <div class="text-gray-500 mt-1">{capturedCount} captured</div>
      </div>
    {/if}
  </div>

  <footer class="flex items-center gap-3 px-3 py-2 bg-gray-800 border-t border-gray-700 flex-shrink-0 text-xs min-h-[2.5rem]">

    {#if batchRunning}
      <div class="flex items-center gap-2 text-purple-300">
        <div class="w-3 h-3 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
        <span class="font-mono">{batchMsg}</span>
      </div>
    {:else if batchMsg}
      <span class="text-green-400 font-mono">{batchMsg}</span>
    {:else if loadStatus}
      <span class="text-blue-300 font-mono">{loadStatus}</span>
    {:else if subjects.length === 0}
      <span class="text-gray-500 italic">
        Click 📂 Load Folder → pick results_C4_final or output_batch.
        Each model auto-rotates to frontal via DECA landmarks.
      </span>
    {:else}
      <span class="text-gray-500">Drag to rotate · Scroll to zoom</span>
    {/if}

    <div class="flex-1" />

    {#if current && !isLoading}
      <button on:click={() => { if (current) { const s = current; loadSubject(subjects.indexOf(s)); } }}
              class="px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded text-xs"
              title="Reload and re-align">
        🔄 Reload
      </button>
      <button on:click={autoFrontalFallback}
              class="px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded text-xs"
              title="Reset camera to frontal">
        🎯 Re-frontal
      </button>
    {/if}
  </footer>
</main>

<style>
  :global(body) { margin: 0; overflow: hidden; }
</style>