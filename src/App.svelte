<!-- <script lang="ts">
  import {
    Engine,
    Scene,
    ArcRotateCamera,
    HemisphericLight,
    Vector3,
    Mesh,
    Node,
    SceneLoader,
    Color4,
  } from "@babylonjs/core";
  import "@babylonjs/loaders/OBJ"; // Import bộ đọc OBJ
  import { onMount, onDestroy, tick } from "svelte";

  // Import các thành phần giao diện cũ
  import FrameSlider from "./lib/FrameSlider.svelte";
  import { secondsToMMSS } from "./lib/utils";
  import { framePassed } from "./lib/store";
  import { Howl } from "howler";
  import {
    IconDeviceTv,
    IconDeviceTvOff,
    IconPlayerPauseFilled,
    IconPlayerPlayFilled,
    IconVolume,
    IconVolumeOff,
  } from "@tabler/icons-svelte";

  // --- BIẾN HỆ THỐNG ---
  let canvas: HTMLCanvasElement;
  let engine: Engine;
  let scene: Scene;
  
  // --- BIẾN GIAO DIỆN ---
  let resizer: HTMLDivElement;
  let leftPanel: HTMLDivElement;
  let videoPlayer: HTMLVideoElement;

  let objectsToShow: Array<Node | undefined> = [];
  let objectsLoaded = 0;
  
  // Loading & Trạng thái
  let isLoading = false;
  let loadingStatus = "";
  let hasVideo = false;

  // Playback
  let isPlaying = false;
  let isMuted = false;
  let toggleShowVideo = true;
  let fps = 0;

  // Settings
  const fpsSettings = [24, 30, 60];
  let currentFpsSetting = 1;
  $: targetFps = fpsSettings[currentFpsSetting];

  const playbackRateSettings = [0.5, 0.75, 1, 2, 3];
  let currentPlaybackRateSetting = 2;
  $: playbackRate = playbackRateSettings[currentPlaybackRateSetting];

  // Tính toán Frame
  $: totalFrames = objectsToShow.length;
  $: currentFrameExact = totalFrames === 0 ? 0 : $framePassed % totalFrames;
  $: currentFrame = Math.round(currentFrameExact);
  $: currentSecond = currentFrameExact / targetFps;

  // Âm thanh
  let soundPlayer = null as Howl | null;
  $: soundPlayer != null && (isPlaying ? soundPlayer.play() : soundPlayer.pause());
  $: soundPlayer != null && soundPlayer.rate(playbackRate);
  $: soundPlayer != null && soundPlayer.mute(isMuted);

  // Video Sync
  $: videoPlayer != null && (isPlaying ? videoPlayer.play() : videoPlayer.pause());
  $: videoPlayer != null && (videoPlayer.playbackRate = playbackRate);

  const AUDIO_FILE_EXTENSIONS = ["mp3", "mpeg", "opus", "ogg", "oga", "wav", "aac", "caf", "m4a", "m4b", "mp4", "weba", "webm", "dolby", "flac"];

  // Cache để dọn dẹp bộ nhớ
  const blobUrls: string[] = [];

  // --- HÀM TIỆN ÍCH TỪ CODE CỦA BẠN ---
  const cleanName = (p: string) => p.replace(/\\/g, "/").split("/").pop()!.toLowerCase();

  // --- LOGIC LOAD FILE (KẾT HỢP LOGIC MỚI + UI CŨ) ---
  const loadFile = async () => {
    try {
        const dirHandle = await window.showDirectoryPicker();
        
        isLoading = true;
        loadingStatus = "Đang quét file...";
        
        // Reset toàn bộ
        objectsToShow.forEach((o) => o?.dispose());
        objectsToShow = [];
        objectsLoaded = 0;
        $framePassed = 0;
        
        // Clear cache cũ
blobUrls.forEach(url => URL.revokeObjectURL(url));
        blobUrls.length = 0; // Xóa mảng
        if (soundPlayer) { soundPlayer.unload(); soundPlayer = null; }

        const fileMap = new Map<string, File>();
        const objFiles: File[] = [];
        let soundFileHandle: FileSystemFileHandle | null = null;

        // 1. Quét file
        for await (const entry of dirHandle.values()) {
            if (entry.kind !== "file") continue;
            if (entry.name.startsWith(".") || entry.name.startsWith("._")) continue;

            const file = await entry.getFile();
            fileMap.set(file.name.toLowerCase(), file);

            if (file.name.toLowerCase().endsWith(".obj") && !file.name.includes("_detail")) {
                objFiles.push(file);
            } else if (AUDIO_FILE_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext))) {
                soundFileHandle = entry;
            }
        }

        if (objFiles.length === 0) {
            alert("Không tìm thấy file OBJ nào!");
            isLoading = false;
            return;
        }

        // 2. Sắp xếp
        objFiles.sort((a, b) => 
            parseInt(a.name.match(/\d+/)?.[0] ?? "0") - parseInt(b.name.match(/\d+/)?.[0] ?? "0")
        );

        loadingStatus = `Tìm thấy ${objFiles.length} frames. Đang xử lý...`;

        // 3. Load từng frame (Dùng logic của bạn đã fix)
        for (let i = 0; i < objFiles.length; i++) {
            if (i % 5 === 0) {
                loadingStatus = `Loading frame ${i}/${objFiles.length}`;
                await new Promise(r => setTimeout(r, 0)); // Cập nhật UI
            }

            let objText = await objFiles[i].text();

            // Xử lý MTL
            const mtlMatch = objText.match(/^mtllib\s+(.*)$/m);
            if (mtlMatch) {
                const mtlName = cleanName(mtlMatch[1]);
                const mtlFile = fileMap.get(mtlName);

                if (mtlFile) {
                    let mtlText = await mtlFile.text();
                    // Regex thay thế đường dẫn texture
                    mtlText = mtlText.replace(
                        /(map_Kd|map_Ka|map_Ks|map_Bump|bump|norm|disp)\s+(.*)/gi,
                        (_, key, raw) => {
                            const texName = cleanName(raw);
                            const texFile = fileMap.get(texName);
                            if (!texFile) return `${key} ${raw}`;

                            const texUrl = URL.createObjectURL(texFile);
                            blobUrls.push(texUrl);
                            return `${key} ${texUrl}`;
                        },
                    );

                    const mtlUrl = URL.createObjectURL(new Blob([mtlText], { type: "text/plain" }));
                    blobUrls.push(mtlUrl);
                    objText = objText.replace(/^mtllib\s+.*$/m, `mtllib ${mtlUrl}`);
                }
            }
const objUrl = URL.createObjectURL(new Blob([objText], { type: "text/plain" }));
            blobUrls.push(objUrl);

            // Import vào Babylon
            const res = await SceneLoader.ImportMeshAsync("", "", objUrl, scene, undefined, ".obj");

            const root = new Mesh(`Frame_${i}`, scene);
            res.meshes.forEach((m) => {
                if (m instanceof Mesh) {
                    m.setParent(root);
                    if (m.material) {
                         m.material.backFaceCulling = false;
                         // Fix lỗi trắng lóa nếu cần
                         (m.material as any).ambientColor = new Color4(0,0,0,1);
                    }
                }
            });

            root.setEnabled(false);
            objectsToShow[i] = root;
            objectsLoaded++;
        }

        // 4. Auto Focus Camera
        if (objectsToShow[0]) {
            objectsToShow[0].setEnabled(true);
            const { min, max } = objectsToShow[0].getHierarchyBoundingVectors();
            objectsToShow[0].setEnabled(false); // Ẩn lại để render loop lo
            
            const center = max.add(min).scale(0.5);
            const size = max.subtract(min);
            const maxDim = Math.max(size.x, size.y, size.z);

            if (scene.activeCamera) {
                (scene.activeCamera as ArcRotateCamera).setTarget(center);
                (scene.activeCamera as ArcRotateCamera).radius = maxDim * 2;
            }
        }

        // 5. Load Audio/Video
        if (soundFileHandle) {
            const file = await soundFileHandle.getFile();
            const url = URL.createObjectURL(file);
            blobUrls.push(url);

            if (file.name.endsWith("mp4") || file.name.endsWith("webm")) {
                videoPlayer.src = url;
                videoPlayer.currentTime = currentSecond;
                videoPlayer.muted = true;
                hasVideo = true;
            }

            soundPlayer = new Howl({
                src: [url],
                format: AUDIO_FILE_EXTENSIONS,
                loop: true,
            });
            soundPlayer.seek(currentSecond);
        }

    } catch (e) {
        console.error(e);
        alert("Lỗi hoặc hủy chọn folder");
    } finally {
        isLoading = false;
        engine.resize();
    }
  };

  // --- SCENE SETUP ---
  onMount(() => {
    engine = new Engine(canvas, true);
    scene = new Scene(engine);
    // Màu nền xám đậm như code bạn test thành công
    scene.clearColor = new Color4(0.15, 0.15, 0.15, 1);
    scene.imageProcessingConfiguration.toneMappingEnabled = false;

    const camera = new ArcRotateCamera(
      "camera",
      Math.PI / 2,
      Math.PI / 2.5,
      2,
      Vector3.Zero(),
      scene,
    );
    camera.attachControl(canvas, true);
    camera.minZ = 0.01;
    camera.wheelPrecision = 50;

    new HemisphericLight("light", new Vector3(0, 1, 0), scene);

    engine.runRenderLoop(() => {
fps = engine.getFps();
      if (isLoading) return;

      objectsToShow.forEach((o) => o?.setEnabled(false));
      objectsToShow[currentFrame]?.setEnabled(true);

      scene.render();

      if (isPlaying && totalFrames > 0) {
        $framePassed += (scene.deltaTime / 1000) * targetFps * playbackRate;
      }
      
      // Sync Audio/Video
      if (soundPlayer && Math.abs(soundPlayer.seek() - currentSecond) > 0.1) {
        soundPlayer.seek(currentSecond);
        if (videoPlayer) videoPlayer.currentTime = currentSecond;
      }
    });

    window.addEventListener("resize", () => engine.resize());
    setupResizer();
  });

  // Logic kéo thả panel trái/phải
  function setupResizer() {
    let isMouseDown = false;
    function mouseMove(event: MouseEvent) { if (isMouseDown) leftPanel.style.flexBasis = event.clientX + "px"; else end(); }
    function end() { isMouseDown = false; engine.resize(); window.removeEventListener("mousemove", mouseMove); window.removeEventListener("mouseup", end); }
    resizer.addEventListener("mousedown", () => { isMouseDown = true; window.addEventListener("mousemove", mouseMove); window.addEventListener("mouseup", end); });
  }

  onDestroy(() => {
    engine.dispose();
    if(soundPlayer) soundPlayer.unload();
  });
</script>

<main class="max-h-[100vh] relative overflow-hidden">
  {#if isLoading}
    <div class="absolute inset-0 z-50 flex items-center justify-center bg-black/80 text-white flex-col gap-4">
      <div class="text-2xl font-bold animate-pulse">Processing...</div>
      <div class="text-blue-300">{loadingStatus}</div>
    </div>
  {/if}

  <div class="absolute bg-white/50 p-2 z-10 top-0 left-0">{fps.toFixed(0)} FPS</div>
  <button class="absolute bg-white/90 p-3 right-2 top-2 z-10 rounded-lg shadow-lg font-bold hover:bg-blue-100 transition-colors" on:click={loadFile} disabled={isLoading}>
    📂 Load Folder
  </button>

  <div class="fixed left-0 w-[100vw] bottom-12 flex justify-center z-10 pointer-events-none">
    <div class="w-[40rem] max-w-[90vw] p-5 bg-blue-200/70 rounded-xl border-gray-400 border shadow-md backdrop-blur-md grainy-bg flex flex-col justify-center pointer-events-auto">
      <div>
        <FrameSlider {totalFrames} {currentFrame} frameRate={targetFps} />
      </div>
      <div class="flex justify-between mt-2 text-sm font-mono font-semibold text-gray-700">
        <div>{secondsToMMSS(currentSecond)}</div>
        <div>Frame {currentFrame + 1}/{totalFrames}</div>
        <div>{secondsToMMSS(totalFrames / targetFps)}</div>
      </div>

      <div class="flex gap-2 justify-between mt-3">
        <div class="flex gap-2 w-[15rem]">
          <button class="p-2 bg-white/50 rounded hover:bg-white transition-colors" on:click={() => { currentFpsSetting = (currentFpsSetting + 1) % fpsSettings.length; }}>{targetFps} FPS</button>
<button class="p-2 bg-white/50 rounded hover:bg-white transition-colors" on:click={() => { currentPlaybackRateSetting = (currentPlaybackRateSetting + 1) % playbackRateSettings.length; }}>x{playbackRate}</button>
        </div>
        <button class="p-3 bg-blue-500/80 text-white rounded-full hover:bg-blue-600 transition-colors w-12 h-12 flex items-center justify-center shadow-sm" on:click={() => (isPlaying = !isPlaying)}>
          {#if isPlaying}<IconPlayerPauseFilled />{:else}<IconPlayerPlayFilled />{/if}
        </button>
        <div class="flex w-[15rem] justify-end gap-2">
          <button class="p-2 bg-white/50 rounded hover:bg-white transition-colors" on:click={async () => { toggleShowVideo = !toggleShowVideo; await tick(); engine.resize(); }}>
            {#if toggleShowVideo}<IconDeviceTv />{:else}<IconDeviceTvOff />{/if}
          </button>
          <button class="p-2 bg-white/50 rounded hover:bg-white transition-colors" on:click={() => (isMuted = !isMuted)}>
            {#if isMuted}<IconVolumeOff />{:else}<IconVolume />{/if}
          </button>
        </div>
      </div>
    </div>
  </div>

  <svg width="0" height="0"><filter id="grainy" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency=".537" /><feColorMatrix type="saturate" values="0" /><feBlend in="SourceGraphic" mode="multiply" /></filter></svg>
  
  <section class="flex w-full h-[100dvh]">
    <div class="flex-grow-0 flex-shrink-0" bind:this={leftPanel} style={`flex-basis: calc(100% - ${toggleShowVideo && hasVideo ? "50%" : "0px"})`}>
      <canvas id="renderCanvas" class="touch-none h-full w-full outline-none" bind:this={canvas} />
    </div>
    <div class="bg-blue-300 cursor-col-resize flex-[0_0_0.5rem] hover:bg-blue-500 transition-colors" style={`display: ${toggleShowVideo && hasVideo ? "block" : "none"}`} bind:this={resizer} />
    <div class="flex-[1_1_auto] overflow-x-auto bg-black items-center justify-center flex" style={`display: ${toggleShowVideo && hasVideo ? "flex" : "none"}`}>
      <video class="max-w-full max-h-full object-contain" bind:this={videoPlayer} />
    </div>
  </section>
</main>

<style>
  .grainy-bg { backdrop-filter: blur(12px) url(#grainy); }
  :global(main > div) { position: relative; }
</style> -->


<script lang="ts">
  import {
    Engine,
    Scene,
    ArcRotateCamera,
    HemisphericLight,
    Vector3,
    Mesh,
    Node,
    SceneLoader,
    Color3,
    Color4,
    StandardMaterial,
    Texture,
    VertexBuffer
  } from "@babylonjs/core";
  import "@babylonjs/loaders/OBJ"; 
  import { onMount, onDestroy, tick } from "svelte";
  import FrameSlider from "./lib/FrameSlider.svelte";
  import { secondsToMMSS } from "./lib/utils";
  import { framePassed } from "./lib/store";
  import { Howl } from "howler";
  import { IconDeviceTv, IconDeviceTvOff, IconPlayerPauseFilled, IconPlayerPlayFilled, IconVolume, IconVolumeOff } from "@tabler/icons-svelte";

  let canvas: HTMLCanvasElement;
let engine: Engine;
  let scene: Scene;
  let resizer: HTMLDivElement;
  let leftPanel: HTMLDivElement;
  let videoPlayer: HTMLVideoElement;
  let objectsToShow: Array<Node | undefined> = [];
  let objectsLoaded = 0;
  let isLoading = false;
  let loadingStatus = "";
  let hasVideo = false;
  let isPlaying = false;
  let isMuted = false;
  let toggleShowVideo = true;
  let fps = 0;
  const fpsSettings = [24, 30, 60];
  let currentFpsSetting = 1;
  $: targetFps = fpsSettings[currentFpsSetting];
  const playbackRateSettings = [0.5, 0.75, 1, 2, 3];
  let currentPlaybackRateSetting = 2;
  $: playbackRate = playbackRateSettings[currentPlaybackRateSetting];
  $: totalFrames = objectsToShow.length;
  $: currentFrameExact = totalFrames === 0 ? 0 : $framePassed % totalFrames;
  $: currentFrame = Math.round(currentFrameExact);
  $: currentSecond = currentFrameExact / targetFps;
  let soundPlayer: Howl | null = null;
  const blobUrls: string[] = [];

  const log = (msg: string, type: 'info' | 'warn' | 'error' = 'info') => {
      console.log(`%c[3D Viewer] ${msg}`, `color: ${type === 'error' ? 'red' : type === 'warn' ? 'orange' : '#00ff00'}; font-weight: bold`);
  };

  const loadFile = async () => {
    try {
        const dirHandle = await window.showDirectoryPicker();
        isLoading = true;
        
        objectsToShow.forEach((o) => o?.dispose());
        objectsToShow = [];
        objectsLoaded = 0;
        $framePassed = 0;
        blobUrls.forEach(url => URL.revokeObjectURL(url));
        blobUrls.length = 0;

        const fileMap = new Map<string, File>();
        const objFiles: File[] = [];
        let globalTextureFile: File | null = null;

        log("--- BẮT ĐẦU QUÉT FOLDER ---");
        for await (const entry of dirHandle.values()) {
            if (entry.kind !== "file") continue;
            const file = await entry.getFile();
            const lowerName = file.name.toLowerCase();
            fileMap.set(lowerName, file);
            
            console.log(`Found: ${file.name} (${file.size} bytes)`);

            if (lowerName.endsWith(".obj") && !file.name.includes("_detail")) {
                objFiles.push(file);
            } else if (lowerName.endsWith(".png") || lowerName.endsWith(".jpg")) {
                if (!globalTextureFile) globalTextureFile = file;
                if (lowerName.includes("texture") || lowerName.includes("uv") || lowerName.includes("map")) {
                    globalTextureFile = file;
                }
            }
        }

        if (objFiles.length === 0) {
            alert("Không tìm thấy file OBJ nào!");
            isLoading = false;
            return;
        }

        objFiles.sort((a, b) => parseInt(a.name.match(/\d+/)?.[0] ?? "0") - parseInt(b.name.match(/\d+/)?.[0] ?? "0"));
        log(`Tìm thấy ${objFiles.length} file OBJ.`);
        if (globalTextureFile) log(`Texture chung: ${globalTextureFile.name}`);
else log("Cảnh báo: Không tìm thấy Texture chung nào!", "warn");

        // 2. Load từng frame
        for (let i = 0; i < objFiles.length; i++) {
            if (i % 5 === 0) {
                loadingStatus = `Debug Frame ${i + 1}/${objFiles.length}`;
                await new Promise(r => setTimeout(r, 0));
            }

            const objText = await objFiles[i].text();
            
            const hasUVText = objText.includes("\nvt ") || objText.startsWith("vt ");
            if (!hasUVText && i === 0) {
                log(`LỖI NGHIÊM TRỌNG: File ${objFiles[i].name} hoàn toàn không có dòng 'vt' (UV Map). Model sẽ không thể dán ảnh!`, "error");
            }

            let manualTextureUrl: string | null = null;
            let texSource = "NONE";
            
            let exactTexName = objFiles[i].name.replace(/\.obj$/i, ".png").toLowerCase();
            let texFile = fileMap.get(exactTexName);
            if (!texFile) {
                exactTexName = objFiles[i].name.replace(/\.obj$/i, ".jpg").toLowerCase();
                texFile = fileMap.get(exactTexName);
            }
            
            if (texFile) {
                texSource = "EXACT_MATCH";
            } else if (globalTextureFile) {
                texFile = globalTextureFile;
                texSource = "GLOBAL_FALLBACK";
            }

            if (texFile) {
                manualTextureUrl = URL.createObjectURL(texFile);
                blobUrls.push(manualTextureUrl);
            }

            if (i === 0) log(`Frame 0 Texture Source: ${texSource} (${texFile?.name})`);

            const objUrl = URL.createObjectURL(new Blob([objText], { type: "text/plain" }));
            blobUrls.push(objUrl);

            try {
                const res = await SceneLoader.ImportMeshAsync("", "", objUrl, scene, undefined, ".obj");
                const root = new Mesh(`Frame_${i}`, scene);

                res.meshes.forEach((m) => {
                    if (m instanceof Mesh) {
                        m.setParent(root);

                        const mat = new StandardMaterial(`mat_${i}`, scene);
                        
                        if (!m.isVerticesDataPresent(VertexBuffer.UVKind)) {
                            if (i===0) log(`Mesh ${m.name} không có dữ liệu UV -> Tô màu ĐỎ`, "error");
                            mat.diffuseColor = new Color3(1, 0, 0); // Đỏ
                            mat.emissiveColor = new Color3(1, 0, 0);
                            m.material = mat;
                            return;
                        }

                        if (m.isVerticesDataPresent(VertexBuffer.ColorKind)) {
                            if (i===0) log(`Mesh ${m.name} có Vertex Color (gây ám màu nâu) -> Đang xóa...`, "warn");
                            m.removeVerticesData(VertexBuffer.ColorKind);
                        }

                        if (manualTextureUrl) {
const tex = new Texture(manualTextureUrl, scene);
                            tex.uScale = 1;

                            mat.diffuseTexture = tex;
                            mat.emissiveTexture = tex;
                            mat.emissiveColor = new Color3(1, 1, 1);
                            
                            mat.disableLighting = true;
                            mat.backFaceCulling = false;
                            
                            m.material = mat;
                        } else {
                            if (i===0) log(`Không load được texture -> Tô màu XANH`, "warn");
                            mat.diffuseColor = new Color3(0, 0, 1);
                            mat.emissiveColor = new Color3(0, 0, 1);
                            m.material = mat;
                        }
                    }
                });

                root.setEnabled(false);
                objectsToShow[i] = root;
                objectsLoaded++;
            } catch (err) {
                console.error(`Error loading frame ${i}:`, err);
            }
        }

        if (objectsToShow[0]) {
            objectsToShow[0].setEnabled(true);
            const { min, max } = objectsToShow[0].getHierarchyBoundingVectors();
            objectsToShow[0].setEnabled(false);
            const center = max.add(min).scale(0.5);
            if (scene.activeCamera) {
                (scene.activeCamera as ArcRotateCamera).setTarget(center);
            }
        }
    } catch (e) {
        console.error(e);
        alert("Lỗi: " + e.message);
    } finally {
        isLoading = false;
        engine.resize();
    }
  };

  onMount(() => {
    engine = new Engine(canvas, true);
    scene = new Scene(engine);
    scene.clearColor = new Color4(0.2, 0.2, 0.2, 1);
    const camera = new ArcRotateCamera("camera", Math.PI/2, Math.PI/2.5, 3, Vector3.Zero(), scene);
    camera.attachControl(canvas, true);
    new HemisphericLight("light", new Vector3(0, 1, 0), scene);
    engine.runRenderLoop(() => {
      fps = engine.getFps();
      if (isLoading) return;
      const frameIndex = currentFrame % objectsToShow.length;
      if (objectsToShow[frameIndex]) {
          scene.meshes.forEach(m => {
             if (m.parent === null && m.name.startsWith("Frame_") && m.name !== `Frame_${frameIndex}`) m.setEnabled(false);
          });
          objectsToShow[frameIndex]?.setEnabled(true);
      }
      scene.render();
      if (isPlaying && totalFrames > 0) $framePassed += (scene.deltaTime / 1000) * targetFps * playbackRate;
    });
    window.addEventListener("resize", () => engine.resize());
    setupResizer();
  });
  
function setupResizer() { /* Logic cũ giữ nguyên */ let isMouseDown = false; function mouseMove(event: MouseEvent) { if (isMouseDown) leftPanel.style.flexBasis = event.clientX + "px"; else end(); } function end() { isMouseDown = false; engine.resize(); window.removeEventListener("mousemove", mouseMove); window.removeEventListener("mouseup", end); } resizer.addEventListener("mousedown", () => { isMouseDown = true; window.addEventListener("mousemove", mouseMove); window.addEventListener("mouseup", end); }); }
  onDestroy(() => { engine?.dispose(); blobUrls.forEach(url => URL.revokeObjectURL(url)); });
</script>

<main class="max-h-[100vh] relative overflow-hidden">
  {#if isLoading}
    <div class="absolute inset-0 z-50 flex items-center justify-center bg-black/80 text-white flex-col gap-4">
      <div class="text-2xl font-bold animate-pulse">Processing...</div>
      <div class="text-blue-300 font-mono text-sm">{loadingStatus}</div>
    </div>
  {/if}
  <div class="absolute bg-white/50 p-2 z-10 top-0 left-0">{fps.toFixed(0)} FPS</div>
  <button class="absolute bg-white/90 p-3 right-2 top-2 z-10 rounded-lg shadow-lg font-bold hover:bg-blue-100 transition-colors" on:click={loadFile} disabled={isLoading}>📂 Load Folder (Debug Mode)</button>
  
  <div class="fixed left-0 w-[100vw] bottom-12 flex justify-center z-10 pointer-events-none">
    <div class="w-[40rem] max-w-[90vw] p-5 bg-blue-200/70 rounded-xl border-gray-400 border shadow-md backdrop-blur-md grainy-bg flex flex-col justify-center pointer-events-auto">
      <div><FrameSlider {totalFrames} {currentFrame} frameRate={targetFps} /></div>
      <div class="flex justify-between mt-2 text-sm font-mono font-semibold text-gray-700">
        <div>{secondsToMMSS(currentSecond)}</div>
        <div>Frame {currentFrame + 1}/{totalFrames}</div>
      </div>
      <div class="flex gap-2 justify-between mt-3">
         <div class="flex gap-2 w-[15rem]">
             <button class="p-2 bg-white/50 rounded" on:click={() => { currentFpsSetting = (currentFpsSetting + 1) % fpsSettings.length; }}>{targetFps} FPS</button>
         </div>
         <button class="p-3 bg-blue-500/80 text-white rounded-full w-12 h-12 flex items-center justify-center shadow-sm" on:click={() => (isPlaying = !isPlaying)}>{#if isPlaying}<IconPlayerPauseFilled />{:else}<IconPlayerPlayFilled />{/if}</button>
         <div class="flex w-[15rem] justify-end gap-2"></div>
      </div>
    </div>
  </div>
  <section class="flex w-full h-[100dvh]">
    <div class="flex-grow-0 flex-shrink-0" bind:this={leftPanel} style={`flex-basis: 100%`}><canvas id="renderCanvas" class="touch-none h-full w-full outline-none" bind:this={canvas} /></div>
    <div class="bg-blue-300 cursor-col-resize flex-[0_0_0.5rem]" bind:this={resizer}></div>
  </section>
</main>
<style> :global(main > div) { position: relative; } </style>