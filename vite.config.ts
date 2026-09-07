import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { VitePWA } from "vite-plugin-pwa";
import { comlink } from "vite-plugin-comlink";
import { createReadStream, existsSync } from "fs";
import path from "path";

// Directories to expose as HTTP static assets (mesh files, textures, GT images)
const DATA_PREFIXES = [
  "/results_C4_final/",
  "/output_batch/",
  "/polyface_09042026/",
  "/eval_report/",
];

const MIME_MAP: Record<string, string> = {
  ".obj":  "text/plain; charset=utf-8",
  ".mtl":  "text/plain; charset=utf-8",
  ".png":  "image/png",
  ".jpg":  "image/jpeg",
  ".jpeg": "image/jpeg",
  ".npz":  "application/octet-stream",
  ".csv":  "text/csv; charset=utf-8",
};

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    comlink(),
    VitePWA({ registerType: "autoUpdate" }),
    svelte(),
    // Serve local data files (mesh, texture, GT photos) in dev mode
    {
      name: "serve-local-data",
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const url = (req.url ?? "").split("?")[0];
          const matched = DATA_PREFIXES.some((p) => url.startsWith(p));
          if (!matched) return next();

          const filePath = path.join(path.resolve(__dirname), url);
          if (!existsSync(filePath)) return next();

          const ext = path.extname(filePath).toLowerCase();
          res.setHeader("Content-Type", MIME_MAP[ext] ?? "application/octet-stream");
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.setHeader("Cache-Control", "public, max-age=3600");
          createReadStream(filePath).pipe(res);
        });
      },
    },
  ],
  build: {
    rollupOptions: {
      input: {
        main:  path.resolve(__dirname, "index.html"),
        eval:  path.resolve(__dirname, "eval.html"),
        align: path.resolve(__dirname, "align.html"),
      },
    },
  },
  worker: {
    plugins: () => [comlink()],
  },
});
