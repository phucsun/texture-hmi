import "./app.css";
import AlignViewer from "./AlignViewer.svelte";

const app = new AlignViewer({ target: document.getElementById("app")! });
export default app;
