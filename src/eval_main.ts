import "./app.css";
import EvalApp from "./EvalApp.svelte";

const app = new EvalApp({
  target: document.getElementById("app")!,
});

export default app;
