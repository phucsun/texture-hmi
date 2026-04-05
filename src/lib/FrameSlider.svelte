<script lang="ts">
  import { secondsToMMSS } from "./utils";
  import { framePassed } from "./store";

  export let currentFrame = 0;

  export let totalFrames = 100;
  export let frameRate = 10;

  $: currentSecond = currentFrame / frameRate;
  $: totalSeconds = totalFrames / frameRate;

  let hoverPositionAsPercentage = null as number | null;
</script>

<!-- svelte-ignore a11y-no-static-element-interactions -->
<!-- svelte-ignore a11y-click-events-have-key-events -->
<div
  class="slider cursor-pointer"
  on:mousemove={(e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const width = rect.width;
    hoverPositionAsPercentage = x / width;
  }}
  on:mouseleave={() => {
    hoverPositionAsPercentage = null;
  }}
  on:click={() => {
    $framePassed = (hoverPositionAsPercentage ?? 0) * totalFrames;
  }}
>
  {#if hoverPositionAsPercentage != null && hoverPositionAsPercentage >= 0 && hoverPositionAsPercentage <= 1}
    <div
      class="absolute -top-[3rem] p-2 bg-white border border-gray-300 rounded-lg shadow-md w-max"
      style={`left: calc(${hoverPositionAsPercentage} * 100% - 1.5rem)`}
    >
      {secondsToMMSS(hoverPositionAsPercentage * totalSeconds)} / Frame {Math.round(
        hoverPositionAsPercentage * totalFrames,
      )}
      <div
        class="absolute"
        style="
        border-left: 1rem solid transparent;
        border-right: 1rem solid transparent;
        border-top: 1rem solid white;"
      ></div>
    </div>
  {/if}
  <div
    class="progress bg-blue-800 h-full"
    style="--current-time: {currentSecond}; --duration: {totalSeconds}"
  />
</div>

<style>
  .slider {
    width: 100%;
    height: 10px;
    background-color: #ccc;
    position: relative;
  }

  .progress {
    width: calc((var(--current-time) / var(--duration)) * 100%);
  }
</style>
