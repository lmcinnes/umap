// Evaluates how the docs parameter explorer updates a layout when a slider
// changes min_dist, against the baseline of a layout built from scratch.
//
//   npm install && npm run evaluate
//
// For each strategy it reports:
// - distance: how far the final layout is from the scratch layout at the
//   target value, after Procrustes alignment, as RMS distance relative to
//   the RMS spread of the scratch layout. "noise floor" is the same distance
//   between scratch layouts that differ only in their random seed.
// - knn: mean overlap of the 15 nearest neighbors in the final layout and in
//   the scratch layout (1 = same neighbors).
// - local scale: mean distance to the 15 nearest neighbors, relative to the
//   spread, divided by the same value for the scratch layout. min_dist
//   mainly changes this value, so 1 means that the new value took full effect.
// - data knn: mean overlap of the 15 nearest neighbors in the layout and in
//   the 4-D data, a measure of layout quality.
// - path, net: mean distance that a point travels on screen, summed over all
//   shown frames (path) and from start to end (net), relative to the spread.
//   path - net is motion that does not move a point toward its end position.
// - max step: the largest mean point motion in one shown frame.
// - hidden: epochs that run before the first shown frame (a delay).

import { readFile } from "node:fs/promises";

import {
  SCRATCH_SCHEDULE,
  WARM_START_SCHEDULE,
  alignTo,
  annealSchedule,
  splitCoordinates,
  transition,
} from "../_static/parameter_explorer/layout.js";
import { createUMAP } from "./node_modules/embedding-atlas/dist/umap.js";

const SEED = 42;
const N_NEIGHBORS = 15;
const NOISE_FLOOR_SEEDS = [1, 2, 3, 4];
const K = 15;

const rows = JSON.parse(await readFile(new URL("../_static/parameter_explorer/data.json", import.meta.url)));
const data = Float32Array.from(rows.flat());
const count = rows.length;
const inputDim = rows[0].length;

async function scratchModel(minDist, seed = SEED) {
  return createUMAP(count, inputDim, 2, data, { nNeighbors: N_NEIGHBORS, minDist, seed });
}

// Runs a schedule and returns the layout after each frame.
async function anneal(umap, schedule) {
  const frames = [];
  for (const learningRate of annealSchedule(schedule)) {
    umap.setParameters({ learningRate });
    await umap.step(schedule.epochsPerFrame);
    frames.push(splitCoordinates(umap.embedding));
  }
  return frames;
}

async function scratchLayout(minDist, seed = SEED) {
  const umap = await scratchModel(minDist, seed);
  const frames = await anneal(umap, SCRATCH_SCHEDULE);
  umap.destroy();
  return frames.at(-1);
}

function spread({ x, y }) {
  const mx = x.reduce((a, b) => a + b) / x.length;
  const my = y.reduce((a, b) => a + b) / y.length;
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += (x[i] - mx) ** 2 + (y[i] - my) ** 2;
  return Math.sqrt(sum / x.length);
}

function meanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.x.length; i++) sum += Math.hypot(a.x[i] - b.x[i], a.y[i] - b.y[i]);
  return sum / a.x.length;
}

function shapeDistance(layout, baseline) {
  const aligned = alignTo(layout, baseline);
  let sum = 0;
  for (let i = 0; i < count; i++) sum += (aligned.x[i] - baseline.x[i]) ** 2 + (aligned.y[i] - baseline.y[i]) ** 2;
  return Math.sqrt(sum / count) / spread(baseline);
}

function nearestNeighborsOf(distance) {
  return Array.from({ length: count }, (_, i) =>
    Array.from({ length: count }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => distance(i, a) - distance(i, b))
      .slice(0, K),
  );
}

const nearestNeighbors = ({ x, y }) => nearestNeighborsOf((i, j) => Math.hypot(x[j] - x[i], y[j] - y[i]));

const dataNeighbors = nearestNeighborsOf((i, j) => {
  let sum = 0;
  for (let d = 0; d < inputDim; d++) sum += (data[i * inputDim + d] - data[j * inputDim + d]) ** 2;
  return sum;
});

function localScale(layout) {
  const neighbors = nearestNeighbors(layout);
  let sum = 0;
  for (let i = 0; i < count; i++) {
    for (const j of neighbors[i]) sum += Math.hypot(layout.x[j] - layout.x[i], layout.y[j] - layout.y[i]);
  }
  return sum / (count * K) / spread(layout);
}

function neighborOverlap(layout, baselineNeighbors) {
  const neighbors = nearestNeighbors(layout);
  let shared = 0;
  for (let i = 0; i < count; i++) {
    const baseline = new Set(baselineNeighbors[i]);
    shared += neighbors[i].filter((j) => baseline.has(j)).length;
  }
  return shared / (count * K);
}

// Motion of the shown frames, starting from the layout shown before the change.
function motion(start, shownFrames) {
  const scale = spread(shownFrames.at(-1));
  let path = 0;
  let maxStep = 0;
  let previous = start;
  for (const frame of shownFrames) {
    const step = meanDistance(frame, previous) / scale;
    path += step;
    maxStep = Math.max(maxStep, step);
    previous = frame;
  }
  return { path, net: meanDistance(shownFrames.at(-1), start) / scale, maxStep, frames: shownFrames.length };
}

// How the computed frames of an update are shown. None of these changes the
// final layout.
const LIVE = { name: "live", show: (frames) => ({ shown: frames, hiddenFrames: 0 }) };
const PRE_ROLL = {
  name: "pre-roll half",
  show: (frames) => ({ shown: frames.slice(Math.floor(frames.length / 2)), hiddenFrames: Math.floor(frames.length / 2) }),
};
// What the explorer does: hide the optimization, then move the points.
const TRANSITION = {
  name: "transition",
  show: (frames, start) => ({ shown: transition(start, frames.at(-1)), hiddenFrames: frames.length }),
};

// The explorer's restart: a new model, aligned to the layout shown before.
async function restart(fromMinDist, toMinDist) {
  const start = await scratchLayout(fromMinDist);
  const umap = await scratchModel(toMinDist);
  const frames = (await anneal(umap, SCRATCH_SCHEDULE)).map((frame) => alignTo(frame, start));
  umap.destroy();
  return { start, frames };
}

// Continues the converged layout at `fromMinDist` with the new min_dist. For a
// drag, the slider passes `steps` values and each one gets `framesPerStep`
// frames before the next value restarts the schedule.
async function warmStart(fromMinDist, steps, schedule, framesPerStep = Infinity) {
  const umap = await scratchModel(fromMinDist);
  const start = (await anneal(umap, SCRATCH_SCHEDULE)).at(-1);
  const frames = [];
  for (const [index, minDist] of steps.entries()) {
    umap.setParameters({ minDist });
    const isLast = index === steps.length - 1;
    let shown = 0;
    for (const learningRate of annealSchedule(schedule)) {
      if (!isLast && shown === framesPerStep) break;
      umap.setParameters({ learningRate });
      await umap.step(schedule.epochsPerFrame);
      frames.push(splitCoordinates(umap.embedding));
      shown++;
    }
  }
  umap.destroy();
  return { start, frames };
}

const STRATEGIES = [
  { name: "warm lr 1, 500 ep", schedule: { initialLearningRate: 1, epochs: 500, epochsPerFrame: 10 } },
  { name: "warm lr 0.2, 100 ep", schedule: { initialLearningRate: 0.2, epochs: 100, epochsPerFrame: 10 } },
  ...[0.1, 0.05, 0.02].map((initialLearningRate) => ({
    name: `warm lr ${initialLearningRate}, 200 ep`,
    schedule: { initialLearningRate, epochs: 200, epochsPerFrame: 10 },
  })),
];

const TRANSITIONS = [
  [0.1, 0.0],
  [0.1, 0.25],
  [0.1, 0.5],
  [0.1, 0.99],
  [0.5, 0.1],
  [0.99, 0.1],
];

const format = (value) => value.toFixed(3);

function layoutReport(layout, baseline, baselineNeighbors) {
  return {
    distance: format(shapeDistance(layout, baseline)),
    knn: format(neighborOverlap(layout, baselineNeighbors)),
    "local scale": format(localScale(layout) / localScale(baseline)),
    "data knn": format(neighborOverlap(layout, dataNeighbors)),
  };
}

function report(name, result, baseline, baselineNeighbors, epochsPerFrame, displays = [LIVE]) {
  const layout = layoutReport(result.frames.at(-1), baseline, baselineNeighbors);
  return displays.map(({ name: display, show }) => {
    const { shown, hiddenFrames } = show(result.frames, result.start);
    const { path, net, maxStep } = motion(result.start, shown);
    return {
      strategy: `${name}, ${display}`,
      ...layout,
      path: format(path),
      net: format(net),
      "max step": format(maxStep),
      hidden: hiddenFrames * epochsPerFrame,
    };
  });
}

const average = (reports, key) => format(reports.reduce((sum, report) => sum + Number(report[key]), 0) / reports.length);

// The explorer during a drag: each update runs to the end, then the points
// move to the result and the next update uses the newest slider value.
async function explorerDrag(fromMinDist, updateValues) {
  const umap = await scratchModel(fromMinDist);
  const start = (await anneal(umap, SCRATCH_SCHEDULE)).at(-1);
  const frames = [];
  let shown = start;
  for (const minDist of updateValues) {
    umap.setParameters({ minDist });
    const target = (await anneal(umap, WARM_START_SCHEDULE)).at(-1);
    frames.push(...transition(shown, target));
    shown = target;
  }
  umap.destroy();
  return { start, frames };
}

async function evaluate(fromMinDist, toMinDist, steps, framesPerStep) {
  const baseline = await scratchLayout(toMinDist);
  const baselineNeighbors = nearestNeighbors(baseline);
  const noise = await Promise.all(NOISE_FLOOR_SEEDS.map((seed) => scratchLayout(toMinDist, seed)));
  const noiseReports = noise.map((layout) => layoutReport(layout, baseline, baselineNeighbors));
  const isDrag = steps.length > 1;
  const rows = [
    {
      strategy: "noise floor (scratch, other seeds)",
      ...Object.fromEntries(["distance", "knn", "local scale", "data knn"].map((key) => [key, average(noiseReports, key)])),
    },
    ...report("restart (scratch, aligned)", await restart(fromMinDist, toMinDist), baseline, baselineNeighbors, SCRATCH_SCHEDULE.epochsPerFrame),
  ];
  for (const { name, schedule } of STRATEGIES) {
    const result = await warmStart(fromMinDist, steps, schedule, framesPerStep);
    rows.push(...report(name, result, baseline, baselineNeighbors, schedule.epochsPerFrame, [LIVE, PRE_ROLL]));
  }
  const explorer = isDrag
    ? await explorerDrag(fromMinDist, steps.filter((_, index) => index % 5 === 4 || index === steps.length - 1))
    : await warmStart(fromMinDist, steps, WARM_START_SCHEDULE);
  const explorerDisplay = isDrag ? { ...LIVE, name: "transition after each update" } : TRANSITION;
  rows.push(...report("explorer", explorer, baseline, baselineNeighbors, WARM_START_SCHEDULE.epochsPerFrame, [explorerDisplay]));
  return rows;
}

for (const [from, to] of TRANSITIONS) {
  console.log(`\nmin_dist ${from} -> ${to}`);
  console.table(await evaluate(from, to, [to]));
}

const dragSteps = Array.from({ length: 20 }, (_, i) => Number((0.1 + (i + 1) * 0.02).toFixed(2)));
console.log("\nmin_dist 0.1 -> 0.5, dragged in steps of 0.02 with 2 frames per step");
console.table(await evaluate(0.1, 0.5, dragSteps, 2));
