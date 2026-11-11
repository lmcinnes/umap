// Interactive UMAP parameter explorer for the "Basic UMAP Parameters" page.
//
// Each element with the class `umap-parameter-explorer` becomes a scatter plot
// with a slider for one UMAP parameter. The embedding is computed in the
// browser with the WebAssembly UMAP port from embedding-atlas and drawn with
// its EmbeddingView component. The component needs WebGPU. Only with it does
// the page show the explorers and their notes in place of the static figures.

import { EmbeddingView } from "embedding-atlas/component.js";
import { createUMAP } from "embedding-atlas/umap.js";

const POINT_COUNT = 800;
const INPUT_DIM = 4;
const SEED = 42;
const TOTAL_EPOCHS = 500;
const EPOCHS_PER_FRAME = 10;
// A converged layout only needs a gentle push when min_dist changes. A high
// learning rate would scatter it and make it look like a restart.
const WARM_START_LEARNING_RATE = 0.2;
const VIEW_MARGIN = 0.9;
const COLOR_LEVELS = 6;
const VIEW_HEIGHT = 420;

const DEFAULT_PARAMETERS = { n_neighbors: 15, min_dist: 0.1 };

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Random points in the 4-dimensional unit cube, read as (R, G, B, a) colors.
function randomColorCube() {
  const random = mulberry32(SEED);
  return Float32Array.from({ length: POINT_COUNT * INPUT_DIM }, random);
}

// EmbeddingView colors points by category, so quantize RGB into a palette.
function colorCategories(data) {
  const quantize = (value) => Math.min(COLOR_LEVELS - 1, Math.floor(value * COLOR_LEVELS));
  const category = new Uint8Array(POINT_COUNT);
  for (let i = 0; i < POINT_COUNT; i++) {
    const [r, g, b] = data.subarray(i * INPUT_DIM, i * INPUT_DIM + 3).map(quantize);
    category[i] = (r * COLOR_LEVELS + g) * COLOR_LEVELS + b;
  }

  const toChannel = (level) => Math.round(((level + 0.5) / COLOR_LEVELS) * 255);
  const categoryColors = Array.from({ length: COLOR_LEVELS ** 3 }, (_, index) => {
    const r = Math.floor(index / COLOR_LEVELS ** 2);
    const g = Math.floor(index / COLOR_LEVELS) % COLOR_LEVELS;
    const b = index % COLOR_LEVELS;
    return `rgb(${toChannel(r)}, ${toChannel(g)}, ${toChannel(b)})`;
  });

  return { category, categoryColors };
}

function splitCoordinates(embedding) {
  const x = new Float32Array(POINT_COUNT);
  const y = new Float32Array(POINT_COUNT);
  for (let i = 0; i < POINT_COUNT; i++) {
    x[i] = embedding[2 * i];
    y[i] = embedding[2 * i + 1];
  }
  return { x, y };
}

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;

// Similarity transform (rotation, optional reflection, uniform scale and
// translation) that best maps `source` onto `target` in the least squares
// sense (2D orthogonal Procrustes analysis).
function alignTo(source, target) {
  const [sx, sy, tx, ty] = [source.x, source.y, target.x, target.y].map(mean);
  let dot = 0;
  let cross = 0;
  let mirroredDot = 0;
  let mirroredCross = 0;
  let sourceNorm = 0;
  for (let i = 0; i < POINT_COUNT; i++) {
    const [ax, ay] = [source.x[i] - sx, source.y[i] - sy];
    const [bx, by] = [target.x[i] - tx, target.y[i] - ty];
    dot += ax * bx + ay * by;
    cross += ax * by - ay * bx;
    mirroredDot += ax * bx - ay * by;
    mirroredCross += ax * by + ay * bx;
    sourceNorm += ax * ax + ay * ay;
  }
  const mirror = Math.hypot(mirroredDot, mirroredCross) > Math.hypot(dot, cross);
  const [c, d] = mirror ? [mirroredDot, mirroredCross] : [dot, cross];
  const scale = Math.hypot(c, d) / Math.max(sourceNorm, 1e-12);
  const angle = Math.atan2(d, c);
  const [cos, sin] = [scale * Math.cos(angle), scale * Math.sin(angle)];

  const x = new Float32Array(POINT_COUNT);
  const y = new Float32Array(POINT_COUNT);
  for (let i = 0; i < POINT_COUNT; i++) {
    const ax = source.x[i] - sx;
    const ay = mirror ? sy - source.y[i] : source.y[i] - sy;
    x[i] = tx + cos * ax - sin * ay;
    y[i] = ty + sin * ax + cos * ay;
  }
  return { x, y };
}

// EmbeddingView maps the larger side of the view to [-1, 1] / scale data units.
function fitViewport({ x, y }, width, height) {
  const extent = (values) => values.reduce(([min, max], value) => [Math.min(min, value), Math.max(max, value)], [Infinity, -Infinity]);
  const [xMin, xMax] = extent(x);
  const [yMin, yMax] = extent(y);
  const largerSide = Math.max(width, height);
  const halfWidth = Math.max((xMax - xMin) / 2, 1e-6);
  const halfHeight = Math.max((yMax - yMin) / 2, 1e-6);
  return {
    x: (xMin + xMax) / 2,
    y: (yMin + yMax) / 2,
    scale: VIEW_MARGIN * Math.min(width / largerSide / halfWidth, height / largerSide / halfHeight),
  };
}

async function hasWebGPU() {
  try {
    return (await navigator.gpu?.requestAdapter()) != null;
  } catch {
    return false;
  }
}

function createSlider(parameter, values, initialValue) {
  const label = document.createElement("label");
  label.className = "umap-parameter-explorer__control";

  const name = document.createElement("code");
  name.textContent = parameter;

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = String(values.length - 1);
  slider.step = "1";
  slider.value = String(values.indexOf(initialValue));

  const output = document.createElement("output");
  output.textContent = String(initialValue);

  label.append(name, slider, output);
  return { label, slider, output };
}

function parameterValues(element) {
  const { min, max, step, values } = element.dataset;
  if (values) {
    return values.split(",").map(Number);
  }
  const stepSize = Number(step);
  const count = Math.round((Number(max) - Number(min)) / stepSize) + 1;
  const decimals = (step.split(".")[1] ?? "").length;
  return Array.from({ length: count }, (_, i) => Number((Number(min) + i * stepSize).toFixed(decimals)));
}

class ParameterExplorer {
  constructor(element, data, colors) {
    this.element = element;
    this.data = data;
    this.colors = colors;
    this.parameter = element.dataset.parameter;
    this.parameters = { ...DEFAULT_PARAMETERS };
    this.umap = null;
    this.modelNeighbors = null;
    this.generation = 0;
    this.updates = Promise.resolve();
    // The view follows the layout until the reader pans or zooms.
    this.followLayout = true;
    // A new model starts from a new spectral layout. Each of its frames is
    // aligned to the layout shown before, so the picture does not flip or turn.
    this.displayed = null;
    this.alignmentTarget = null;

    const values = parameterValues(element);
    const { label, slider, output } = createSlider(this.parameter, values, this.parameters[this.parameter]);
    slider.addEventListener("input", () => {
      const value = values[Number(slider.value)];
      output.textContent = String(value);
      this.setParameter(value);
    });

    const container = document.createElement("div");
    container.className = "umap-parameter-explorer__view";
    element.replaceChildren(label, container);

    this.width = container.clientWidth;
    this.view = new EmbeddingView(container, {
      data: { x: new Float32Array(POINT_COUNT), y: new Float32Array(POINT_COUNT), category: colors.category },
      categoryColors: colors.categoryColors,
      width: this.width,
      height: VIEW_HEIGHT,
      config: { colorScheme: "light", autoLabelEnabled: false },
      // The explorer controls the viewport. Each frame fits it to the layout
      // until the reader pans or zooms.
      viewportState: { x: 0, y: 0, scale: 1 },
      onViewportState: (viewportState) => {
        this.followLayout = false;
        this.view.update({ viewportState });
      },
    });
    new ResizeObserver(() => {
      this.width = container.clientWidth;
      this.view.update({ width: this.width });
    }).observe(container);
  }

  render() {
    const raw = splitCoordinates(this.umap.embedding);
    const coordinates = this.alignmentTarget ? alignTo(raw, this.alignmentTarget) : raw;
    this.displayed = coordinates;
    this.view.update({
      data: { ...coordinates, category: this.colors.category },
      ...(this.followLayout && { viewportState: fitViewport(coordinates, this.width, VIEW_HEIGHT) }),
    });
  }

  setParameter(value) {
    this.parameters[this.parameter] = value;
    this.followLayout = true;
    this.update();
  }

  // Updates run one after another. Each update brings the model in line with
  // the current parameters, so an update that a newer one supersedes is skipped.
  update() {
    const generation = ++this.generation;
    this.updates = this.updates
      .then(() => generation === this.generation && this.syncAndAnneal(generation))
      .catch((error) => console.error("UMAP parameter explorer:", error));
    return this.updates;
  }

  async syncAndAnneal(generation) {
    const { n_neighbors, min_dist } = this.parameters;
    if (this.umap === null || this.modelNeighbors !== n_neighbors) {
      // n_neighbors changes the fuzzy graph itself, so build a new model.
      this.umap?.destroy();
      this.alignmentTarget = this.displayed;
      this.umap = await createUMAP(POINT_COUNT, INPUT_DIM, 2, this.data, { nNeighbors: n_neighbors, minDist: min_dist, seed: SEED });
      this.modelNeighbors = n_neighbors;
      await this.anneal(generation, 1);
    } else {
      // min_dist only changes the low dimensional curve, so the current
      // layout is a valid start: re-optimize it in place.
      this.umap.setParameters({ minDist: min_dist });
      await this.anneal(generation, WARM_START_LEARNING_RATE);
    }
  }

  // Replays UMAP's linear learning-rate decay in small steps so that every
  // frame shows the layout as it converges. A newer update stops it.
  async anneal(generation, initialLearningRate) {
    const epochs = TOTAL_EPOCHS * initialLearningRate;
    for (let epoch = 0; epoch < epochs && generation === this.generation; epoch += EPOCHS_PER_FRAME) {
      this.umap.setParameters({ learningRate: initialLearningRate * (1 - epoch / epochs) });
      await this.umap.step(EPOCHS_PER_FRAME);
      this.render();
      await new Promise(requestAnimationFrame);
    }
  }
}

async function main() {
  const elements = document.querySelectorAll(".umap-parameter-explorer");
  if (elements.length === 0 || !(await hasWebGPU())) {
    return;
  }

  document.documentElement.classList.add("umap-parameter-explorer-enabled");

  const data = randomColorCube();
  const colors = colorCategories(data);

  for (const element of elements) {
    const explorer = new ParameterExplorer(element, data, colors);
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        observer.disconnect();
        explorer.update();
      }
    });
    observer.observe(element);
  }
}

main();
