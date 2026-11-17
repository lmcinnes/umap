// Each `.umap-parameter-explorer` element becomes a scatter plot with a slider
// for one UMAP parameter. EmbeddingView needs WebGPU. Only with it does the
// page show the explorers and their notes, and collapse the static figures.

import { EmbeddingView } from "embedding-atlas/component.js";
import { createUMAP } from "embedding-atlas/umap.js";

import {
  SCRATCH_SCHEDULE,
  WARM_START_SCHEDULE,
  alignTo,
  annealSchedule,
  hasSettled,
  learningRateAt,
  smoothToward,
  splitCoordinates,
} from "./layout.js";

const POINT_COUNT = 800;
const INPUT_DIM = 4;
const SEED = 42;
const VIEW_MARGIN = 0.9;
const COLOR_LEVELS = 6;
const VIEW_HEIGHT = 420;

const DEFAULT_PARAMETERS = { n_neighbors: 15, min_dist: 0.1 };

// The data from the page: np.random.seed(42); np.random.rand(800, 4).
// Each point is read as an (R, G, B, a) color.
async function loadColorCube() {
  const response = await fetch(new URL("data.json", import.meta.url));
  if (!response.ok) {
    throw new Error(`Could not load the data: HTTP ${response.status}`);
  }
  const rows = await response.json();
  return Float32Array.from(rows.flat());
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

// EmbeddingView maps the larger side of the view to [-1, 1] / scale data units.
function fitViewport({ x, y }, width, height) {
  const extent = (values) => [Math.min(...values), Math.max(...values)];
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

// The same checks that the EmbeddingView renderer makes before it draws.
async function canRenderEmbeddingView() {
  if (navigator.gpu?.wgslLanguageFeatures == null) {
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter?.features.has("shader-f16") ?? false;
  } catch {
    return false;
  }
}

const nextFrame = () => new Promise(requestAnimationFrame);

const ENABLED_CLASS = "umap-parameter-explorer-enabled";

function setExplorersEnabled(enabled) {
  document.documentElement.classList.toggle(ENABLED_CLASS, enabled);
  for (const staticFigures of document.querySelectorAll(".umap-parameter-explorer-static")) {
    staticFigures.open = !enabled;
  }
}

function disableExplorers(error) {
  console.error("UMAP parameter explorer:", error);
  setExplorersEnabled(false);
}

const createElement = (tag, properties) => Object.assign(document.createElement(tag), properties);

function parameterValues({ min, max, step, values }) {
  if (values) {
    return values.split(",").map(Number);
  }
  const [start, end, stepSize] = [min, max, step].map(Number);
  const count = Math.round((end - start) / stepSize) + 1;
  const decimals = (step.split(".")[1] ?? "").length;
  return Array.from({ length: count }, (_, i) => Number((start + i * stepSize).toFixed(decimals)));
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
    // A layout that a newer update interrupted is not converged.
    this.converged = false;
    this.restartRequested = false;
    this.warmStartRunning = false;
    this.generation = 0;
    this.updates = Promise.resolve();
    this.followLayout = true;
    this.displayed = null;
    // A new model starts from a new spectral layout. Each of its frames is
    // aligned to the layout shown before, so the picture does not flip or turn.
    this.alignmentTarget = null;

    this.values = parameterValues(element.dataset);
    const initialValue = this.parameters[this.parameter];
    this.slider = createElement("input", { type: "range", max: this.values.length - 1, value: this.values.indexOf(initialValue) });
    this.slider.addEventListener("input", () => this.selectValue(this.values[this.slider.value]));
    this.output = createElement("output", { textContent: initialValue });
    const label = createElement("label", { className: "umap-parameter-explorer__control" });
    label.append(createElement("code", { textContent: this.parameter }), this.slider, this.output);

    const restartButton = createElement("button", {
      type: "button",
      className: "umap-parameter-explorer__restart",
      textContent: "Restart",
      title: "Watch UMAP optimize the embedding from its start layout",
    });
    restartButton.addEventListener("click", () => this.restart());

    const controls = createElement("div", { className: "umap-parameter-explorer__controls" });
    controls.append(label, restartButton);

    const container = document.createElement("div");
    element.replaceChildren(controls, container);

    this.width = container.clientWidth;
    this.view = new EmbeddingView(container, {
      data: { x: new Float32Array(POINT_COUNT), y: new Float32Array(POINT_COUNT), category: colors.category },
      categoryColors: colors.categoryColors,
      width: this.width,
      height: VIEW_HEIGHT,
      config: { colorScheme: "light", autoLabelEnabled: false },
      // A controlled viewport, so the view does not report its own default.
      viewportState: { x: 0, y: 0, scale: 1 },
      onViewportState: (viewportState) => {
        this.followLayout = false;
        this.view.update({ viewportState });
      },
    });
    new ResizeObserver(() => {
      this.width = container.clientWidth;
      this.view.update({ width: this.width, ...this.fittedViewport(this.displayed) });
    }).observe(container);
  }

  // The view follows the layout until the reader pans or zooms.
  fittedViewport(coordinates) {
    return this.followLayout && coordinates && { viewportState: fitViewport(coordinates, this.width, VIEW_HEIGHT) };
  }

  layout() {
    const raw = splitCoordinates(this.umap.embedding);
    return this.alignmentTarget ? alignTo(raw, this.alignmentTarget) : raw;
  }

  show(coordinates) {
    this.displayed = coordinates;
    this.view.update({ data: { ...coordinates, category: this.colors.category }, ...this.fittedViewport(coordinates) });
  }

  selectValue(value) {
    this.slider.value = this.values.indexOf(value);
    this.output.textContent = value;
    this.parameters[this.parameter] = value;
    this.followLayout = true;
    // A running warm start takes up min_dist changes itself.
    if (this.parameter !== "min_dist" || !this.warmStartRunning) {
      this.update();
    }
  }

  restart() {
    this.restartRequested = true;
    this.followLayout = true;
    this.update();
  }

  // Updates run one after another. Each update brings the model in line with
  // the current parameters, so an update that a newer one supersedes is skipped.
  update() {
    const generation = ++this.generation;
    this.updates = this.updates
      .then(() => generation === this.generation && this.syncAndAnneal(generation))
      .catch(disableExplorers);
  }

  async syncAndAnneal(generation) {
    const { n_neighbors, min_dist } = this.parameters;
    // The first layout and a restart show raw frames, so the reader can watch
    // UMAP converge.
    const smooth = this.displayed !== null && !this.restartRequested;
    if (this.modelNeighbors !== n_neighbors || this.restartRequested) {
      // n_neighbors changes the fuzzy graph itself, so build a new model.
      this.restartRequested = false;
      const umap = await createUMAP(POINT_COUNT, INPUT_DIM, 2, this.data, { nNeighbors: n_neighbors, minDist: min_dist, seed: SEED });
      this.umap?.destroy();
      this.umap = umap;
      this.modelNeighbors = n_neighbors;
      this.alignmentTarget = this.displayed;
    } else if (this.converged) {
      // min_dist only changes the low dimensional curve, so the converged
      // layout is a good start.
      return this.warmStart();
    } else {
      this.umap.setParameters({ minDist: min_dist });
    }
    await this.animate(generation, smooth);
  }

  // Smoothing filters out the large, noisy moves of the first epochs.
  async animate(generation, smooth) {
    this.converged = false;
    let shown = this.displayed;
    for (const learningRate of annealSchedule(SCRATCH_SCHEDULE)) {
      if (generation !== this.generation) {
        return;
      }
      this.umap.setParameters({ learningRate });
      await this.umap.step(SCRATCH_SCHEDULE.epochsPerFrame);
      shown = smooth ? smoothToward(shown, this.layout()) : this.layout();
      this.show(shown);
      await nextFrame();
    }
    this.converged = true;

    const target = this.layout();
    while (smooth && !hasSettled(shown, target)) {
      if (generation !== this.generation) {
        return;
      }
      shown = smoothToward(shown, target);
      this.show(shown);
      await nextFrame();
    }
    this.show(target);
  }

  // A min_dist change during the schedule restarts it with the new value, so
  // the figure always follows the newest slider value. The loop ends when the
  // shown points settle, or when a newer update needs a new model.
  async warmStart() {
    this.warmStartRunning = true;
    try {
      let shown = this.displayed;
      let minDist = null;
      let epoch = 0;
      while (!this.restartRequested && this.modelNeighbors === this.parameters.n_neighbors) {
        if (minDist !== this.parameters.min_dist) {
          minDist = this.parameters.min_dist;
          this.umap.setParameters({ minDist });
          epoch = 0;
        }
        if (epoch < WARM_START_SCHEDULE.epochs) {
          this.umap.setParameters({ learningRate: learningRateAt(WARM_START_SCHEDULE, epoch) });
          await this.umap.step(WARM_START_SCHEDULE.epochsPerFrame);
          epoch += WARM_START_SCHEDULE.epochsPerFrame;
        }
        const target = this.layout();
        shown = smoothToward(shown, target);
        this.show(shown);
        if (epoch >= WARM_START_SCHEDULE.epochs && hasSettled(shown, target)) {
          this.show(target);
          return;
        }
        await nextFrame();
      }
    } finally {
      this.warmStartRunning = false;
    }
  }
}

// Makes each mention of a slider value in the section, such as
// ``n_neighbors=2``, a button that shows this value in the explorer.
function linkValueMentions(explorer) {
  const section = explorer.element.closest("section");
  const pattern = new RegExp(`^${explorer.parameter}\\s*=\\s*([0-9.]+)$`);
  for (const literal of section?.querySelectorAll("code.literal") ?? []) {
    const value = Number(literal.textContent.trim().match(pattern)?.[1]);
    if (!explorer.values.includes(value)) {
      continue;
    }
    const button = createElement("button", {
      type: "button",
      className: "umap-parameter-explorer__value",
      title: "Show this value in the interactive figure",
    });
    literal.replaceWith(button);
    button.append(literal);
    button.addEventListener("click", () => {
      if (!document.documentElement.classList.contains(ENABLED_CLASS)) {
        return;
      }
      explorer.selectValue(value);
      explorer.element.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }
}

async function main() {
  const elements = document.querySelectorAll(".umap-parameter-explorer");
  if (elements.length === 0 || !(await canRenderEmbeddingView())) {
    return;
  }

  const data = await loadColorCube();
  const colors = colorCategories(data);
  const explorers = Array.from(elements, (element) => new ParameterExplorer(element, data, colors));

  // The explorers start when they scroll into view, which they can do only
  // once they are shown.
  setExplorersEnabled(true);
  for (const explorer of explorers) {
    linkValueMentions(explorer);
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        observer.disconnect();
        explorer.update();
      }
    });
    observer.observe(explorer.element);
  }
}

main().catch(disableExplorers);
