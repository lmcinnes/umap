// Layout code of the parameter explorer that does not touch the page, so it
// can also run in Node, for example to evaluate the schedules below.

export function splitCoordinates(embedding) {
  const count = embedding.length / 2;
  const x = new Float32Array(count);
  const y = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    x[i] = embedding[2 * i];
    y[i] = embedding[2 * i + 1];
  }
  return { x, y };
}

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;

// Similarity transform (rotation, optional reflection, uniform scale and
// translation) that best maps `source` onto `target` in the least squares
// sense (2D orthogonal Procrustes analysis).
export function alignTo(source, target) {
  const count = source.x.length;
  const [sx, sy, tx, ty] = [source.x, source.y, target.x, target.y].map(mean);
  let dot = 0;
  let cross = 0;
  let mirroredDot = 0;
  let mirroredCross = 0;
  let sourceNorm = 0;
  for (let i = 0; i < count; i++) {
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

  const x = new Float32Array(count);
  const y = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const ax = source.x[i] - sx;
    const ay = mirror ? sy - source.y[i] : source.y[i] - sy;
    x[i] = tx + cos * ax - sin * ay;
    y[i] = ty + sin * ax + cos * ay;
  }
  return { x, y };
}

// UMAP's linear learning-rate decay, cut into frames of `epochsPerFrame`
// epochs. Yields the learning rate for each frame.
export function* annealSchedule({ initialLearningRate, epochs, epochsPerFrame }) {
  for (let epoch = 0; epoch < epochs; epoch += epochsPerFrame) {
    yield initialLearningRate * (1 - epoch / epochs);
  }
}

// A new layout, like `UMAP().fit` with the default schedule.
export const SCRATCH_SCHEDULE = { initialLearningRate: 1, epochs: 500, epochsPerFrame: 10 };

// Continues a converged layout after a min_dist change. An evaluation (see
// https://github.com/lmcinnes/umap/pull/1341) shows that this reaches layouts
// as close to a new layout as a new layout with another random seed. Learning rates
// of 0.05 and lower do not fully apply large decreases of min_dist. The
// explorer does not show these epochs, so they are a delay: keep them few.
export const WARM_START_SCHEDULE = { initialLearningRate: 0.2, epochs: 100, epochsPerFrame: 10 };

// After a warm start, points move to their new positions over this many
// frames, in a straight line. Showing the optimization itself makes the
// points jitter.
export const TRANSITION_FRAMES = 12;

const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

// Frames that move each point in a straight line from `from` to `to`.
export function transition(from, to, frameCount = TRANSITION_FRAMES) {
  return Array.from({ length: frameCount }, (_, frame) => {
    const t = easeInOut((frame + 1) / frameCount);
    return {
      x: from.x.map((value, i) => value + (to.x[i] - value) * t),
      y: from.y.map((value, i) => value + (to.y[i] - value) * t),
    };
  });
}
