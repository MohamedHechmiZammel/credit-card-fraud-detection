// Plain-JS scorer for a native XGBoost JSON model (binary:logistic, numeric splits only).

export function loadModel(json) {
  const learner = json.learner;
  const p = parseFloat(learner.learner_model_param.base_score);
  const trees = learner.gradient_booster.model.trees.map((t) => ({
    left: Int32Array.from(t.left_children),
    right: Int32Array.from(t.right_children),
    feat: Int32Array.from(t.split_indices),
    cond: Float32Array.from(t.split_conditions), // stored as float32, as XGBoost does
    defLeft: Uint8Array.from(t.default_left),
  }));
  return { trees, baseMargin: Math.log(p / (1 - p)) };
}

function leafValue(t, x) {
  let n = 0;
  while (t.left[n] !== -1) {
    const v = x[t.feat[n]];
    if (Number.isNaN(v)) n = t.defLeft[n] ? t.left[n] : t.right[n];
    else n = Math.fround(v) < t.cond[n] ? t.left[n] : t.right[n];
  }
  return t.cond[n];
}

export function scoreRow(model, x) {
  let leafSum = 0;
  for (const t of model.trees) leafSum += leafValue(t, x);
  const margin = model.baseMargin + leafSum;
  return { margin, leafSum, prob: 1 / (1 + Math.exp(-margin)) };
}
