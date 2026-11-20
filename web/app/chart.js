// Draws the chart chosen by insights.chartSpec as inline SVG: horizontal bars
// for categories, a line for time. One series, so no legend box; the result
// table below the chart is the table view. Hover/focus shows a tooltip.

const SVG = "http://www.w3.org/2000/svg";
const BAR = 18; // bar thickness, under the 24px cap
const GAP = 8;

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

const fmt = (v) => (Math.abs(v) >= 1000 ? v.toLocaleString("en-US", { maximumFractionDigits: 1 }) : String(Math.round(v * 100) / 100));

function niceMax(v) {
  if (v <= 0) return 0;
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v);
}

function tooltip(container) {
  const tip = document.createElement("div");
  tip.className = "chart-tip";
  tip.hidden = true;
  container.appendChild(tip);
  return {
    show(label, value, name, x, y) {
      tip.replaceChildren();
      const strong = document.createElement("strong");
      strong.textContent = label;
      tip.append(strong, document.createTextNode(` ${name}: ${fmt(value)}`));
      tip.hidden = false;
      const box = container.getBoundingClientRect();
      tip.style.left = `${Math.min(x, box.width - tip.offsetWidth - 4)}px`;
      tip.style.top = `${Math.max(0, y - tip.offsetHeight - 8)}px`;
    },
    hide() { tip.hidden = true; },
  };
}

function barChart(container, spec) {
  const width = Math.max(320, container.clientWidth || 600);
  const labelW = Math.min(180, Math.max(60, ...spec.points.map((p) => p.label.length * 7.2 + 8)));
  const lo = Math.min(0, ...spec.points.map((p) => p.value));
  const hi = niceMax(Math.max(0, ...spec.points.map((p) => p.value))) || 1;
  const plotW = width - labelW - 16;
  const x = (v) => labelW + ((v - lo) / (hi - lo)) * plotW;
  const height = spec.points.length * (BAR + GAP) + 24;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img",
    "aria-label": `Bar chart of ${spec.valueName} by ${spec.labelName}` });
  // recessive grid: 0 and the max
  for (const v of [lo, 0, hi].filter((v, i, a) => a.indexOf(v) === i)) {
    el("line", { x1: x(v), x2: x(v), y1: 0, y2: height - 20, class: v === 0 ? "axis" : "grid" }, svg);
    el("text", { x: x(v), y: height - 6, class: "tick", "text-anchor": "middle" }, svg).textContent = fmt(v);
  }
  const tip = tooltip(container);
  spec.points.forEach((p, i) => {
    const y = i * (BAR + GAP);
    const label = el("text", { x: labelW - 8, y: y + BAR / 2 + 4, class: "label", "text-anchor": "end" }, svg);
    label.textContent = p.label.length > 24 ? `${p.label.slice(0, 23)}…` : p.label;
    const x0 = x(Math.min(0, p.value));
    const w = Math.max(1, Math.abs(x(p.value) - x(0)));
    // 4px rounded data end, square at the baseline
    const r = Math.min(4, w);
    const right = p.value >= 0;
    const d = right
      ? `M${x0},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${BAR - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`
      : `M${x0 + w},${y}h${-(w - r)}a${r},${r} 0 0 0 ${-r},${r}v${BAR - 2 * r}a${r},${r} 0 0 0 ${r},${r}h${w - r}z`;
    el("path", { d, class: "bar" }, svg);
    // hit target bigger than the mark: the whole row
    const hit = el("rect", { x: 0, y: y - GAP / 2, width, height: BAR + GAP, class: "hit", tabindex: 0,
      "aria-label": `${p.label}: ${fmt(p.value)}` }, svg);
    const show = () => tip.show(p.label, p.value, spec.valueName, x(p.value) * (container.clientWidth / width), (y / height) * svg.getBoundingClientRect().height);
    hit.addEventListener("mouseenter", show);
    hit.addEventListener("focus", show);
    hit.addEventListener("mouseleave", tip.hide);
    hit.addEventListener("blur", tip.hide);
  });
  container.prepend(svg);
}

function lineChart(container, spec) {
  const width = Math.max(320, container.clientWidth || 600);
  const height = 220;
  const pad = { l: 56, r: 16, t: 12, b: 28 };
  const lo = Math.min(0, ...spec.points.map((p) => p.value));
  const hi = niceMax(Math.max(0, ...spec.points.map((p) => p.value))) || 1;
  const n = spec.points.length;
  const x = (i) => pad.l + (n === 1 ? 0 : (i / (n - 1)) * (width - pad.l - pad.r));
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (height - pad.t - pad.b);
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img",
    "aria-label": `Line chart of ${spec.valueName} by ${spec.labelName}` });
  for (const v of [lo, (lo + hi) / 2, hi].filter((v, i, a) => a.indexOf(v) === i)) {
    el("line", { x1: pad.l, x2: width - pad.r, y1: y(v), y2: y(v), class: v === 0 ? "axis" : "grid" }, svg);
    el("text", { x: pad.l - 6, y: y(v) + 4, class: "tick", "text-anchor": "end" }, svg).textContent = fmt(v);
  }
  const step = Math.ceil(n / 8);
  spec.points.forEach((p, i) => {
    if (i % step === 0 || i === n - 1) {
      el("text", { x: x(i), y: height - 8, class: "tick", "text-anchor": "middle" }, svg).textContent = p.label;
    }
  });
  el("path", { d: spec.points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.value)}`).join(""), class: "line" }, svg);
  const marker = el("circle", { r: 5, class: "marker", visibility: "hidden" }, svg);
  const cross = el("line", { y1: pad.t, y2: height - pad.b, class: "cross", visibility: "hidden" }, svg);
  const tip = tooltip(container);
  const hit = el("rect", { x: pad.l, y: 0, width: width - pad.l - pad.r, height, class: "hit", tabindex: 0,
    "aria-label": spec.points.map((p) => `${p.label}: ${fmt(p.value)}`).join("; ") }, svg);
  let focusIndex = n - 1;
  const showAt = (i) => {
    const p = spec.points[i];
    for (const node of [marker, cross]) node.setAttribute("visibility", "visible");
    marker.setAttribute("cx", x(i));
    marker.setAttribute("cy", y(p.value));
    cross.setAttribute("x1", x(i));
    cross.setAttribute("x2", x(i));
    const scale = svg.getBoundingClientRect().width / width;
    tip.show(p.label, p.value, spec.valueName, x(i) * scale, y(p.value) * scale);
  };
  const hide = () => { for (const node of [marker, cross]) node.setAttribute("visibility", "hidden"); tip.hide(); };
  hit.addEventListener("mousemove", (e) => {
    const box = svg.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * width;
    showAt(Math.max(0, Math.min(n - 1, Math.round(((px - pad.l) / (width - pad.l - pad.r)) * (n - 1)))));
  });
  hit.addEventListener("mouseleave", hide);
  hit.addEventListener("focus", () => showAt(focusIndex));
  hit.addEventListener("blur", hide);
  hit.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      focusIndex = Math.max(0, Math.min(n - 1, focusIndex + (e.key === "ArrowRight" ? 1 : -1)));
      showAt(focusIndex);
      e.preventDefault();
    }
  });
  container.prepend(svg);
}

const drawn = new WeakMap(); // container -> {spec, width}
let observer = null;

function draw(container, spec) {
  container.replaceChildren();
  (spec.kind === "line" ? lineChart : barChart)(container, spec);
  drawn.set(container, { spec, width: container.clientWidth });
}

/**
 * Renders the spec into `container` (emptied first); null spec clears it.
 * The SVG is laid out in real pixels, so it is redrawn when the container's
 * width changes (window resized, phone rotated) instead of being scaled,
 * which would shrink its text.
 */
export function renderChart(container, spec) {
  container.replaceChildren();
  container.hidden = !spec;
  drawn.delete(container);
  if (!spec) return;
  draw(container, spec);
  if (!observer && "ResizeObserver" in window) {
    observer = new ResizeObserver((entries) => {
      for (const { target } of entries) {
        const last = drawn.get(target);
        if (last && Math.abs(target.clientWidth - last.width) > 24) draw(target, last.spec);
      }
    });
  }
  observer?.observe(container);
}
