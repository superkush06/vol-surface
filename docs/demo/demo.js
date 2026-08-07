/* vol-surface, in the browser.
 *
 * Every number drawn here comes back from `volsurf` running under Pyodide.
 * Nothing is refitted in JavaScript: this file asks the library a question and
 * draws the answer, so a claim on the page is a claim about the library.
 */

const CSS = getComputedStyle(document.documentElement);
const C = (n) => CSS.getPropertyValue(n).trim();
const INK = C("--ink"), QUIET = C("--quiet"), HAIR = C("--hair"),
      CALM = C("--calm"), WARN = C("--warn"), GOLD = C("--gold");
const $ = (s) => document.querySelector(s);
const MONO = (px, w) => `${w || 400} ${px}px "IBM Plex Mono", Menlo, monospace`;

let py = null, runs = 0;
const tally = () => { const e = $("#runCount"); if (e) e.textContent = runs; };

/* Pyodide is single threaded, so everything queues rather than interleaves. */
let pyQueue = Promise.resolve();
function pyRun(code) {
  const job = pyQueue.then(() => py.runPythonAsync(code));
  pyQueue = job.catch(() => {});
  return job;
}
const pyJSON = async (e) => JSON.parse(await pyRun(`import json; json.dumps(${e})`));
const yieldToPaint = () =>
  document.hidden ? Promise.resolve() : new Promise(requestAnimationFrame);

function ctx(cv, h) {
  const w = cv.parentNode.clientWidth;
  const s = Math.min(2, window.devicePixelRatio || 1);
  if (cv._w !== w || cv._h !== h || cv._s !== s) {
    cv.style.height = h + "px";
    cv.width = Math.round(w * s); cv.height = Math.round(h * s);
    cv._w = w; cv._h = h; cv._s = s;
  }
  const c = cv.getContext("2d");
  c.setTransform(s, 0, 0, s, 0, 0);
  c.clearRect(0, 0, w, h);
  return { c, w, h };
}

/* A hand-built slider: colour is the label, and the readout is the value. */
function slider(el, readout, lo, hi, val, colour, onDone, fmt, live) {
  el.innerHTML = '<div class="trk"></div><div class="fil"></div><div class="knb"></div>';
  const fil = el.querySelector(".fil"), knb = el.querySelector(".knb");
  el.querySelector(".trk").style.background = colour;
  fil.style.background = colour; knb.style.background = colour; el.style.color = colour;
  const show = fmt || ((v) => v.toFixed(2));
  let v = val, down = false;
  const paint = () => {
    const t = (v - lo) / (hi - lo);
    knb.style.left = `calc(16px + ${t} * (100% - 32px))`;
    fil.style.width = `calc(16px + ${t} * (100% - 32px))`;
    readout.textContent = show(v);
    el.setAttribute("aria-valuemin", String(lo));
    el.setAttribute("aria-valuemax", String(hi));
    el.setAttribute("aria-valuenow", String(v));
    el.setAttribute("aria-valuetext", show(v));
  };
  const step = (hi - lo) / 400;
  const at = (cx) => {
    const r = el.getBoundingClientRect();
    const t = Math.min(1, Math.max(0, (cx - r.left - 16) / (r.width - 32)));
    const nv = Math.round((lo + t * (hi - lo)) / step) * step;
    if (Math.abs(nv - v) > 1e-12) { v = nv; paint(); if (live) onDone(v); }
  };
  el.addEventListener("pointerdown", (e) => {
    down = true; el.setPointerCapture(e.pointerId); at(e.clientX); e.preventDefault();
  });
  el.addEventListener("pointermove", (e) => { if (down) at(e.clientX); });
  el.addEventListener("pointerup", () => { if (down) { down = false; onDone(v); } });
  el.addEventListener("pointercancel", () => { down = false; });
  el.addEventListener("keydown", (e) => {
    const d = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 }[e.key];
    if (!d) return;
    v = Math.min(hi, Math.max(lo, v + d * step * 8));
    paint(); onDone(v); e.preventDefault();
  });
  paint();
  return { set: (nv) => { v = nv; paint(); } };
}

function watch(sel, go) {
  let done = false;
  new IntersectionObserver((es) => {
    if (es[0].isIntersecting && !done) { done = true; go(); }
  }, { rootMargin: "180px" }).observe($(sel));
}

/* ===== 00 · the inversion ================================================ */

let INV = null, SCR = null;
const IV = { sigma: 0.20, T: 1.0, noise: 25 };

function drawInvert() {
  const H = 300;
  const { c, w } = ctx($("#invert-fig"), H);
  if (!INV) return;
  const padL = 62, padR = 74, top = 26, bot = 40;
  const pw = w - padL - padR, ph = H - top - bot;
  const ks = INV.ks;
  const X = (k) => padL + (k - ks[0]) / (ks[ks.length - 1] - ks[0]) * pw;

  // price falls across strikes on the left axis
  const pmax = Math.max(...INV.price), pmin = Math.min(...INV.price);
  const YP = (v) => top + ph - (v - pmin) / ((pmax - pmin) || 1) * ph;
  // and the recovered vol is flat on the right, at a deliberately tight scale
  const band = 0.02;
  const YV = (v) => top + ph / 2 - (v - IV.sigma) / band * (ph / 2);

  c.font = MONO(9.5, 500); c.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const v = pmin + (pmax - pmin) * i / 4;
    c.strokeStyle = INK; c.globalAlpha = .08;
    c.beginPath(); c.moveTo(padL, YP(v) + .5); c.lineTo(padL + pw, YP(v) + .5); c.stroke();
    c.globalAlpha = 1; c.fillStyle = QUIET; c.textAlign = "right";
    c.fillText(v.toFixed(1), padL - 9, YP(v));
  }
  c.textAlign = "left"; c.fillStyle = GOLD;
  c.fillText("OPTION PRICE", padL - 54, 12);
  c.textAlign = "right"; c.fillStyle = CALM;
  c.fillText("VOL RECOVERED", w - padR + 62, 12);

  c.strokeStyle = GOLD; c.lineWidth = 2.2; c.beginPath();
  INV.price.forEach((v, i) => (i ? c.lineTo(X(ks[i]), YP(v)) : c.moveTo(X(ks[i]), YP(v))));
  c.stroke();

  // the vol that went in, and the vol that came back
  c.strokeStyle = INK; c.globalAlpha = .3; c.setLineDash([3, 3]);
  c.beginPath(); c.moveTo(padL, YV(IV.sigma)); c.lineTo(padL + pw, YV(IV.sigma)); c.stroke();
  c.setLineDash([]); c.globalAlpha = 1;
  c.strokeStyle = CALM; c.lineWidth = 2.6; c.beginPath();
  let st = false;
  INV.iv.forEach((v, i) => {
    if (v === null) { st = false; return; }
    st ? c.lineTo(X(ks[i]), YV(v)) : (c.moveTo(X(ks[i]), YV(v)), st = true);
  });
  c.stroke();
  c.fillStyle = QUIET; c.font = MONO(9.5, 500); c.textAlign = "left";
  c.fillText(`${(IV.sigma * 100).toFixed(1)}% IN`, padL + 6, YV(IV.sigma) - 11);
  c.textAlign = "right";
  c.fillText(`SCALE ±${(band * 100).toFixed(0)} VOL PTS`, w - padR + 62, YV(IV.sigma) + 13);

  // the discrete screen, marked where it fires
  if (SCR) {
    const bad = new Set(SCR.butterfly_idx);
    SCR.ks.forEach((k, i) => {
      if (k < ks[0] || k > ks[ks.length - 1]) return;
      c.fillStyle = bad.has(i) ? WARN : "rgba(53,50,44,.35)";
      c.beginPath(); c.arc(X(k), top + ph + 14, bad.has(i) ? 3.6 : 2, 0, 7); c.fill();
    });
    c.fillStyle = QUIET; c.textAlign = "left"; c.font = MONO(9, 500);
    c.fillText("RAW QUOTES SCREENED", padL, top + ph + 30);
  }
  c.fillStyle = QUIET; c.font = MONO(9.5, 500); c.textAlign = "center";
  [-0.4, -0.2, 0, 0.2, 0.4].forEach((k) => c.fillText(k.toFixed(1), X(k), H - 6));
}

async function computeInvert() {
  if (INV) return;
  await refreshInvert();
  slider($("#slSig"), $("#vSig"), 0.05, 0.80, IV.sigma, CALM,
         (v) => { IV.sigma = v; refreshInvert(); }, (v) => (v * 100).toFixed(1) + "%");
  slider($("#slTau"), $("#vTau"), 0.08, 3.0, IV.T, CALM,
         (v) => { IV.T = v; refreshInvert(); }, (v) => v.toFixed(2) + "y");
  slider($("#slScr"), $("#vScr"), 0, 800, IV.noise, WARN,
         (v) => { IV.noise = Math.round(v); refreshScreen(); },
         (v) => `${Math.round(v)} bp`);
}

async function refreshInvert() {
  $("#invNote").textContent = "pricing and inverting";
  INV = await pyJSON(`vs.price_curve(T=${IV.T}, sigma=${IV.sigma})`);
  const one = await pyJSON(`vs.price_and_invert(T=${IV.T}, sigma=${IV.sigma})`);
  runs += 1; tally();
  INV.one = one;
  if (!SCR) await refreshScreen(); else { drawInvert(); sayInvert(); }
}

async function refreshScreen() {
  SCR = await pyJSON(`vs.screen_quotes(noise_bps=${IV.noise})`);
  runs += 1; tally();
  drawInvert(); sayInvert();
}

function sayInvert() {
  if (!INV) return;
  const one = INV.one;
  const err = INV.worst_err;
  const clean = SCR && SCR.butterfly_idx.length === 0 && SCR.calendar_idx.length === 0;
  $("#invNote").textContent = `${INV.recovered} of ${INV.n} strikes inverted`;
  $("#invState").textContent = `worst error ${err === null ? "n/a" : err.toExponential(1)}`;
  const v = $("#invVerdict");
  v.classList.toggle("bad", !clean);
  $("#ivTag").textContent = clean ? "screen clean" : "screen fires";
  $("#ivMsg").innerHTML = clean
    ? `At <span class="n">${SCR ? SCR.noise_bps : 25} bp</span> of noise the raw quotes carry
       no butterfly or calendar violation, so they are worth fitting.`
    : `At <span class="n">${SCR.noise_bps} bp</span> the raw quotes already violate the
       butterfly condition at <span class="n">${SCR.butterfly_idx.length}</span> of
       <span class="n">${SCR.n}</span> strikes${SCR.calendar_idx.length
         ? `, and calendar at ${SCR.calendar_idx.length}` : ""}. No model has touched them
       yet.`;
  $("#sInvert").innerHTML =
    `Priced at <span class="n">${(IV.sigma * 100).toFixed(1)}%</span> and inverted back, the
     worst disagreement across all <span class="n">${INV.n}</span> strikes is
     <span class="n">${err === null ? "n/a" : err.toExponential(1)}</span>, which is solver
     tolerance rather than approximation. At the money the option is worth
     <span class="n">${one.price.toFixed(4)}</span> with delta
     <span class="n">${one.greeks.delta.toFixed(3)}</span> and vega
     <span class="n">${one.greeks.vega.toFixed(1)}</span>.
     The dots along the bottom are a different check: <span class="n">butterfly_violations</span>
     and <span class="n">calendar_violations</span> read strikes and vols directly, with no
     model in between, which is the test you run on a screen before deciding whether it is
     worth fitting at all. Push the noise up and watch them start firing, well before any
     fitter is involved.`;
}

/* ===== 01 · the surface ================================================== */

let SURF = null;
const SHAPE = { level: 1, skew: 1, wings: 1 };
let shapeH = {};

// Vol is the only quantity here, so it gets a single monotone ramp: paper
// through to ink through a warm mid. A two-hue scale would imply a midpoint
// that means something, and none does.
function volTint(t) {
  const stops = [[243, 239, 231], [214, 199, 170], [193, 148, 96],
                 [150, 90, 58], [83, 48, 40]];
  const x = Math.max(0, Math.min(0.999, t)) * (stops.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = stops[i], b = stops[i + 1];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},` +
         `${Math.round(a[1] + (b[1] - a[1]) * f)},` +
         `${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}

// The surface, as a surface. A heatmap is the honest 2D reading of this
// data, but the object every options desk pictures is the sheet itself, so
// this projects it: log-moneyness across, expiry back, implied vol up.
// Quads are shaded by height and drawn far-to-near, which is all the hidden
// surface removal a single-valued height field needs.
const VIEW = { yaw: -0.62, pitch: 0.62, spin: true };

// Raw orthographic projection into unscaled units. The fit to the canvas is
// worked out separately from the corners, so the sheet stays framed however
// far it is turned instead of walking off the edge.
function raw(x, y, z) {
  const cy = Math.cos(VIEW.yaw), sy = Math.sin(VIEW.yaw);
  const cp = Math.cos(VIEW.pitch), sp = Math.sin(VIEW.pitch);
  const rx = x * cy - y * sy;
  const ry = x * sy + y * cy;
  return { u: rx, v: ry * sp - z * cp, depth: ry * cp + z * sp };
}

let FIT2D = { s: 1, ox: 0, oy: 0 };
function project(x, y, z) {
  const r = raw(x, y, z);
  return { sx: FIT2D.ox + r.u * FIT2D.s, sy: FIT2D.oy + r.v * FIT2D.s,
           depth: r.depth };
}

function fitView(w, h, zmax, padX, padY) {
  let u0 = 1e9, u1 = -1e9, v0 = 1e9, v1 = -1e9;
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [0, zmax]) {
    const r = raw(x, y, z);
    u0 = Math.min(u0, r.u); u1 = Math.max(u1, r.u);
    v0 = Math.min(v0, r.v); v1 = Math.max(v1, r.v);
  }
  const s = Math.min((w - padX * 2) / (u1 - u0), (h - padY * 2) / (v1 - v0));
  FIT2D = { s, ox: w / 2 - (u0 + u1) / 2 * s, oy: h / 2 - (v0 + v1) / 2 * s };
}

function drawSurf() {
  const H = 540;
  const { c, w } = ctx($("#surf"), H);
  if (!SURF) {
    c.fillStyle = QUIET; c.font = MONO(11, 500);
    c.textAlign = "center"; c.textBaseline = "middle";
    c.fillText("fitting five expiries", w / 2, H / 2);
    return;
  }
  const { ks, ts, grid, vmin, vmax } = SURF;
  const nk = ks.length, nt = ts.length;
  const lt0 = Math.log(ts[0]), lt1 = Math.log(ts[nt - 1]);
  const NX = (i) => -1 + 2 * i / (nk - 1);
  const NY = (j) => -1 + 2 * (Math.log(ts[j]) - lt0) / (lt1 - lt0);
  const ZMAX = 0.80;
  const NZ = (v) => ZMAX * (v - vmin) / (vmax - vmin);
  fitView(w, H, ZMAX, 34, 22);

  // the floor, so the sheet has something to sit above
  c.strokeStyle = HAIR; c.lineWidth = 1;
  [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach((pt, i, arr) => {
    const a = project(pt[0], pt[1], 0);
    const b = project(arr[(i + 1) % 4][0], arr[(i + 1) % 4][1], 0);
    c.beginPath(); c.moveTo(a.sx, a.sy); c.lineTo(b.sx, b.sy); c.stroke();
  });

  const quads = [];
  for (let j = 0; j < nt - 1; j++) {
    for (let i = 0; i < nk - 1; i++) {
      const pts = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]].map(([a, b]) =>
        project(NX(a), NY(b), NZ(grid[b][a])));
      const mean = (grid[j][i] + grid[j][i + 1] + grid[j + 1][i] + grid[j + 1][i + 1]) / 4;
      quads.push({ pts, t: (mean - vmin) / (vmax - vmin),
                   d: (pts[0].depth + pts[2].depth) / 2 });
    }
  }
  quads.sort((a, b) => a.d - b.d);          // far first
  quads.forEach((q) => {
    c.fillStyle = volTint(q.t);
    c.strokeStyle = "rgba(53,50,44,.10)"; c.lineWidth = .5;
    c.beginPath();
    q.pts.forEach((p, i) => (i ? c.lineTo(p.sx, p.sy) : c.moveTo(p.sx, p.sy)));
    c.closePath(); c.fill(); c.stroke();
  });

  // the five ridges that were actually quoted
  c.lineWidth = 1.6;
  SURF.quoted_T.forEach((T, qi) => {
    let j = 0, best = 1e9;
    ts.forEach((t, jj) => { const d = Math.abs(Math.log(t) - Math.log(T));
                            if (d < best) { best = d; j = jj; } });
    c.strokeStyle = "#4A453C"; c.beginPath();
    for (let i = 0; i < nk; i++) {
      const p = project(NX(i), NY(j), NZ(grid[j][i]) + 0.004);
      i ? c.lineTo(p.sx, p.sy) : c.moveTo(p.sx, p.sy);
    }
    c.stroke();
    const e = project(NX(nk - 1), NY(j), NZ(grid[j][nk - 1]) + 0.004);
    c.fillStyle = INK; c.font = MONO(10, 500);
    c.textAlign = "left"; c.textBaseline = "middle";
    c.fillText(SURF.labels[qi], e.sx + 7, e.sy);
  });

  // axes, labelled where they end
  c.font = MONO(9.5, 500); c.fillStyle = QUIET;
  const kL = project(-1, -1, 0), kR = project(1, -1, 0);
  const tF = project(-1, 1, 0);
  c.textAlign = "center"; c.textBaseline = "middle";
  c.fillText("k = " + ks[0].toFixed(1), kL.sx - 4, kL.sy + 14);
  c.fillText("k = +" + ks[ks.length - 1].toFixed(1), kR.sx + 4, kR.sy + 14);
  c.fillText("LOG-MONEYNESS", (kL.sx + kR.sx) / 2, Math.max(kL.sy, kR.sy) + 30);
  c.textAlign = "right";
  c.fillText("MATURITY", tF.sx - 10, tF.sy);
  c.textAlign = "left";
  c.fillText("IMPLIED VOL " + (vmin * 100).toFixed(0) + "% to "
             + (vmax * 100).toFixed(0) + "%", 4, 14);
  c.textAlign = "right";
  c.fillText(VIEW.spin ? "DRAG TO TURN" : "DRAGGING", w - 4, 14);
}

// Drag to turn it; the idle spin stops the moment anyone does.
function surfControls() {
  const cv = $("#surf");
  let down = false, lx = 0, ly = 0;
  cv.style.cursor = "grab";
  cv.addEventListener("pointerdown", (e) => {
    down = true; VIEW.spin = false; lx = e.clientX; ly = e.clientY;
    cv.setPointerCapture(e.pointerId); cv.style.cursor = "grabbing";
    e.preventDefault();
  });
  cv.addEventListener("pointermove", (e) => {
    if (!down) return;
    VIEW.yaw += (e.clientX - lx) * 0.008;
    VIEW.pitch = Math.max(0.12, Math.min(1.35, VIEW.pitch + (e.clientY - ly) * 0.006));
    lx = e.clientX; ly = e.clientY;
    drawSurf();
  });
  const up = () => { down = false; cv.style.cursor = "grab"; };
  cv.addEventListener("pointerup", up);
  cv.addEventListener("pointercancel", up);
  // A bounded sway, not a full rotation. Turning all the way around takes the
  // sheet through angles where the ridges are edge-on and the expiry labels
  // pile up, so the first figure a reader meets is sometimes its worst view.
  // This keeps it moving without ever leaving a legible one.
  const yaw0 = VIEW.yaw;
  let phase = 0;
  (function spin() {
    if (VIEW.spin && SURF && !document.hidden) {
      phase += 0.0042;
      VIEW.yaw = yaw0 + 0.30 * Math.sin(phase);
      drawSurf();
    }
    requestAnimationFrame(spin);
  })();
}

async function computeSurf() {
  if (SURF) return;
  await refreshSurf();
  surfControls();
  const mk = (id, out, lo, hi, key) =>
    slider($(id), $(out), lo, hi, SHAPE[key], CALM,
           (v) => { SHAPE[key] = v; refreshSurf(); },
           (v) => v.toFixed(2) + "×");
  shapeH.level = mk("#slLev", "#vLev", 0.4, 2.0, "level");
  shapeH.skew = mk("#slSkw", "#vSkw", 0.0, 1.4, "skew");
  shapeH.wings = mk("#slWng", "#vWng", 0.4, 2.5, "wings");
}

// Each shape is five slices refitted from scratch, which is why this takes a
// beat: the sheet is the output of a calibration, not a parametric surface
// being redrawn.
async function refreshSurf() {
  $("#surfNote").textContent = "refitting five slices";
  const g = await pyJSON(`vs.surface_grid(nk=41, nt=31, level=${SHAPE.level}, ` +
                         `skew=${SHAPE.skew}, wings=${SHAPE.wings})`);
  runs += 5; tally();
  if (!g.ok) {
    $("#surfNote").textContent = "no admissible fit at this shape";
    $("#surfState").textContent = g.why || "";
    const v = $("#surfVerdict"); v.classList.add("bad");
    $("#svTag").textContent = "no fit";
    $("#svMsg").textContent =
      "The fitter could not land a slice at this shape, so the surface above is the last one that held.";
    return;
  }
  SURF = g;
  drawSurf();
  $("#surfNote").textContent = `${SURF.ts.length} expiries interpolated from 5`;
  $("#surfState").textContent =
    `vol ${(SURF.vmin * 100).toFixed(1)}% to ${(SURF.vmax * 100).toFixed(1)}%`;
  const clean = SURF.calendar_free && SURF.butterfly_free;
  const v = $("#surfVerdict");
  v.classList.toggle("bad", !clean);
  $("#svTag").textContent = clean ? "admissible" : "arbitrage";
  $("#svMsg").innerHTML = clean
    ? `Calendar and butterfly both clean. One-year ATM
       <span class="n">${((SURF.atm_1y || 0) * 100).toFixed(1)}%</span>, skew
       <span class="n">${((SURF.skew_1y || 0) * 100).toFixed(1)}</span> vol points across
       k = &plusmn;0.2.`
    : `This surface fails a no-arbitrage check: calendar
       ${SURF.calendar_free ? "clean" : "violated"}, butterfly
       ${SURF.butterfly_free ? "clean" : "violated"}. Steep enough wings or a high enough
       level and the fit stops being admissible.`;
  $("#sSurf").innerHTML =
    `Five slices are refitted every time you move a handle, and the
     <span class="n">${SURF.ts.length}</span> rows drawn come from those
     <span class="n">5</span>. <span class="n">Level</span> scales total variance,
     <span class="n">skew</span> leans rho toward the downside, and
     <span class="n">wings</span> steepens b. Push level or wings far enough and the
     butterfly check fails, which is the same test experiment one is about, applied to a
     whole surface instead of one slice.`;
}

/* ===== 01 · break it ===================================================== */

const P0 = { a: 0.0211, b: 0.075, rho: -0.60, m: 0.0, sigma: 0.20 };
let P = { ...P0 }, SL = null, handles = {};

function panel(c, x, y, w, h, ks, ys, opts) {
  const good = ys.filter((v) => v !== null && isFinite(v));
  if (!good.length) return;
  let lo = opts.lo !== undefined ? opts.lo : Math.min(...good);
  let hi = opts.hi !== undefined ? opts.hi : Math.max(...good);
  if (opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  // g and the density both peak far above where they dip, so a plain
  // min-to-max range renders the violation as a sliver at the bottom of the
  // panel. When there is one, cap the top so it occupies real estate
  // proportional to how badly the slice fails, and say the peak was clipped.
  let clipped = false;
  if (opts.zero && lo < 0 && hi > -lo * 3.2) { hi = -lo * 3.2; clipped = true; }
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const pad = (hi - lo) * .12; lo -= pad; hi += pad;
  const X = (k) => x + (k - ks[0]) / (ks[ks.length - 1] - ks[0]) * w;
  const Y = (v) => y + h - (v - lo) / (hi - lo) * h;

  c.strokeStyle = HAIR; c.lineWidth = 1;
  c.beginPath(); c.moveTo(x, y); c.lineTo(x, y + h); c.stroke();

  if (opts.zero) {                       // the line that decides admissibility
    const yz = Y(0);
    c.strokeStyle = INK; c.globalAlpha = .35; c.setLineDash([3, 3]);
    c.beginPath(); c.moveTo(x, yz); c.lineTo(x + w, yz); c.stroke();
    c.setLineDash([]); c.globalAlpha = 1;
    // everything below zero is the part that cannot be true
    c.fillStyle = WARN; c.globalAlpha = .16;
    c.beginPath(); let open = false;
    ks.forEach((k, i) => {
      const v = ys[i];
      if (v !== null && isFinite(v) && v < 0) {
        if (!open) { c.moveTo(X(k), yz); open = true; }
        c.lineTo(X(k), Y(v));
      } else if (open) { c.lineTo(X(ks[i - 1]), yz); c.closePath(); open = false; }
    });
    if (open) { c.lineTo(X(ks[ks.length - 1]), yz); c.closePath(); }
    c.fill(); c.globalAlpha = 1;
  }

  c.strokeStyle = opts.colour; c.lineWidth = 2; c.lineJoin = "round";
  c.beginPath(); let started = false;
  ks.forEach((k, i) => {
    const v = ys[i];
    if (v === null || !isFinite(v)) { started = false; return; }
    started ? c.lineTo(X(k), Y(v)) : (c.moveTo(X(k), Y(v)), started = true);
  });
  c.stroke();

  c.font = MONO(9.5, 500); c.textBaseline = "middle"; c.textAlign = "right";
  c.fillStyle = QUIET;
  c.fillText(opts.fmt(hi - pad), x - 8, Y(hi - pad) + 4);
  c.fillText(opts.fmt(lo + pad), x - 8, Y(lo + pad));
  c.textAlign = "left"; c.fillStyle = opts.colour;
  c.fillText(opts.title, x + 6, y + 11);
  if (clipped) {
    c.fillStyle = QUIET; c.textAlign = "right";
    c.fillText("PEAK CLIPPED", x + w - 4, y + 11);
  }
}

function drawSlice() {
  const H = 470;
  const { c, w } = ctx($("#slice"), H);
  if (!SL) return;
  const padL = 64, padR = 18, top = 18, gap = 22;
  const pw = w - padL - padR, ph = (H - top - 34 - gap * 2) / 3;
  const ks = SL.ks;

  panel(c, padL, top, pw, ph, ks, SL.iv,
        { colour: CALM, title: "IMPLIED VOL", fmt: (v) => (v * 100).toFixed(0) + "%" });
  panel(c, padL, top + ph + gap, pw, ph, ks, SL.g,
        { colour: SL.arb_free ? CALM : WARN, title: "g(k)  GATHERAL-JACQUIER",
          zero: true, fmt: (v) => v.toFixed(2) });
  panel(c, padL, top + (ph + gap) * 2, pw, ph, ks, SL.density,
        { colour: SL.arb_free ? CALM : WARN, title: "RISK-NEUTRAL DENSITY",
          zero: true, fmt: (v) => v.toFixed(2) });

  c.font = MONO(9.5, 500); c.fillStyle = QUIET;
  c.textAlign = "center"; c.textBaseline = "middle";
  const X = (k) => padL + (k - ks[0]) / (ks[ks.length - 1] - ks[0]) * pw;
  [-0.4, -0.2, 0, 0.2, 0.4].forEach((k) => c.fillText(k.toFixed(1), X(k), H - 20));
  c.fillText("LOG-MONEYNESS  k", padL + pw / 2, H - 6);
}

function sayBreak() {
  if (!SL) return;
  const v = $("#verdict");
  v.classList.toggle("bad", !SL.arb_free);
  if (!SL.defined) {
    $("#vTag").textContent = "undefined";
    $("#vMsg").textContent =
      "Total variance has gone non-positive, so this is not a smile at all.";
  } else if (SL.arb_free) {
    $("#vTag").textContent = "admissible";
    $("#vMsg").innerHTML =
      `g(k) bottoms out at <span class="n">${SL.min_g.toFixed(3)}</span>, above zero, so the ` +
      `slice is butterfly arbitrage free and the fitter would return it.`;
  } else {
    $("#vTag").textContent = "arbitrage";
    $("#vMsg").innerHTML =
      `g(k) reaches <span class="n">${SL.min_g.toFixed(3)}</span> at k = ` +
      `<span class="n">${SL.min_g_k.toFixed(2)}</span>. The fitter rejects this.`;
  }
  $("#sBreak").innerHTML = SL.arb_free
    ? `This slice is admissible. g(k) stays positive across the whole strip, the density is
       positive everywhere, and the quote is a coherent statement about where the stock can
       end up. Raise <span class="n">b</span> toward the top of its range and pull
       <span class="n">sigma</span> down, and it stops being one.`
    : `Between k = <span class="n">${SL.bad_from}</span> and
       <span class="n">${SL.bad_to}</span> the implied density is negative, which is
       <span class="n">${(SL.bad_frac * 100).toFixed(0)}%</span> of the strip. A negative
       density is not a modelling inconvenience: it means a butterfly spread struck across
       that region has a negative cost and a non-negative payoff, so somebody can be paid to
       take it. That is why <span class="n">fit_svi_slice</span> checks
       <span class="n">g(k)</span> before it returns, and why a surface that quietly fits the
       marks better is not automatically the one you want.`;
}

async function refreshSlice() {
  SL = await pyJSON(`vs.slice_curves(a=${P.a}, b=${P.b}, rho=${P.rho}, ` +
                    `m=${P.m}, sigma=${P.sigma})`);
  drawSlice(); sayBreak();
}

async function computeBreak() {
  if (SL) return;
  await refreshSlice();
  const mk = (id, out, lo, hi, key, fmt) =>
    slider($(id), $(out), lo, hi, P[key], CALM, (v) => { P[key] = v; refreshSlice(); },
           fmt, true);
  handles.a = mk("#slA", "#vA", -0.02, 0.08, "a", (v) => v.toFixed(3));
  handles.b = mk("#slB", "#vB", 0.02, 1.00, "b", (v) => v.toFixed(3));
  handles.rho = mk("#slR", "#vR", -0.98, 0.60, "rho", (v) => v.toFixed(2));
  handles.m = mk("#slM", "#vM", -0.25, 0.25, "m", (v) => v.toFixed(2));
  handles.sigma = mk("#slS", "#vS", 0.03, 0.45, "sigma", (v) => v.toFixed(3));
  $("#reset").addEventListener("click", () => {
    P = { ...P0 };
    Object.keys(handles).forEach((k) => handles[k].set(P[k]));
    refreshSlice();
  });
}

/* ===== 02 · calibrate ==================================================== */

let FIT = null, noiseBp = 25;

function drawResid() {
  const H = 320;
  const { c, w } = ctx($("#resid"), H);
  if (!FIT) return;
  const padL = 60, padR = 20, top = 26, bot = 40;
  const pw = w - padL - padR, ph = H - top - bot;
  const all = FIT.rows.flatMap((r) => r.quotes.concat(r.fitted));
  const lo = Math.min(...all) * 0.96, hi = Math.max(...all) * 1.03;
  const ks = FIT.rows[0].ks;
  const X = (k) => padL + (k - ks[0]) / (ks[ks.length - 1] - ks[0]) * pw;
  const Y = (v) => top + ph - (v - lo) / (hi - lo) * ph;

  c.font = MONO(9.5, 500); c.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4;
    c.strokeStyle = INK; c.globalAlpha = .09;
    c.beginPath(); c.moveTo(padL, Y(v) + .5); c.lineTo(padL + pw, Y(v) + .5); c.stroke();
    c.globalAlpha = 1; c.fillStyle = QUIET; c.textAlign = "right";
    c.fillText((v * 100).toFixed(0) + "%", padL - 9, Y(v));
  }
  c.textAlign = "left"; c.fillStyle = QUIET;
  c.fillText("IMPLIED VOL", padL - 52, 12);
  c.textAlign = "center";
  [-0.4, -0.2, 0, 0.2, 0.4].forEach((k) => c.fillText(k.toFixed(1), X(k), H - 22));
  c.fillText("LOG-MONEYNESS  k", padL + pw / 2, H - 8);

  FIT.rows.forEach((r, i) => {
    const t = FIT.rows.length === 1 ? 0 : i / (FIT.rows.length - 1);
    const col = volTint(0.25 + 0.6 * t);
    c.strokeStyle = col; c.lineWidth = 2; c.beginPath();
    r.fitted.forEach((v, j) => (j ? c.lineTo(X(r.ks[j]), Y(v)) : c.moveTo(X(r.ks[j]), Y(v))));
    c.stroke();
    c.fillStyle = col;
    r.quotes.forEach((v, j) => {
      c.beginPath(); c.arc(X(r.ks[j]), Y(v), 2.2, 0, 7); c.fill();
    });
    c.font = MONO(10, 500); c.textAlign = "left";
    c.fillText(`${r.label}  ${r.rms_bp.toFixed(1)}bp`,
               X(ks[ks.length - 1]) - 74, Y(r.fitted[r.fitted.length - 1]) - 10);
  });
}

function fitTable() {
  $("#fitBody").innerHTML = FIT.rows.map((r) =>
    `<tr><td>${r.label}</td><td>${r.a.toFixed(5)}</td><td>${r.b.toFixed(4)}</td>` +
    `<td>${r.rho.toFixed(4)}</td><td>${r.m.toFixed(4)}</td><td>${r.sigma.toFixed(4)}</td>` +
    `<td>${(r.atm * 100).toFixed(2)}%</td><td>${r.rms_bp.toFixed(1)}bp</td></tr>`).join("");
}

async function refreshFit() {
  $("#fitNote").textContent = "fitting five slices";
  FIT = await pyJSON(`vs.fit_surface(noise_bps=${noiseBp})`);
  runs += 5; tally();
  drawResid(); fitTable();
  $("#fitNote").textContent = `${FIT.rows.length} slices, 21 strikes each`;
  $("#fitState").textContent =
    `${FIT.best_bp.toFixed(1)} to ${FIT.worst_bp.toFixed(1)} bp`;
  const clean = FIT.calendar_free && FIT.butterfly_free;
  $("#sFit").innerHTML =
    `With <span class="n">${FIT.noise_bps.toFixed(0)} bp</span> of noise on every quote the
     fit lands within <span class="n">${FIT.best_bp.toFixed(1)}</span> to
     <span class="n">${FIT.worst_bp.toFixed(1)} bp</span> of the marks, expiry by expiry, and
     the surface comes back ${clean ? "free of both calendar and butterfly arbitrage"
       : "flagged: " + (FIT.calendar_free ? "" : "calendar ") +
         (FIT.butterfly_free ? "" : "butterfly ") + "arbitrage present"}.
     Those two facts are worth reading together. Fitting noisy marks more tightly is easy if
     you stop caring whether the result is admissible, and the check is what stops the fitter
     buying accuracy with coherence.`;
}

async function computeFit() {
  if (FIT) return;
  await refreshFit();
  slider($("#slN"), $("#vN"), 0, 100, noiseBp, GOLD,
         (v) => { noiseBp = Math.round(v); refreshFit(); },
         (v) => `${Math.round(v)} bp`);
}

/* ===== 03 · SABR ========================================================= */

let SB = null, SBP = null, SBC = null;
let sabrH = {};

function drawSabr() {
  const H = 300;
  const { c, w } = ctx($("#sabrfig"), H);
  if (!SB) return;
  const padL = 60, padR = 20, top = 26, bot = 40;
  const pw = w - padL - padR, ph = H - top - bot;
  // Scale to the quotes and the fitted lines only. A hand-set SABR can leave
  // the chart entirely, and rescaling for it would flatten everything else
  // into a line; it is clipped and labelled instead.
  const base = SB.quotes.concat(SB.svi, SB.sabr).filter((v) => v !== null);
  const lo = Math.min(...base) * 0.90, hi = Math.max(...base) * 1.12;
  const ks = SB.ks;
  const X = (k) => padL + (k - ks[0]) / (ks[ks.length - 1] - ks[0]) * pw;
  const Y = (v) => top + ph - (v - lo) / (hi - lo) * ph;
  const clamp = (y) => Math.max(top - 6, Math.min(top + ph + 6, y));

  c.font = MONO(9.5, 500); c.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4;
    c.strokeStyle = INK; c.globalAlpha = .09;
    c.beginPath(); c.moveTo(padL, Y(v) + .5); c.lineTo(padL + pw, Y(v) + .5); c.stroke();
    c.globalAlpha = 1; c.fillStyle = QUIET; c.textAlign = "right";
    c.fillText((v * 100).toFixed(0) + "%", padL - 9, Y(v));
  }
  c.textAlign = "left"; c.fillStyle = QUIET;
  c.fillText("IMPLIED VOL, ONE-YEAR SLICE", padL - 52, 12);
  c.textAlign = "center";
  [-0.4, -0.2, 0, 0.2, 0.4].forEach((k) => c.fillText(k.toFixed(1), X(k), H - 22));
  c.fillText("LOG-MONEYNESS  k", padL + pw / 2, H - 8);

  // the fitted SVI, for reference
  c.strokeStyle = CALM; c.lineWidth = 2.2; c.beginPath();
  SB.svi.forEach((v, j) => (j ? c.lineTo(X(ks[j]), Y(v)) : c.moveTo(X(ks[j]), Y(v))));
  c.stroke();
  c.fillStyle = CALM; c.font = MONO(11, 500); c.textAlign = "left";
  c.fillText(`SVI  ${SB.svi_bp.toFixed(1)}bp`,
             X(ks[ks.length - 1]) - 92, clamp(Y(SB.svi[SB.svi.length - 1]) - 12));

  // and SABR wherever the handles have it
  const cur = SBC || { ks: SB.ks, iv: SB.sabr, bp: SB.sabr_bp };
  let off = false;
  c.strokeStyle = GOLD; c.lineWidth = 2.4; c.beginPath();
  let started = false;
  cur.iv.forEach((v, j) => {
    if (v === null) { started = false; return; }
    const y = Y(v);
    if (y < top - 6 || y > top + ph + 6) off = true;
    started ? c.lineTo(X(cur.ks[j]), clamp(y)) : (c.moveTo(X(cur.ks[j]), clamp(y)), started = true);
  });
  c.stroke();
  c.fillStyle = GOLD; c.textAlign = "left";
  const lastY = clamp(Y(cur.iv[cur.iv.length - 1] || SB.sabr[SB.sabr.length - 1]));
  c.fillText(`SABR  ${cur.bp === null ? "n/a" : cur.bp.toFixed(1) + "bp"}`,
             X(ks[ks.length - 1]) - 100, lastY + 14);
  if (off) {
    c.fillStyle = WARN; c.font = MONO(9.5, 500); c.textAlign = "right";
    c.fillText("SABR CURVE CLIPPED", padL + pw - 4, top + 11);
  }

  c.fillStyle = INK;
  SB.quotes.forEach((v, j) => {
    c.beginPath(); c.arc(X(ks[j]), Y(v), 2.6, 0, 7); c.fill();
  });
  c.font = MONO(9.5, 500); c.textAlign = "left"; c.fillStyle = QUIET;
  c.fillText("DOTS ARE THE QUOTES", padL + 6, top + ph - 10);
}

async function computeSabr() {
  if (SB) return;
  $("#sabrNote").textContent = "fitting both models";
  SB = await pyJSON("vs.sabr_fit()");
  runs += 2; tally();
  SBP = { alpha: SB.alpha, beta: SB.beta, rho: SB.rho, nu: SB.nu };
  drawSabr(); saySabr();
  $("#sabrNote").textContent = "one-year slice, 21 strikes";

  const mk = (id, out, lo, hi, key, fmt) =>
    slider($(id), $(out), lo, hi, SBP[key], GOLD,
           (v) => { SBP[key] = v; refreshSabr(); }, fmt);
  sabrH.alpha = mk("#slAl", "#vAl", 0.3, 4.0, "alpha", (v) => v.toFixed(3));
  sabrH.beta = mk("#slBe", "#vBe", 0.0, 1.0, "beta", (v) => v.toFixed(2));
  sabrH.rho = mk("#slRh", "#vRh", -0.95, 0.95, "rho", (v) => v.toFixed(2));
  sabrH.nu = mk("#slNu", "#vNu", 0.05, 2.0, "nu", (v) => v.toFixed(3));
  $("#sabrReset").addEventListener("click", () => {
    SBP = { alpha: SB.alpha, beta: SB.beta, rho: SB.rho, nu: SB.nu };
    Object.keys(sabrH).forEach((k) => sabrH[k].set(SBP[k]));
    refreshSabr();
  });
}

async function refreshSabr() {
  SBC = await pyJSON(`vs.sabr_curve(${SBP.alpha}, ${SBP.beta}, ${SBP.rho}, ${SBP.nu})`);
  runs += 1; tally();
  drawSabr(); saySabr();
}

function saySabr() {
  const bp = SBC ? SBC.bp : SB.sabr_bp;
  const atFit = !SBC || Math.abs(bp - SB.sabr_bp) < 0.05;
  $("#sabrState").textContent =
    `SABR ${bp === null ? "n/a" : bp.toFixed(1) + "bp"} · SVI ${SB.svi_bp.toFixed(1)}bp`;
  $("#sSabr").innerHTML = atFit
    ? `Fitted, SABR lands at <span class="n">${SB.sabr_bp.toFixed(1)} bp</span> against SVI's
       <span class="n">${SB.svi_bp.toFixed(1)} bp</span>. That ordering is expected rather
       than a verdict: SVI has five free parameters shaping one slice, SABR has three here
       with beta pinned, and its smile is the output of a stochastic-volatility model rather
       than a shape chosen to fit. Move the handles and the fit is rescored against the same
       21 marks, so you can see how quickly it degrades away from the optimum.`
    : `At <span class="n">alpha ${SBP.alpha.toFixed(3)}</span>,
       <span class="n">beta ${SBP.beta.toFixed(2)}</span>,
       <span class="n">rho ${SBP.rho.toFixed(2)}</span>,
       <span class="n">nu ${SBP.nu.toFixed(3)}</span> the error against the same 21 marks is
       <span class="n">${bp === null ? "undefined" : bp.toFixed(1) + " bp"}</span>, against
       <span class="n">${SB.sabr_bp.toFixed(1)} bp</span> at the fitted parameters. Note how
       much of that comes from <span class="n">alpha</span> alone: it sets the level, and
       what it means depends on <span class="n">beta</span>, because alpha is the volatility
       of F to the beta rather than of F itself.`;
}

/* ===== chrome ============================================================ */

const redraw = () => {
  drawInvert(); drawSurf(); drawSlice(); drawResid(); drawSabr();
};
addEventListener("resize", redraw);
document.fonts.ready.then(redraw);

(function chrome() {
  const links = [...document.querySelectorAll("nav a")];
  const ids = links.map((a) => a.getAttribute("href").slice(1));
  const bar = $("#prog");
  const onScroll = () => {
    const h = document.body.scrollHeight - innerHeight;
    bar.style.width = (h > 0 ? Math.min(1, scrollY / h) * 100 : 0) + "%";
    let cur = 0;
    ids.forEach((id, i) => {
      const el = document.getElementById(id);
      if (el && el.getBoundingClientRect().top < innerHeight * .45) cur = i;
    });
    links.forEach((a, i) => a.classList.toggle("here", i === cur));
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
})();

(async function boot() {
  try {
    $("#surfNote").textContent = "loading python";
    py = await loadPyodide();
    // One always-fresh file names the bundle; the bundle is then fetched at a
    // URL that changes when its bytes do. A stale zip against fresh HTML is an
    // ImportError with nothing pointing at the cause.
    const fresh = { cache: "no-store" };
    const stamp = await (await fetch("bundle.json", fresh)).json();
    // The zip's hash cannot version vs.py, which is fetched separately and is
    // not in it. Carry the driver's own hash so editing it busts the cache.
    const v = `?v=${stamp.sha}${stamp.driver ? "-" + stamp.driver : ""}`;
    py.unpackArchive(await (await fetch("volsurf-pkg.zip" + v, fresh)).arrayBuffer(), "zip");
    py.FS.writeFile("vs.py", await (await fetch("vs.py" + v, fresh)).text());
    await pyRun("import sys; sys.path.insert(0,'.')\nimport vs");
    watch("#invert-fig", computeInvert);
    watch("#surf", computeSurf);
    watch("#slice", computeBreak);
    watch("#resid", computeFit);
    watch("#sabrfig", computeSabr);
  } catch (e) {
    $("#surfNote").textContent = "failed: " + e.message;
    console.error(e);
  }
})();
