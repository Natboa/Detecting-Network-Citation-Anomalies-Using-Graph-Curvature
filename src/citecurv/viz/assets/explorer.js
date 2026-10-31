/* Citation anomaly explorer: draws the JSON blob embedded next to this script.
   Every number comes from Python (citecurv.viz.precompute); this file only renders. No network access. */
(function () {
  "use strict";
  var ROOT_ID = "%%ROOT%%";
  var root = document.getElementById(ROOT_ID);
  if (!root) return;
  var D = JSON.parse(document.getElementById(ROOT_ID + "-data").textContent);
  var M = D.meta, COMP = M.components, NC = D.nodes;
  var SVGNS = "http://www.w3.org/2000/svg";
  var tipEl = root.querySelector(".cx-tip");
  var COMP_COLOR = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"];
  var DIV = ["var(--dn3)", "var(--dn2)", "var(--dn1)", "var(--d0)", "var(--dp1)", "var(--dp2)", "var(--dp3)"];
  var uid = 0;

  // ------------------------------------------------------------------ DOM helpers
  function setProps(e, p) {
    for (var k in p) {
      var v = p[k];
      if (v === undefined || v === null || v === false) continue;
      if (k === "text") e.textContent = v;
      else if (k.slice(0, 2) === "on" && typeof v === "function") e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : String(v));
    }
  }
  function addKids(e, kids) {
    kids.flat(4).forEach(function (k) {
      if (k === null || k === undefined || k === false) return;
      e.appendChild(k instanceof Node ? k : document.createTextNode(String(k)));
    });
  }
  function el(tag, props) { var e = document.createElement(tag); if (props) setProps(e, props); addKids(e, [].slice.call(arguments, 2)); return e; }
  function sv(tag, props) { var e = document.createElementNS(SVGNS, tag); if (props) setProps(e, props); addKids(e, [].slice.call(arguments, 2)); return e; }
  function nextId(p) { uid += 1; return ROOT_ID + "-" + p + uid; }

  // ------------------------------------------------------------------ formatting
  var NF = new Intl.NumberFormat("en-US");
  function fmtInt(x) { return x === null || x === undefined ? "n/a" : NF.format(x); }
  function fmt(x, d) {
    if (x === null || x === undefined || !isFinite(x)) return "n/a";
    var a = Math.abs(x);
    if (d === undefined) d = a >= 1000 ? 0 : a >= 100 ? 1 : a >= 10 ? 2 : 3;
    if (a !== 0 && a < 0.001) return x.toExponential(1).replace("-", "−");
    return x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }).replace("-", "−");
  }
  function fmtSigned(x, d) { return x === null || x === undefined || !isFinite(x) ? "n/a" : (x > 0 ? "+" : "") + fmt(x, d); }
  function fmtPct(x, d) { return x === null || x === undefined ? "n/a" : (x * 100).toFixed(d === undefined ? 1 : d) + "%"; }
  function day(t) { return new Date(t * 86400000).toISOString().slice(0, 10); }
  function compact(v) {
    var a = Math.abs(v);
    if (a >= 1e6) return +(v / 1e6).toFixed(1) + "M";
    if (a >= 1e4) return +(v / 1e3).toFixed(0) + "K";
    if (a >= 1 || a === 0) return NF.format(+v.toFixed(2));
    return +v.toPrecision(2) + "";
  }
  function short(s, n) { s = s || ""; return s.length > n ? s.slice(0, n - 1) + "…" : s; }
  function compName(c) { return (D.component_text[c] || {}).name || c; }

  // ------------------------------------------------------------------ papers
  function nodeId(i) { return NC.id[i]; }
  function nodeTitle(i) { return NC.title[i] || NC.id[i]; }
  function nodeField(i) { return NC.field[i] === null ? null : D.fields[NC.field[i]]; }
  function nodeVenue(i) { return NC.venue[i] === null ? null : D.venues[NC.venue[i]]; }
  var hasWorks = !D.missing.works;
  var hasLabels = !D.missing.labels;

  // ------------------------------------------------------------------ tooltip (values lead, labels follow)
  function showTip(c, x, y) {
    tipEl.textContent = "";
    if (c.head) tipEl.appendChild(el("div", { class: "cx-tip__head", text: c.head }));
    (c.rows || []).forEach(function (r) {
      tipEl.appendChild(el("div", { class: "cx-tip__row" },
        el("b", { text: r[1] }),
        el("span", { class: "cx-tip__key" }, r[2] ? el("span", { class: "cx-tip__line", style: "background:" + r[2] }) : null, r[0])));
    });
    tipEl.hidden = false;
    var b = tipEl.getBoundingClientRect();
    var left = x + 14, top = y + 14;
    if (left + b.width > window.innerWidth - 8) left = x - b.width - 14;
    if (top + b.height > window.innerHeight - 8) top = y - b.height - 14;
    tipEl.style.left = Math.max(8, left) + "px";
    tipEl.style.top = Math.max(8, top) + "px";
  }
  function hideTip() { tipEl.hidden = true; }

  /* Hover + keyboard for chart marks: arrows move between marks, Enter activates, Escape hides. */
  function bindMarks(svg, marks, label) {
    svg.setAttribute("tabindex", "0");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", label + (marks.length ? ". Use the arrow keys to read values." : ""));
    var cur = -1;
    function set(i, ev) {
      if (cur >= 0 && marks[cur] && marks[cur].hot) marks[cur].hot.classList.remove("is-hot");
      cur = i;
      if (i < 0) { hideTip(); return; }
      var m = marks[i];
      if (m.hot) m.hot.classList.add("is-hot");
      var x, y;
      if (ev && ev.clientX !== undefined) { x = ev.clientX; y = ev.clientY; }
      else { var b = (m.hot || m.hit).getBoundingClientRect(); x = b.left + b.width / 2; y = b.top; }
      showTip(m.tip(), x, y);
    }
    marks.forEach(function (m, i) {
      m.hit.addEventListener("pointermove", function (ev) { set(i, ev); });
      m.hit.addEventListener("pointerleave", function () { set(-1); });
      if (m.activate) {
        m.hit.style.cursor = "pointer";
        m.hit.addEventListener("click", m.activate);
      }
    });
    svg.addEventListener("keydown", function (ev) {
      if (!marks.length) return;
      var i = cur, k = ev.key;
      if (k === "ArrowRight" || k === "ArrowDown") i = Math.min(marks.length - 1, cur + 1);
      else if (k === "ArrowLeft" || k === "ArrowUp") i = Math.max(0, cur - 1);
      else if (k === "Home") i = 0;
      else if (k === "End") i = marks.length - 1;
      else if (k === "Escape") { set(-1); return; }
      else if ((k === "Enter" || k === " ") && cur >= 0 && marks[cur].activate) { ev.preventDefault(); marks[cur].activate(); return; }
      else return;
      ev.preventDefault();
      set(i);
    });
    svg.addEventListener("blur", function () { set(-1); });
  }

  // ------------------------------------------------------------------ scales and axes
  function niceTicks(lo, hi, n, integer) {
    if (!(hi > lo)) hi = lo + 1;
    var step0 = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(step0))), err = step0 / mag;
    var step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    if (integer) step = Math.max(1, Math.round(step));
    var out = [];
    for (var v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function lin(d0, d1, r0, r1) { var k = (r1 - r0) / ((d1 - d0) || 1); return function (v) { return r0 + (v - d0) * k; }; }
  function frame(box, w, h, m) {
    box.textContent = "";
    var svg = sv("svg", { width: w, height: h, viewBox: "0 0 " + w + " " + h });
    box.appendChild(svg);
    var g = sv("g", { transform: "translate(" + m.l + "," + m.t + ")" });
    svg.appendChild(g);
    return { svg: svg, g: g, iw: Math.max(10, w - m.l - m.r), ih: Math.max(10, h - m.t - m.b), m: m };
  }
  function yAxis(f, ys, vals, fm) {
    var grid = sv("g", { class: "grid" }), ax = sv("g", { class: "ax" });
    vals.forEach(function (v) {
      grid.appendChild(sv("line", { x1: 0, x2: f.iw, y1: ys(v), y2: ys(v) }));
      ax.appendChild(sv("text", { x: -6, y: ys(v), "text-anchor": "end", "dominant-baseline": "middle", text: fm(v) }));
    });
    f.g.appendChild(grid);
    f.g.appendChild(ax);
  }
  function xAxis(f, xs, vals, fm) {
    var ax = sv("g", { class: "ax" });
    ax.appendChild(sv("line", { x1: 0, x2: f.iw, y1: f.ih, y2: f.ih }));
    vals.forEach(function (v) {
      ax.appendChild(sv("line", { x1: xs(v), x2: xs(v), y1: f.ih, y2: f.ih + 4 }));
      ax.appendChild(sv("text", { x: xs(v), y: f.ih + 16, "text-anchor": "middle", text: fm(v) }));
    });
    f.g.appendChild(ax);
  }
  function axisTitles(f, xt, yt) {
    if (xt) f.g.appendChild(sv("text", { class: "lbl", x: f.iw / 2, y: f.ih + 32, "text-anchor": "middle", text: xt }));
    if (yt) f.g.appendChild(sv("text", { class: "lbl", x: -f.m.l + 2, y: -8, "text-anchor": "start", text: yt }));
  }
  function thinTicks(vals, iw, minGap) {
    var every = Math.max(1, Math.ceil(vals.length / Math.max(1, Math.floor(iw / minGap))));
    return vals.filter(function (_, i) { return i % every === 0; });
  }
  function colPath(x, y, w, h, r) {
    if (h <= 0.2 || w <= 0) return "";
    r = Math.min(r, w / 2, h);
    return "M" + x + "," + (y + h) + "V" + (y + r) + "Q" + x + "," + y + " " + (x + r) + "," + y + "H" + (x + w - r) +
      "Q" + (x + w) + "," + y + " " + (x + w) + "," + (y + r) + "V" + (y + h) + "Z";
  }
  function hbarPath(x0, x1, y, h, r) {  /* square at x0 (baseline), rounded at x1 (data end) */
    var w = Math.abs(x1 - x0);
    if (w < 0.2) return "";
    r = Math.min(r, w, h / 2);
    if (x1 >= x0) return "M" + x0 + "," + y + "H" + (x1 - r) + "Q" + x1 + "," + y + " " + x1 + "," + (y + r) + "V" + (y + h - r) +
      "Q" + x1 + "," + (y + h) + " " + (x1 - r) + "," + (y + h) + "H" + x0 + "Z";
    return "M" + x0 + "," + y + "H" + (x1 + r) + "Q" + x1 + "," + y + " " + x1 + "," + (y + r) + "V" + (y + h - r) +
      "Q" + x1 + "," + (y + h) + " " + (x1 + r) + "," + (y + h) + "H" + x0 + "Z";
  }
  /* classed diverging scale: br = [neutral, b1, b2] on |v|; 3 = neutral, 0..2 negative, 4..6 positive */
  function divClass(v, br) {
    if (v === null || v === undefined || !isFinite(v)) return -1;
    var a = Math.abs(v);
    if (a <= br[0]) return 3;
    var step = a <= br[1] ? 1 : a <= br[2] ? 2 : 3;
    return v < 0 ? 3 - step : 3 + step;
  }

  // ------------------------------------------------------------------ small building blocks
  function card(title, sub) {
    var head = el("div", { class: "cx-card__head" }, el("div", null, el("h2", { text: title }), sub ? el("p", { class: "cx-card__sub", text: sub }) : null));
    var c = el("section", { class: "cx-card" }, head);
    c.head = head;
    return c;
  }
  function tile(label, value, sub, small) {
    return el("div", { class: "cx-tile" }, el("div", { class: "cx-tile__label", text: label }),
      el("div", { class: "cx-tile__value" + (small ? " cx-tile__value--sm" : ""), text: value }),
      sub ? el("div", { class: "cx-tile__sub", text: sub }) : null);
  }
  function table(cols, rows, opts) {
    opts = opts || {};
    var thead = el("thead", null, el("tr", null, cols.map(function (c) { return el("th", { class: c.num ? "num" : null, scope: "col", text: c.label }); })));
    var tbody = el("tbody", null, rows.map(function (r) {
      return el("tr", null, r.map(function (v, j) {
        var td = el("td", { class: cols[j].num ? "num" : null });
        addKids(td, [v]);
        return td;
      }));
    }));
    var t = el("table", { class: opts.kv ? "cx-kv" : null }, opts.caption ? el("caption", { class: "sr", text: opts.caption }) : null, opts.kv ? null : thead, tbody);
    return el("div", { class: "cx-tablewrap" }, t);
  }
  function dataTable(summary, cols, rows) {
    return el("details", null, el("summary", { text: summary }), table(cols, rows, { caption: summary }));
  }
  function legend(items) {
    return el("div", { class: "cx-legend" }, items.map(function (it) {
      var sw = el("span", { class: "cx-swatch" + (it.kind ? " cx-swatch--" + it.kind : ""), style: it.kind === "ring" || it.kind === "outline" || it.kind === "box" ? "border-color:" + it.color : "background:" + it.color });
      return el("span", { class: "cx-legend__item" }, sw, it.label);
    }));
  }
  function seg(label, options, current, onPick) {
    var wrap = el("span", { class: "cx-seg", role: "group", "aria-label": label });
    options.forEach(function (o) {
      var b = el("button", { type: "button", "aria-pressed": String(o.value === current), disabled: o.disabled || null, title: o.title || null, text: o.label });
      b.addEventListener("click", function () {
        wrap.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", "false"); });
        b.setAttribute("aria-pressed", "true");
        onPick(o.value);
      });
      wrap.appendChild(b);
    });
    return el("span", { class: "cx-seglabel" }, el("span", { "aria-hidden": "true", text: label }), wrap);
  }
  function statusLine(ok, text) {
    return el("span", { class: "cx-status " + (ok ? "cx-status--pass" : "cx-status--fail") },
      el("span", { class: "cx-status__icon", "aria-hidden": "true", text: ok ? "✓" : "!" }), text);
  }
  function chip(kind, text) {
    return el("span", { class: "cx-chip cx-chip--" + kind }, kind === "pos" ? el("span", { class: "cx-dot", "aria-hidden": "true" }) : null, text);
  }
  function missingNote(key) { return el("p", { class: "cx-empty", text: D.missing[key] || "Not available." }); }

  // ------------------------------------------------------------------ chart registry (redraw on resize)
  var charts = {};
  function chart(panel, box, draw) {
    (charts[panel] = charts[panel] || []).push({ box: box, draw: draw, w: 0 });
    var w = box.clientWidth;
    if (w > 0) { draw(box, w); charts[panel][charts[panel].length - 1].w = w; }
    return box;
  }
  function redraw(panel, force) {
    (charts[panel] || []).forEach(function (c) {
      if (!c.box.isConnected) return;
      var w = c.box.clientWidth;
      if (w > 0 && (force || w !== c.w)) { c.w = w; c.draw(c.box, w); }
    });
  }

  // ------------------------------------------------------------------ charts
  function drawHistogram(box, width, o) {
    var H = o.hist, n = H.counts.length;
    var f = frame(box, width, o.height || 196, { l: 46, r: 14, t: 34, b: 40 });
    var lo = H.lo, hi = H.hi, bw = (hi - lo) / n;
    var xs = lin(lo, hi, 0, f.iw);
    var ymax = Math.max(1, Math.max.apply(null, H.counts));
    var yt = niceTicks(0, ymax, 4, true), ys = lin(0, yt[yt.length - 1], f.ih, 0);
    yAxis(f, ys, yt, compact);
    xAxis(f, xs, niceTicks(lo, hi, Math.max(2, Math.floor(f.iw / 70))), function (v) { return fmt(v, Math.abs(hi - lo) >= 20 ? 0 : 1); });
    axisTitles(f, o.xLabel, null);
    f.g.appendChild(sv("text", { class: "lbl", x: -f.m.l + 2, y: -22, text: o.yLabel || "events per bin" }));
    var band = f.iw / n, barw = Math.max(1, Math.min(24, band - 2)), marks = [];
    var color = o.color || "var(--total)";
    H.counts.forEach(function (c, i) {
      var x = xs(lo + i * bw) + (band - barw) / 2;
      var bar = sv("path", { class: "mark", d: colPath(x, ys(c), barw, f.ih - ys(c), Math.min(4, barw / 2)), style: "fill:" + color });
      var hit = sv("rect", { class: "hit", x: xs(lo + i * bw), y: 0, width: band, height: f.ih });
      f.g.appendChild(bar);
      f.g.appendChild(hit);
      marks.push({ hit: hit, hot: bar, tip: function () {
        return { head: fmt(lo + i * bw) + " to " + fmt(lo + (i + 1) * bw), rows: [["events", fmtInt(c)], ["share", fmtPct(c / H.n, 2)]] };
      } });
    });
    function vline(v, label, cls, anchorRight) {
      var clamped = Math.max(lo, Math.min(hi, v)), x = xs(clamped);
      f.g.appendChild(sv("line", { x1: x, x2: x, y1: -4, y2: f.ih, style: "stroke:var(--ink);stroke-width:1.5" }));
      var txt = (v > hi ? label + " → " : v < lo ? "← " + label + " " : label + " ") + fmt(v);
      var right = anchorRight !== undefined ? anchorRight : x > f.iw * 0.6;
      f.g.appendChild(sv("text", { class: cls || "lbl-ink", x: x + (right ? -4 : 4), y: -7, "text-anchor": right ? "end" : "start", text: txt }));
    }
    if (o.marker && o.marker.value !== null && o.marker.value !== undefined) vline(o.marker.value, o.marker.label);
    bindMarks(f.svg, marks, o.aria || "Histogram");
  }

  function drawCurves(box, width, o) {
    var w = Math.min(width, 460), h = Math.round(w * 0.78);
    var f = frame(box, w, h, { l: 44, r: 14, t: 22, b: 42 });
    var xs = lin(0, 1, 0, f.iw), ys = lin(0, 1, f.ih, 0), t5 = [0, 0.25, 0.5, 0.75, 1];
    yAxis(f, ys, t5, function (v) { return v.toFixed(2); });
    xAxis(f, xs, t5, function (v) { return v.toFixed(2); });
    axisTitles(f, o.xLabel, o.yLabel);
    if (o.ref) {
      var r = o.ref.diag ? [0, 0, 1, 1] : [0, o.ref.y, 1, o.ref.y];
      f.g.appendChild(sv("line", { x1: xs(r[0]), y1: ys(r[1]), x2: xs(r[2]), y2: ys(r[3]), style: "stroke:var(--axis);stroke-width:1" }));
      f.g.appendChild(sv("text", { class: "lbl", x: o.ref.diag ? xs(0.62) : xs(0) + 4, y: o.ref.diag ? ys(0.62) + 14 : ys(o.ref.y) - 5, "text-anchor": "start", text: o.ref.label }));
    }
    o.series.slice().reverse().forEach(function (s) {
      if (!s.pts.length) return;
      var d = s.pts.map(function (p, i) { return (i ? "L" : "M") + xs(p[0]).toFixed(1) + "," + ys(p[1]).toFixed(1); }).join("");
      f.g.appendChild(sv("path", { d: d, class: "mark", style: "fill:none;stroke:" + s.color + ";stroke-width:" + (s.main ? 2 : 1.5) + ";stroke-linejoin:round;stroke-linecap:round" + (s.main ? "" : ";opacity:.9") }));
    });
    var main = o.series[0];
    var cross = sv("line", { y1: 0, y2: f.ih, style: "stroke:var(--muted);stroke-width:1", visibility: "hidden" });
    var dot = sv("circle", { r: 4, style: "fill:" + main.color + ";stroke:var(--card);stroke-width:2", visibility: "hidden" });
    f.g.appendChild(cross);
    f.g.appendChild(dot);
    function nearest(pts, x) {
      var lo2 = 0, hi2 = pts.length - 1;
      while (hi2 - lo2 > 1) { var mid = (lo2 + hi2) >> 1; if (pts[mid][0] < x) lo2 = mid; else hi2 = mid; }
      return Math.abs(pts[lo2][0] - x) <= Math.abs(pts[hi2][0] - x) ? pts[lo2] : pts[hi2];
    }
    function at(x, cx, cy) {
      var p = nearest(main.pts, x);
      cross.setAttribute("x1", xs(p[0])); cross.setAttribute("x2", xs(p[0])); cross.setAttribute("visibility", "visible");
      dot.setAttribute("cx", xs(p[0])); dot.setAttribute("cy", ys(p[1])); dot.setAttribute("visibility", "visible");
      var rows = o.series.map(function (s) { return [s.name, s.pts.length ? fmt(nearest(s.pts, p[0])[1]) : "n/a", s.color]; });
      if (cx === undefined) { var b = dot.getBoundingClientRect(); cx = b.left; cy = b.top; }
      showTip({ head: o.xLabel + " " + fmt(p[0]), rows: rows }, cx, cy);
    }
    var overlay = sv("rect", { class: "hit", x: 0, y: 0, width: f.iw, height: f.ih });
    f.g.appendChild(overlay);
    overlay.addEventListener("pointermove", function (ev) {
      var b = overlay.getBoundingClientRect();
      at(Math.max(0, Math.min(1, (ev.clientX - b.left) / b.width)), ev.clientX, ev.clientY);
    });
    function off() { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); hideTip(); }
    overlay.addEventListener("pointerleave", off);
    f.svg.setAttribute("tabindex", "0");
    f.svg.setAttribute("role", "img");
    f.svg.setAttribute("aria-label", o.aria + ". Use the left and right arrow keys to move along the curve.");
    var kx = -0.02;
    f.svg.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
        ev.preventDefault();
        kx = Math.max(0, Math.min(1, kx + (ev.key === "ArrowRight" ? 0.02 : -0.02)));
        at(kx);
      } else if (ev.key === "Escape") off();
    });
    f.svg.addEventListener("blur", off);
  }

  // ------------------------------------------------------------------ app state, rows, tabs
  var R = D.rows;
  var rows = R.event_id.map(function (id, i) {
    var r = { i: i, id: id, src: R.src[i], dst: R.dst[i], t: R.t[i], total: R.total[i], rank: R.rank[i], cold: !!R.cold[i],
      label: R.label[i], z: COMP.map(function (c) { return R.z[c][i]; }), raw: COMP.map(function (c) { return R.raw[c][i]; }),
      detail: !!D.details[String(id)] };
    r.date = day(r.t);
    r.hay = [nodeId(r.src), nodeId(r.dst), NC.title[r.src] || "", NC.title[r.dst] || "", String(id)].join(" \u0001 ").toLowerCase();
    return r;
  });
  var rowById = {};
  rows.forEach(function (r) { rowById[r.id] = r; });
  var detailOrder = rows.filter(function (r) { return r.detail; }).sort(function (a, b) { return a.rank - b.rank; });

  var state = {
    tab: "overview", detailId: detailOrder.length ? detailOrder[0].id : null,
    f: { from: "", to: "", field: "", venue: "", min: "", cold: false, pos: false, q: "" },
    sort: { key: "order", dir: 1 }, page: 0, pageSize: 50,
    radius: Math.min(M.radius, 2), nodeColor: "role", edgeColor: "forman", heatMode: "share", compare: false,
    roll: "papers", rollSel: { papers: 0, authors: 0 }, rollPage: 0, cartel: 0, k: null,
  };
  var views = [
    { key: "overview", label: "Overview", build: buildOverview },
    { key: "list", label: "Ranked list", build: buildList },
    { key: "detail", label: "Citation detail", build: buildDetail },
    { key: "rollup", label: "Papers & authors", build: buildRollup },
  ];
  if (D.cartels) views.push({ key: "cartels", label: "Cartels (synthetic)", build: buildCartels });
  var tabsEl = root.querySelector(".cx-tabs"), panelsEl = root.querySelector('[data-slot="panels"]');
  var built = {};
  views.forEach(function (v) {
    v.tabId = nextId("tab-"); v.panelId = nextId("panel-");
    v.tab = el("button", { type: "button", class: "cx-tab", role: "tab", id: v.tabId, "aria-controls": v.panelId, "aria-selected": "false", tabindex: "-1", text: v.label });
    v.panel = el("section", { class: "cx-panel", role: "tabpanel", id: v.panelId, "aria-labelledby": v.tabId, hidden: true });
    v.tab.addEventListener("click", function () { activate(v.key); });
    tabsEl.appendChild(v.tab);
    panelsEl.appendChild(v.panel);
  });
  tabsEl.addEventListener("keydown", function (ev) {
    var i = views.findIndex(function (v) { return v.key === state.tab; }), j = i;
    if (ev.key === "ArrowRight") j = (i + 1) % views.length;
    else if (ev.key === "ArrowLeft") j = (i - 1 + views.length) % views.length;
    else if (ev.key === "Home") j = 0;
    else if (ev.key === "End") j = views.length - 1;
    else return;
    ev.preventDefault();
    activate(views[j].key);
    views[j].tab.focus();
  });
  function activate(key, rebuild) {
    hideTip();
    state.tab = key;
    views.forEach(function (v) {
      var on = v.key === key;
      v.tab.setAttribute("aria-selected", String(on));
      v.tab.setAttribute("tabindex", on ? "0" : "-1");
      v.panel.hidden = !on;
      if (on && (!built[key] || rebuild)) {
        charts[key] = [];
        v.panel.textContent = "";
        v.build(v.panel);
        built[key] = true;
        redraw(key);  // charts created before their card was attached draw now
      } else if (on) {
        redraw(key);
      }
    });
  }
  function openDetail(id) {
    state.detailId = id;
    state.visited = true;
    activate("detail", true);
    var top = tabsEl.getBoundingClientRect().top;
    if (top < 0) window.scrollTo({ top: window.scrollY + top - 8 });
  }

  // ------------------------------------------------------------------ OVERVIEW
  function buildOverview(p) {
    var years = (new Date(M.t_max).getUTCFullYear() - new Date(M.t_min).getUTCFullYear()) + 1;
    p.appendChild(el("div", { class: "cx-tiles" },
      tile("Citation events", fmtInt(M.n_events), M.events_source === "events" ? "in the event stream" : "rows of the scores file (no event stream given)"),
      tile("Papers", fmtInt(M.n_papers), M.n_works !== null ? fmtInt(M.n_works) + " in the works table" : "endpoints of events"),
      tile("Time span", M.t_min.slice(0, 4) + "–" + M.t_max.slice(0, 4), M.t_min + " to " + M.t_max + " (" + years + " years)", true),
      tile("Cold-start share", fmtPct(M.cold_share), fmtInt(M.n_cold) + " of " + fmtInt(M.n_scores) + " score rows, not ranked"),
      tile("Ranked events", fmtInt(M.n_scored), "details for the top " + fmtInt(Object.keys(D.details).length)),
      tile("Flag threshold", fmt(M.threshold), "total of the top " + fmtPct(M.flag_share, 0) + " (" + fmtInt(M.n_flags) + " events)")));

    if (M.notes.length) {
      var nc = card("Read before interpreting");
      nc.appendChild(el("ul", { class: "cx-notes" }, M.notes.map(function (n) { return el("li", { text: n }); })));
      nc.style.marginTop = "16px";
      p.appendChild(nc);
    }

    var g = el("div", { class: "cx-grid cx-grid--2" });
    p.appendChild(g);
    // total score distribution
    var hc = card("Total score distribution", "All ranked events; the line marks the flag threshold (top " + fmtPct(M.flag_share, 0) + ").");
    var H = D.hist.total;
    if (H) {
      hc.appendChild(chart("overview", el("div", { class: "cx-chart" }), function (box, w) {
        drawHistogram(box, w, { hist: H, xLabel: "total score (weighted sum of z-scores)", marker: { value: M.threshold, label: "threshold" }, aria: "Histogram of total scores" });
      }));
      hc.appendChild(el("p", { class: "cx-caption", text: fmtInt(H.n) + " ranked events. Bins cover the 0.5th to 99.5th percentile; " + fmtInt(H.under) + " below and " + fmtInt(H.over) + " above that range are not drawn." }));
      hc.appendChild(dataTable("Data table", [{ label: "from" }, { label: "to" }, { label: "events", num: true }],
        H.counts.map(function (c, i) { var bw = (H.hi - H.lo) / H.counts.length; return [fmt(H.lo + i * bw), fmt(H.lo + (i + 1) * bw), fmtInt(c)]; })));
    } else hc.appendChild(el("p", { class: "cx-empty", text: "No ranked events." }));
    g.appendChild(hc);

    // ranking quality
    var qc = card("Ranking quality against labels", D.curves ? "Precision-recall and ROC of the total score over " + fmtInt(D.curves.n) + " labelled ranked events (" + fmtInt(D.curves.n_pos) + " positive)." : null);
    if (D.curves) buildCurves(qc); else qc.appendChild(D.missing.curves ? el("p", { class: "cx-empty", text: D.missing.curves }) : missingNote("labels"));
    g.appendChild(qc);

    // D4 heatmap
    var dc = card("Flag share by field and year (D4 fairness check)", "Where the top " + fmtPct(M.flag_share, 0) + " of ranked events fall. The check fails when one (field, year) cell holds more than " + fmtPct(0.5, 0) + " of all flags.");
    dc.style.marginTop = "16px";
    if (D.heatmap) buildHeatmap(dc); else dc.appendChild(D.missing.heatmap ? el("p", { class: "cx-empty", text: D.missing.heatmap }) : missingNote("works"));
    p.appendChild(dc);

    var g2 = el("div", { class: "cx-grid cx-grid--2" });
    p.appendChild(g2);
    var wc = card("Combiner inputs", "Weights and frozen z statistics behind every total (NFR10).");
    wc.appendChild(table([{ label: "component" }, { label: "weight w", num: true }, { label: "z mean", num: true }, { label: "z sd", num: true }],
      COMP.map(function (c, i) {
        var zs = M.zstats[c] || [null, null];
        return [el("span", { class: "cx-legend__item" }, el("span", { class: "cx-swatch", style: "background:" + COMP_COLOR[i] }), compName(c) + " (" + c + ")"), fmt(M.weights[c]), fmt(zs[0]), fmt(zs[1])];
      })));
    wc.appendChild(el("p", { class: "cx-caption", text: "Weights: " + (M.weights_source === "given" ? "from the run" : "not given, equal weights assumed") + ". z statistics: " + (M.zstats_source === "given" ? "frozen from the training period" : "not given, computed from this file") + ". Pipeline design: " + M.combiner_text }));
    g2.appendChild(wc);
    var mc = card("Run metrics", D.metrics.length ? "As passed to the explorer (for example ranking_metrics output)." : null);
    if (D.metrics.length) mc.appendChild(table([{ label: "metric" }, { label: "value" }], D.metrics.map(function (kv) { return [kv[0], kv[1] === null ? "n/a" : typeof kv[1] === "number" ? fmt(kv[1], Number.isInteger(kv[1]) ? 0 : 4) : String(kv[1])]; }), { kv: true }));
    else mc.appendChild(missingNote("metrics"));
    var run = M.run || {};
    mc.appendChild(el("p", { class: "cx-caption", text: "Run: model_variant " + (run.model_variant || "n/a") + ", seed " + (run.seed === undefined || run.seed === null ? "n/a" : run.seed) + ", config_hash " + (run.config_hash || "n/a") + "." }));
    g2.appendChild(mc);

    D.tables.forEach(function (t) {
      var tc = card(t.name);
      tc.style.marginTop = "16px";
      var numeric = t.columns.map(function (_, j) { return t.rows.every(function (r) { return r[j] === null || typeof r[j] === "number"; }); });
      tc.appendChild(table(t.columns.map(function (c, j) { return { label: c, num: numeric[j] }; }),
        t.rows.map(function (r) { return r.map(function (v) { return v === null ? "n/a" : typeof v === "number" ? (Number.isInteger(v) ? fmtInt(v) : fmt(v, 4)) : String(v); }); })));
      p.appendChild(tc);
    });
    if (!D.tables.length) { var tcm = card("Comparison and ablation tables"); tcm.style.marginTop = "16px"; tcm.appendChild(missingNote("tables")); p.appendChild(tcm); }

    var gone = Object.keys(D.missing).filter(function (k) { return D.missing[k] && ["tables", "metrics"].indexOf(k) < 0; });
    if (gone.length) {
      var mcard = card("Inputs not provided");
      mcard.style.marginTop = "16px";
      mcard.appendChild(el("ul", { class: "cx-notes" }, gone.map(function (k) { return el("li", { text: D.missing[k] }); })));
      p.appendChild(mcard);
    }
  }

  function buildCurves(qc) {
    var C = D.curves;
    var tiles = [tile("AUROC", fmt(C.total.auroc), "total score", true), tile("Average precision", fmt(C.total.ap), "prevalence " + fmtPct(C.n_pos / C.n, 2), true)];
    Object.keys(C.at).forEach(function (k) { tiles.push(tile("Precision@" + fmtInt(+k), fmt(C.at[k].p), C.at[k].found + " positives found", true)); });
    qc.appendChild(el("div", { class: "cx-tiles" }, tiles));
    var ctr = el("div", { class: "cx-controls", style: "margin-top:12px" });
    var cb = el("input", { type: "checkbox", id: nextId("cmp") });
    cb.checked = state.compare;
    ctr.appendChild(el("label", { for: cb.id, style: "color:var(--ink)" }, cb, "Compare ranking by each component alone"));
    qc.appendChild(ctr);
    var grid = el("div", { class: "cx-grid", style: "margin-top:4px;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr))" });
    var lg = el("div");
    qc.appendChild(grid);
    qc.appendChild(lg);
    function series(kind) {
      var s = [{ name: "total", color: "var(--total)", pts: C.total[kind], main: true }];
      if (state.compare) COMP.forEach(function (c, i) { s.push({ name: c, color: COMP_COLOR[i], pts: C.components[c][kind] }); });
      return s;
    }
    var pr = el("div", { class: "cx-chart" }), roc = el("div", { class: "cx-chart" });
    grid.appendChild(el("div", null, el("h3", { text: "Precision-recall" }), pr));
    grid.appendChild(el("div", null, el("h3", { text: "ROC" }), roc));
    function drawBoth() {
      chart("overview", pr, function (box, w) { drawCurves(box, w, { series: series("pr"), xLabel: "recall", yLabel: "precision", ref: { y: C.n_pos / C.n, label: "chance (prevalence)" }, aria: "Precision-recall curve" }); });
      chart("overview", roc, function (box, w) { drawCurves(box, w, { series: series("roc"), xLabel: "false positive rate", yLabel: "true positive rate", ref: { diag: true, label: "chance" }, aria: "ROC curve" }); });
      lg.textContent = "";
      if (state.compare) lg.appendChild(legend([{ color: "var(--total)", label: "total", kind: "line" }].concat(COMP.map(function (c, i) { return { color: COMP_COLOR[i], label: c + " alone", kind: "line" }; }))));
    }
    cb.addEventListener("change", function () { state.compare = cb.checked; charts.overview = charts.overview.filter(function (c) { return c.box !== pr && c.box !== roc; }); drawBoth(); });
    drawBoth();
    qc.appendChild(dataTable("Data table: AUROC and AP per scorer", [{ label: "scorer" }, { label: "AUROC", num: true }, { label: "AP", num: true }],
      [["total", fmt(C.total.auroc), fmt(C.total.ap)]].concat(COMP.map(function (c) { return [c + " alone", fmt(C.components[c].auroc), fmt(C.components[c].ap)]; }))));
    if (C.unlabelled) qc.appendChild(el("p", { class: "cx-caption", text: fmtInt(C.unlabelled) + " ranked events have no label row and are left out of these curves." }));
  }

  function buildHeatmap(dc) {
    var Hm = D.heatmap;
    var max = Hm.max_share;
    dc.head.appendChild(statusLine(Hm.passed, (Hm.passed ? "PASS" : "FAIL") + ": largest cell " + fmtPct(max) + " (" + Hm.max_cell[0] + ", " + Hm.max_cell[1] + ")"));
    var ctr = el("div", { class: "cx-controls" });
    var box = el("div", { class: "cx-chart" });
    var lg = el("div");
    ctr.appendChild(seg("Colour", [{ value: "share", label: "Share of all flags" }, { value: "rate", label: "Rate vs overall" }], state.heatMode, function (v) { state.heatMode = v; redraw("overview", true); }));
    dc.appendChild(ctr);
    dc.appendChild(el("div", { class: "cx-tablewrap" }, box));
    dc.appendChild(lg);
    var SHARE_BINS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5], RATE_BINS = [0.5, 1, 1.5, 2, 3, 5];
    function cls(v, bins) { if (v === null || v <= 0) return -1; for (var i = 0; i < bins.length; i++) if (v <= bins[i]) return i; return 6; }
    chart("overview", box, function (bx, width) {
      var nF = Hm.fields.length, nY = Hm.years.length;
      var lw = width < 520 ? 92 : 140;
      var cw = Math.max(22, Math.min(56, (width - lw - 12) / nY)), ch = 26;
      var w = lw + cw * nY + 12, hgt = 24 + ch * nF + 26;
      var f = frame(bx, w, hgt, { l: lw, r: 12, t: 22, b: 26 });
      var marks = [];
      Hm.fields.forEach(function (fname, i) {
        f.g.appendChild(sv("text", { class: "lbl-ink", x: -8, y: i * ch + ch / 2, "text-anchor": "end", "dominant-baseline": "middle", text: short(fname, lw < 120 ? 12 : 20) }));
        Hm.years.forEach(function (yr, j) {
          var share = Hm.share[i][j], rate = Hm.rate[i][j], ratio = rate === null ? null : rate / M.flag_share;
          var k = state.heatMode === "share" ? cls(share, SHARE_BINS) : cls(Hm.flags[i][j] ? ratio : 0, RATE_BINS);
          var x = j * cw, y = i * ch, over = share !== null && share > Hm.limit;
          var rect = sv("rect", { class: "mark", x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 3,
            style: k < 0 ? "fill:var(--card);stroke:var(--grid)" : "fill:var(--sq" + (k + 1) + ")" });
          f.g.appendChild(rect);
          if (over) {
            f.g.appendChild(sv("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 3, style: "fill:none;stroke:var(--crit);stroke-width:2.5" }));
            f.g.appendChild(sv("text", { x: x + cw - 5, y: y + 9, "text-anchor": "end", style: "fill:var(--crit);font-size:10px;font-weight:700", text: "!" }));
          }
          var label = state.heatMode === "share" ? (Hm.flags[i][j] ? Math.round(share * 100) + "" : "") : (Hm.flags[i][j] && ratio !== null ? ratio.toFixed(1) : "");
          if (label && cw >= 26) f.g.appendChild(sv("text", { x: x + cw / 2, y: y + ch / 2 + 1, "text-anchor": "middle", "dominant-baseline": "middle", style: "font-size:10.5px;font-family:var(--font);fill:" + (k < 0 ? "var(--muted)" : "var(--sqt" + (k + 1) + ")"), text: label }));
          marks.push({ hit: rect, hot: rect, tip: function () {
            return { head: fname + ", " + yr, rows: [["flags", fmtInt(Hm.flags[i][j])], ["share of all flags", fmtPct(share)], ["events in cell", fmtInt(Hm.events[i][j])],
              ["flag rate in cell", rate === null ? "n/a" : fmtPct(rate, 2)], ["vs overall " + fmtPct(M.flag_share, 0), ratio === null ? "n/a" : ratio.toFixed(2) + "×"]] };
          } });
        });
      });
      var every = Math.max(1, Math.ceil(34 / cw));
      Hm.years.forEach(function (yr, j) {
        if (j % every === 0) f.g.appendChild(sv("text", { class: "lbl", x: j * cw + cw / 2, y: -8, "text-anchor": "middle", text: String(yr) }));
      });
      f.g.appendChild(sv("text", { class: "lbl", x: 0, y: ch * nF + 18, text: state.heatMode === "share" ? "cell label: % of all " + fmtInt(Hm.n_flags) + " flags" : "cell label: flag rate in the cell / overall rate" }));
      bindMarks(f.svg, marks, "Heatmap of flag share by field and year");
      lg.textContent = "";
      var bins = state.heatMode === "share" ? SHARE_BINS : RATE_BINS;
      var names = state.heatMode === "share"
        ? ["≤ 2%", "2–5%", "5–10%", "10–20%", "20–35%", "35–50%", "> 50%"]
        : ["≤ 0.5×", "0.5–1×", "1–1.5×", "1.5–2×", "2–3×", "3–5×", "> 5×"];
      lg.appendChild(legend([{ color: "var(--axis)", label: "no flags", kind: "box" }].concat(names.map(function (n, i) { return { color: "var(--sq" + (i + 1) + ")", label: n }; }))
        .concat([{ color: "var(--crit)", label: "cell above 50% (D4 fail)", kind: "outline" }])));
      void bins;
    });
    dc.appendChild(el("p", { class: "cx-caption", text: "Field = field of the citing paper; year = year of t. " + fmtInt(Hm.n_flags) + " flags among " + fmtInt(Hm.n) + " ranked events (threshold " + fmt(Hm.threshold) + ")." }));
    var trows = [];
    Hm.fields.forEach(function (fname, i) { Hm.years.forEach(function (yr, j) { if (Hm.events[i][j]) trows.push([fname, String(yr), fmtInt(Hm.flags[i][j]), fmtInt(Hm.events[i][j]), fmtPct(Hm.share[i][j]), Hm.rate[i][j] === null ? "n/a" : fmtPct(Hm.rate[i][j], 2)]); }); });
    dc.appendChild(dataTable("Data table", [{ label: "field" }, { label: "year" }, { label: "flags", num: true }, { label: "events", num: true }, { label: "share of flags", num: true }, { label: "rate in cell", num: true }], trows));
  }

  // ------------------------------------------------------------------ RANKED LIST
  function filtered() {
    var F = state.f, q = F.q.trim().toLowerCase(), mn = F.min === "" ? null : +F.min;
    var out = rows.filter(function (r) {
      if (r.cold && !F.cold) return false;
      if (F.pos && r.label !== 1) return false;
      if (F.from && r.date < F.from) return false;
      if (F.to && r.date > F.to) return false;
      if (mn !== null && !(r.total >= mn)) return false;
      if (F.field !== "" && NC.field[r.src] !== +F.field && NC.field[r.dst] !== +F.field) return false;
      if (F.venue !== "" && NC.venue[r.src] !== +F.venue && NC.venue[r.dst] !== +F.venue) return false;
      if (q && r.hay.indexOf(q) < 0) return false;
      return true;
    });
    var k = state.sort.key, dir = state.sort.dir;
    function val(r) {
      if (k === "order") return r.i;
      if (k === "date") return r.t;
      if (k === "total") return r.total;
      if (k === "label") return r.label === null ? -1 : r.label;
      return r.z[+k.slice(1)];
    }
    out.sort(function (a, b) {
      var va = val(a), vb = val(b);
      if (va === null && vb === null) return a.i - b.i;
      if (va === null) return 1;
      if (vb === null) return -1;
      return (va - vb) * dir || a.i - b.i;
    });
    return out;
  }
  function zBar(z, color) {
    var w = 44, h = 12, c = w / 2, lim = 4;
    var x = z === null ? c : c + Math.max(-lim, Math.min(lim, z)) / lim * (c - 1);
    return sv("svg", { width: w, height: h, viewBox: "0 0 " + w + " " + h, "aria-hidden": "true" },
      sv("rect", { x: 0, y: h / 2 - 0.5, width: w, height: 1, style: "fill:var(--grid)" }),
      sv("rect", { x: c - 0.5, y: 0, width: 1, height: h, style: "fill:var(--axis)" }),
      z === null ? null : sv("path", { d: hbarPath(c, x, 2, h - 4, 2), style: "fill:" + color }));
  }
  function citeCell(src, dst) {
    return el("div", { class: "cx-cite" },
      el("div", { class: "cx-cite__line", title: nodeTitle(src) }, el("span", { class: "cx-cite__arrow", "aria-hidden": "true", text: "" }), el("span", { text: nodeTitle(src) })),
      el("div", { class: "cx-cite__line", title: nodeTitle(dst) }, el("span", { class: "cx-cite__arrow", text: "→" }), el("span", { text: nodeTitle(dst) })),
      el("div", { class: "cx-cite__ids", text: nodeId(src) + " → " + nodeId(dst) }));
  }
  function buildList(p) {
    var F = state.f;
    var fc = el("section", { class: "cx-card" });
    p.appendChild(fc);
    function field(label, input) { return el("label", { class: "cx-field" }, label, input); }
    function bind(input, key, evName) {
      input.value = F[key];
      input.addEventListener(evName || "input", function () { F[key] = input.value; state.page = 0; refresh(); });
      return input;
    }
    var grid = el("div", { class: "cx-filters" });
    grid.appendChild(field("Search titles and ids", bind(el("input", { type: "search", placeholder: "title, paper id, event id" }), "q")));
    grid.appendChild(field("From date", bind(el("input", { type: "date", min: M.t_min, max: M.t_max }), "from", "change")));
    grid.appendChild(field("To date", bind(el("input", { type: "date", min: M.t_min, max: M.t_max }), "to", "change")));
    if (hasWorks) {
      var fs = el("select", null, el("option", { value: "", text: "All fields" }), D.fields.map(function (f, i) { return el("option", { value: String(i), text: f }); }).sort(function (a, b) { return a.textContent.localeCompare(b.textContent); }));
      grid.appendChild(field("Field (either paper)", bind(fs, "field", "change")));
      var vs = el("select", null, el("option", { value: "", text: "All venues" }), D.venues.map(function (v, i) { return el("option", { value: String(i), text: v }); }).sort(function (a, b) { return a.textContent.localeCompare(b.textContent); }));
      grid.appendChild(field("Venue (either paper)", bind(vs, "venue", "change")));
    }
    grid.appendChild(field("Minimum total", bind(el("input", { type: "number", step: "0.1", inputmode: "decimal", placeholder: "any" }), "min")));
    function check(label, key, disabled, title) {
      var cb = el("input", { type: "checkbox", disabled: disabled || null });
      cb.checked = F[key];
      cb.addEventListener("change", function () { F[key] = cb.checked; state.page = 0; refresh(); });
      return el("label", { class: "cx-field cx-field--check", title: title || null }, cb, label);
    }
    grid.appendChild(check("Include cold-start events", "cold", false, "Cold-start events are excluded from scoring; they appear unranked."));
    grid.appendChild(check("Positives only", "pos", !hasLabels, hasLabels ? null : D.missing.labels));
    fc.appendChild(grid);

    var bar = el("div", { class: "cx-toolbar" });
    var count = el("span", { class: "muted", role: "status", "aria-live": "polite" });
    var reset = el("button", { type: "button", class: "cx-btn", text: "Reset filters" });
    var exp = el("button", { type: "button", class: "cx-btn cx-btn--primary", text: "Export CSV" });
    bar.appendChild(count);
    bar.appendChild(el("div", { class: "cx-toolbar__actions" }, reset, exp));
    fc.appendChild(bar);
    var wrap = el("div", { class: "cx-tablewrap" });
    fc.appendChild(wrap);
    var pager = el("div", { class: "cx-pager" });
    fc.appendChild(pager);
    fc.appendChild(el("p", { class: "cx-caption", text: "Rows: the top " + fmtInt(M.top_n) + " ranked events" + " plus cold-start events in the same score range (shown unranked). z bars span −4 to +4; positive = more anomalous than the reference. Rank opens the detail view." }));

    reset.addEventListener("click", function () {
      state.f = F = { from: "", to: "", field: "", venue: "", min: "", cold: false, pos: false, q: "" };
      state.sort = { key: "order", dir: 1 }; state.page = 0;
      activate("list", true);
    });
    exp.addEventListener("click", function () { exportCsv(filtered()); });

    var cols = [{ key: "order", label: "Rank" }, { key: null, label: "Citation (citing → cited)" }, { key: "date", label: "Date" }, { key: "total", label: "Total", num: true }]
      .concat(COMP.map(function (c, i) { return { key: "z" + i, label: "z " + c, comp: i }; }));
    if (hasLabels) cols.push({ key: "label", label: "Label" });

    function refresh() {
      var list = filtered();
      var pages = Math.max(1, Math.ceil(list.length / state.pageSize));
      state.page = Math.min(state.page, pages - 1);
      var slice = list.slice(state.page * state.pageSize, (state.page + 1) * state.pageSize);
      count.textContent = fmtInt(list.length) + " of " + fmtInt(rows.length) + " rows";
      var thead = el("tr", null, cols.map(function (c) {
        var th = el("th", { scope: "col", class: c.num ? "num" : null });
        if (c.key) {
          var sorted = state.sort.key === c.key;
          th.setAttribute("aria-sort", sorted ? (state.sort.dir > 0 ? "ascending" : "descending") : "none");
          var b = el("button", { type: "button", class: "cx-sort" }, c.comp !== undefined ? el("span", { class: "cx-swatch", style: "background:" + COMP_COLOR[c.comp] + ";width:8px;height:8px" }) : null, c.label);
          b.addEventListener("click", function () {
            if (state.sort.key === c.key) state.sort.dir = -state.sort.dir;
            else state.sort = { key: c.key, dir: c.key === "order" || c.key === "date" ? 1 : -1 };
            refresh();
          });
          th.appendChild(b);
        } else th.textContent = c.label;
        return th;
      }));
      var tbody = el("tbody");
      slice.forEach(function (r) {
        var rk = r.rank === null ? chip("cold", "cold start") : null;
        var btn = null;
        if (r.rank !== null) {
          btn = el("button", { type: "button", class: "cx-rankbtn", "aria-label": "Open details for the citation ranked " + r.rank + (r.detail ? "" : " (no precomputed detail)"), disabled: r.detail ? null : true, text: "#" + fmtInt(r.rank) });
          btn.addEventListener("click", function (ev) { ev.stopPropagation(); openDetail(r.id); });
        }
        var tr = el("tr", { class: (r.detail ? "is-clickable" : "") + (state.visited && r.id === state.detailId ? " is-current" : "") },
          el("td", { class: "num" }, btn || rk),
          el("td", null, citeCell(r.src, r.dst)),
          el("td", { class: "num", text: r.date }),
          el("td", { class: "num" }, el("b", { text: fmt(r.total) })),
          COMP.map(function (c, i) {
            return el("td", null, el("span", { class: "cx-zcell", title: c + " z = " + fmtSigned(r.z[i], 2) + ", raw = " + fmt(r.raw[i]) }, zBar(r.z[i], COMP_COLOR[i]), el("span", { class: "num", text: fmtSigned(r.z[i], 2) })));
          }),
          hasLabels ? el("td", null, r.label === 1 ? chip("pos", "positive") : r.label === 0 ? el("span", { class: "muted", text: "0" }) : el("span", { class: "muted", text: "no label" })) : null);
        if (r.detail) tr.addEventListener("click", function () { openDetail(r.id); });
        tbody.appendChild(tr);
      });
      wrap.textContent = "";
      wrap.appendChild(el("table", null, el("caption", { class: "sr", text: "Ranked citation events" }), el("thead", null, thead), tbody));
      if (!slice.length) wrap.appendChild(el("p", { class: "cx-empty", text: "No rows match these filters." }));
      pager.textContent = "";
      var prev = el("button", { type: "button", class: "cx-btn", disabled: state.page === 0 ? true : null, text: "Previous" });
      var next = el("button", { type: "button", class: "cx-btn", disabled: state.page >= pages - 1 ? true : null, text: "Next" });
      prev.addEventListener("click", function () { state.page -= 1; refresh(); });
      next.addEventListener("click", function () { state.page += 1; refresh(); });
      pager.appendChild(el("span", { class: "muted", text: "Page " + (state.page + 1) + " of " + pages }));
      pager.appendChild(prev);
      pager.appendChild(next);
    }
    refresh();
  }
  function csvCell(v) {
    if (v === null || v === undefined) return "";
    var s = String(v);
    return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }
  function exportCsv(list) {
    var head = ["rank", "event_id", "date", "t", "src_id", "dst_id", "src_title", "dst_title", "src_field", "dst_field", "total"]
      .concat(COMP.map(function (c) { return "z_" + c; }), COMP, ["cold_start", "label"]);
    var lines = [head.join(",")];
    list.forEach(function (r) {
      lines.push([r.rank, r.id, r.date, r.t, nodeId(r.src), nodeId(r.dst), NC.title[r.src], NC.title[r.dst], nodeField(r.src), nodeField(r.dst), r.total]
        .concat(r.z, r.raw, [r.cold ? 1 : 0, r.label]).map(csvCell).join(","));
    });
    var blob = new Blob([lines.join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
    var a = el("a", { href: URL.createObjectURL(blob), download: "citation_flags_filtered.csv" });
    root.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }

  // ------------------------------------------------------------------ CITATION DETAIL
  function buildDetail(p) {
    if (state.detailId === null) { p.appendChild(el("p", { class: "cx-card cx-empty", text: "No ranked events, so there is nothing to explain." })); return; }
    var r = rowById[state.detailId], d = D.details[String(state.detailId)];
    var list = filtered().filter(function (x) { return x.detail; });
    if (!list.some(function (x) { return x.id === r.id; })) list = detailOrder;
    var pos = list.findIndex(function (x) { return x.id === r.id; });

    var nav = el("div", { class: "cx-detailnav" });
    var back = el("button", { type: "button", class: "cx-btn", text: "← Ranked list" });
    back.addEventListener("click", function () { activate("list"); });
    var prev = el("button", { type: "button", class: "cx-btn", disabled: pos <= 0 ? true : null, text: "Previous" });
    var next = el("button", { type: "button", class: "cx-btn", disabled: pos >= list.length - 1 ? true : null, text: "Next" });
    prev.addEventListener("click", function () { openDetail(list[pos - 1].id); });
    next.addEventListener("click", function () { openDetail(list[pos + 1].id); });
    var sel = el("select", { "aria-label": "Jump to a ranked citation" }, detailOrder.map(function (x) {
      return el("option", { value: String(x.id), selected: x.id === r.id ? true : null, text: "#" + x.rank + " · " + short(nodeTitle(x.src), 34) + " → " + short(nodeTitle(x.dst), 34) });
    }));
    sel.addEventListener("change", function () { openDetail(+sel.value); });
    nav.appendChild(back); nav.appendChild(prev); nav.appendChild(next); nav.appendChild(sel);
    p.appendChild(nav);

    // header
    var flagged = M.threshold !== null && r.total >= M.threshold;
    var hc = el("section", { class: "cx-card" });
    hc.appendChild(el("div", { class: "cx-card__head" },
      el("h2", { text: "Rank #" + fmtInt(r.rank) + " · " + r.date }),
      el("div", { class: "cx-controls", style: "margin:0" }, flagged ? chip("flag", "in top " + fmtPct(M.flag_share, 0)) : null,
        r.label === 1 ? chip("pos", "positive label") : r.label === 0 ? chip("cold", "label 0") : null)));
    function paper(role, i) {
      var meta = [nodeId(i), nodeField(i), nodeVenue(i), NC.date[i]].filter(Boolean).join(" · ");
      return el("div", { class: "cx-paper" }, el("div", { class: "cx-paper__role", text: role }), el("div", { class: "cx-paper__title", text: nodeTitle(i) }), el("div", { class: "cx-paper__meta", text: meta }));
    }
    hc.appendChild(el("div", { class: "cx-pair" }, paper("citing paper (u, src)", r.src), el("div", { class: "cx-pair__arrow", "aria-hidden": "true", text: "→" }), paper("cited paper (v, dst)", r.dst)));
    hc.appendChild(el("div", { class: "cx-tiles", style: "margin-top:12px" },
      tile("Total score", fmt(r.total), "threshold " + fmt(M.threshold), true),
      tile("Event id", String(r.id), "t = " + r.t + " days since 1970-01-01", true),
      d.curv ? tile("Forman screen F", fmt(d.curv.forman, 0), d.curv.escalated ? "ORC " + fmt(d.curv.orc) + " (escalated)" : "not escalated to ORC", true) : null));
    p.appendChild(hc);

    // why flagged: contributions
    var wc = card("Why this was flagged", "Each component adds w × z to the total. Bars run from the running sum before the component to the sum after it.");
    wc.style.marginTop = "16px";
    wc.appendChild(chart("detail", el("div", { class: "cx-chart" }), function (box, w) { drawWaterfall(box, w, r, d); }));
    var gap = d.sum_gap;
    if (gap !== null && Math.abs(gap) > 1e-6) wc.appendChild(el("p", { class: "cx-warn", text: "The contributions add up to " + fmt(r.total - gap) + ", not to the stored total " + fmt(r.total) + " (gap " + fmt(gap) + "). The stored total is authoritative; check the weights and z statistics passed to the explorer." }));
    else wc.appendChild(el("p", { class: "cx-caption", text: "Sum of contributions = stored total = " + fmt(r.total) + "." + (M.weights_source !== "given" || M.zstats_source !== "given" ? " Uses fallback weights or z statistics (see Overview)." : "") }));
    wc.appendChild(dataTable("Data table", [{ label: "component" }, { label: "raw", num: true }, { label: "z", num: true }, { label: "w", num: true }, { label: "w × z", num: true }, { label: "percentile", num: true }],
      COMP.map(function (c) { return [c, fmt(d.raw[c]), fmtSigned(d.z[c]), fmt(M.weights[c]), fmtSigned(d.contrib[c]), fmtPct(d.pct[c])]; })));
    p.appendChild(wc);

    // component cards
    var cg = el("div", { class: "cx-grid cx-grid--2" });
    COMP.forEach(function (c, i) {
      var T = D.component_text[c] || {};
      var cc = el("section", { class: "cx-card" });
      cc.appendChild(el("div", { class: "cx-comp__head" }, el("span", { class: "cx-comp__key", style: "background:" + COMP_COLOR[i] }), el("h3", { text: (T.name || c) + " (" + c + ")" })));
      cc.appendChild(el("div", { class: "cx-comp__value" }, "raw " + fmt(d.raw[c]), el("span", { class: "muted", style: "font-size:13px;font-weight:400", text: "  · z " + fmtSigned(d.z[c], 2) + " · percentile " + fmtPct(d.pct[c]) })));
      var Hc = D.hist.components[c];
      if (Hc) {
        cc.appendChild(chart("detail", el("div", { class: "cx-chart", style: "margin-top:6px" }), function (box, w) {
          drawHistogram(box, w, { hist: Hc, color: COMP_COLOR[i], xLabel: c + " (raw value)", marker: { value: d.raw[c], label: "this event" }, height: 160, aria: "Reference distribution of " + c + " with this event marked" });
        }));
        cc.appendChild(el("p", { class: "cx-caption", text: "Reference: " + fmtInt(Hc.n) + " ranked events in this file (median " + fmt(Hc.median) + ")." }));
      } else cc.appendChild(el("p", { class: "cx-empty", text: "No finite values for this component." }));
      if (T.formula) cc.appendChild(el("div", { class: "cx-formula", text: T.formula }));
      if (T.text) cc.appendChild(el("p", { class: "cx-text", text: T.text }));
      cg.appendChild(cc);
    });
    p.appendChild(cg);

    // local subgraph + timeline
    var gc = card("Local citation graph at time t", d.graph ? "Graph state on " + r.date + ": events with ts ≤ t; this citation is drawn highlighted and excluded from the context." : null);
    gc.style.marginTop = "16px";
    if (d.graph) buildGraph(gc, r, d); else gc.appendChild(missingNote("events"));
    p.appendChild(gc);
    var tc = card("Citations received by the cited paper up to t", d.timeline ? "Bursts show as tall columns. This citation is counted in the last bin." : null);
    tc.style.marginTop = "16px";
    if (d.timeline && d.timeline.c.length) {
      var TL = d.timeline;
      tc.appendChild(el("div", { class: "cx-tiles" }, tile("Received up to t", fmtInt(TL.n), "this citation included", true), tile("Same day as t", fmtInt(TL.same_day), null, true), tile("Last 30 days", fmtInt(TL.last30), null, true), tile("Bin width", fmtInt(TL.w) + (TL.w === 1 ? " day" : " days"), null, true)));
      tc.appendChild(chart("detail", el("div", { class: "cx-chart", style: "margin-top:10px" }), function (box, w) { drawTimeline(box, w, TL); }));
      var nb = TL.c.length;
      tc.appendChild(dataTable("Data table", [{ label: "from" }, { label: "to" }, { label: "citations", num: true }], TL.c.map(function (c, j) {
        var end = TL.end - (nb - 1 - j) * TL.w; return [day(end - TL.w + 1), day(end), fmtInt(c)];
      })));
    } else tc.appendChild(d.timeline ? el("p", { class: "cx-empty", text: "No citations recorded for this paper up to t." }) : missingNote("events"));
    p.appendChild(tc);

    // inputs (NFR10) + formulas
    var ic = card("Inputs and definitions (NFR10 traceability)", "Every stored value for this event, as passed to the explorer.");
    ic.style.marginTop = "16px";
    ic.appendChild(table([{ label: "field" }, { label: "value" }], d.inputs.map(function (kv) { return [kv[0], kv[1] === null ? "null" : typeof kv[1] === "number" ? String(kv[1]) : String(kv[1])]; }), { kv: true, caption: "Inputs" }));
    var defs = el("details", null, el("summary", { text: "Component definitions" }),
      el("p", { class: "cx-warn", text: M.component_status }),
      COMP.map(function (c) { var T = D.component_text[c] || {}; return el("div", { style: "margin-top:10px" }, el("h3", { text: (T.name || c) + " (" + c + ")" }), el("div", { class: "cx-formula", text: T.formula || "" }), el("p", { class: "cx-text", text: T.text || "" })); }),
      el("div", { style: "margin-top:10px" }, el("h3", { text: "Combiner" }), el("p", { class: "cx-text", text: M.combiner_text })));
    ic.appendChild(defs);
    p.appendChild(ic);
  }

  function drawWaterfall(box, width, r, d) {
    var lw = width < 520 ? 94 : 168, band = 34, n = COMP.length + 1;
    var f = frame(box, width, band * n + 34, { l: lw, r: width < 520 ? 52 : 150, t: 6, b: 28 });
    var segs = [], acc = 0;
    COMP.forEach(function (c, i) { var v = d.contrib[c]; var s0 = acc; if (v !== null) acc += v; segs.push({ c: c, i: i, v: v, s0: s0, s1: acc }); });
    var vals = [0, r.total];
    segs.forEach(function (s) { vals.push(s.s0, s.s1); });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), pad = (hi - lo || 1) * 0.06;
    var tk = niceTicks(lo - pad, hi + pad, Math.max(2, Math.floor(f.iw / 80)));
    var xs = lin(tk[0], tk[tk.length - 1], 0, f.iw);
    var grid = sv("g", { class: "grid" }), ax = sv("g", { class: "ax" });
    tk.forEach(function (v) { grid.appendChild(sv("line", { x1: xs(v), x2: xs(v), y1: 0, y2: band * n })); ax.appendChild(sv("text", { x: xs(v), y: band * n + 16, "text-anchor": "middle", text: fmt(v, 1) })); });
    f.g.appendChild(grid);
    f.g.appendChild(ax);
    f.g.appendChild(sv("line", { x1: xs(0), x2: xs(0), y1: -4, y2: band * n, style: "stroke:var(--axis);stroke-width:1" }));
    var marks = [], bh = 14;
    segs.forEach(function (s, k) {
      var y = k * band + (band - bh) / 2;
      f.g.appendChild(sv("rect", { x: -lw + 2, y: y + 2, width: 10, height: 10, rx: 2, style: "fill:" + COMP_COLOR[s.i] }));
      f.g.appendChild(sv("text", { class: "lbl-ink", x: -lw + 18, y: y + bh / 2, "dominant-baseline": "middle", text: lw > 120 ? compName(s.c) : s.c }));
      var bar = null;
      if (s.v !== null) {
        bar = sv("path", { class: "mark", d: hbarPath(xs(s.s0), xs(s.s1), y, bh, 3), style: "fill:" + COMP_COLOR[s.i] });
        f.g.appendChild(bar);
        if (k < segs.length) f.g.appendChild(sv("line", { x1: xs(s.s1), x2: xs(s.s1), y1: y + bh, y2: y + band, style: "stroke:var(--axis);stroke-width:1" }));
      }
      var endX = s.v === null ? xs(s.s0) : xs(Math.max(s.s0, s.s1));
      var txt = s.v === null ? "n/a" : fmtSigned(s.v, 2);
      var label = sv("text", { class: "lbl-ink", x: endX + 6, y: y + bh / 2, "dominant-baseline": "middle" }, sv("tspan", { style: "font-weight:600", text: txt }),
        width >= 520 && s.v !== null ? sv("tspan", { class: "lbl", dx: 6, text: "w " + fmt(M.weights[s.c], 2) + " × z " + fmtSigned(d.z[s.c], 2) }) : null);
      f.g.appendChild(label);
      var hit = sv("rect", { class: "hit", x: -lw, y: k * band, width: f.iw + lw, height: band });
      f.g.appendChild(hit);
      marks.push({ hit: hit, hot: bar, tip: function () {
        return { head: compName(s.c) + " (" + s.c + ")", rows: [["w × z", fmtSigned(s.v), COMP_COLOR[s.i]], ["z", fmtSigned(d.z[s.c])], ["weight", fmt(M.weights[s.c])], ["raw value", fmt(d.raw[s.c])], ["running sum", fmt(s.s1)]] };
      } });
    });
    var yT = COMP.length * band + (band - bh) / 2;
    f.g.appendChild(sv("text", { class: "lbl-ink", x: -lw + 2, y: yT + bh / 2, "dominant-baseline": "middle", style: "font-weight:600", text: "Total" }));
    var tbar = sv("path", { class: "mark", d: hbarPath(xs(0), xs(r.total), yT, bh, 3), style: "fill:var(--total)" });
    f.g.appendChild(tbar);
    f.g.appendChild(sv("text", { class: "lbl-ink", x: xs(Math.max(0, r.total)) + 6, y: yT + bh / 2, "dominant-baseline": "middle", style: "font-weight:600", text: fmt(r.total, 2) }));
    var thit = sv("rect", { class: "hit", x: -lw, y: COMP.length * band, width: f.iw + lw, height: band });
    f.g.appendChild(thit);
    marks.push({ hit: thit, hot: tbar, tip: function () { return { head: "Total score", rows: [["stored total", fmt(r.total), "var(--total)"], ["flag threshold", fmt(M.threshold)]] }; } });
    bindMarks(f.svg, marks, "Contribution of each component to the total score");
  }

  function drawTimeline(box, width, TL) {
    var nb = TL.c.length;
    var f = frame(box, width, 190, { l: 46, r: 14, t: 22, b: 40 });
    var ymax = Math.max(1, Math.max.apply(null, TL.c)), yt = niceTicks(0, ymax, 4, true), ys = lin(0, yt[yt.length - 1], f.ih, 0);
    yAxis(f, ys, yt, compact);
    f.g.appendChild(sv("line", { class: "ax", x1: 0, x2: f.iw, y1: f.ih, y2: f.ih, style: "stroke:var(--axis)" }));
    var band = f.iw / nb, bw = Math.max(1, Math.min(24, band - 2)), marks = [];
    TL.c.forEach(function (c, j) {
      var end = TL.end - (nb - 1 - j) * TL.w, start = end - TL.w + 1, last = j === nb - 1;
      var x = j * band + (band - bw) / 2;
      var bar = sv("path", { class: "mark", d: colPath(x, ys(c), bw, f.ih - ys(c), Math.min(4, bw / 2)), style: "fill:" + (last ? "var(--pick)" : "var(--total)") });
      var hit = sv("rect", { class: "hit", x: j * band, y: 0, width: band, height: f.ih });
      f.g.appendChild(bar);
      f.g.appendChild(hit);
      marks.push({ hit: hit, hot: bar, tip: function () { return { head: day(start) + (TL.w > 1 ? " to " + day(end) : ""), rows: [["citations", fmtInt(c)]].concat(last ? [["contains t", "yes"]] : []) }; } });
    });
    var ax = sv("g", { class: "ax" });
    var picks = nb > 2 ? [0, Math.floor(nb / 2), nb - 1] : [0, nb - 1];
    picks.filter(function (v, i, a) { return a.indexOf(v) === i; }).forEach(function (j) {
      var end = TL.end - (nb - 1 - j) * TL.w;
      var x = j * band + band / 2;
      ax.appendChild(sv("text", { x: x, y: f.ih + 16, "text-anchor": j === 0 ? "start" : j === nb - 1 ? "end" : "middle", text: day(j === nb - 1 ? end : end - TL.w + 1) }));
    });
    f.g.appendChild(ax);
    var lx = (nb - 0.5) * band;
    f.g.appendChild(sv("text", { class: "lbl-ink", x: lx, y: -9, "text-anchor": "end", text: "bin containing t" }));
    axisTitles(f, "date (bin start; last label = t)", "citations per " + (TL.w === 1 ? "day" : TL.w + "-day bin"));
    bindMarks(f.svg, marks, "Citations received by the cited paper over time");
  }

  // ---- graph drawing shared by the detail and cartel views
  function drawNet(box, width, spec) {
    var w = width, h = spec.height || Math.round(Math.min(520, Math.max(300, w * 0.62))), pad = 30;
    box.textContent = "";
    var svg = sv("svg", { width: w, height: h, viewBox: "0 0 " + w + " " + h });
    box.appendChild(svg);
    function px(n) { return [pad + n.x * (w - 2 * pad), pad + n.y * (h - 2 * pad)]; }
    var pairs = {};
    spec.edges.forEach(function (e) { pairs[e.a + ">" + e.b] = true; });
    var eg = sv("g"), hg = sv("g"), ng = sv("g"), lg = sv("g");
    svg.appendChild(eg); svg.appendChild(hg); svg.appendChild(ng); svg.appendChild(lg);
    var edgeMarks = [];
    spec.edges.forEach(function (e) {
      var A = spec.nodes[e.a], B = spec.nodes[e.b];
      if (!A || !B) return;
      var p1 = px(A), p2 = px(B), dx = p2[0] - p1[0], dy = p2[1] - p1[1], L = Math.hypot(dx, dy) || 1;
      var ux = dx / L, uy = dy / L, off = pairs[e.b + ">" + e.a] ? 3 : 0;
      var ox = -uy * off, oy = ux * off;
      var x1 = p1[0] + ux * (A.r + 1) + ox, y1 = p1[1] + uy * (A.r + 1) + oy;
      var x2 = p2[0] - ux * (B.r + 3) + ox, y2 = p2[1] - uy * (B.r + 3) + oy;
      var g = sv("g");
      if (e.halo) g.appendChild(sv("line", { x1: x1, y1: y1, x2: x2, y2: y2, style: "stroke:var(--pick);stroke-width:8;opacity:.28;stroke-linecap:round" }));
      g.appendChild(sv("line", { class: "mark", x1: x1, y1: y1, x2: x2 - ux * 5, y2: y2 - uy * 5, style: "stroke:" + e.stroke + ";stroke-width:" + e.width + ";stroke-linecap:round" + (e.opacity ? ";opacity:" + e.opacity : "") }));
      var ah = 4 + e.width;
      g.appendChild(sv("path", { d: "M" + x2 + "," + y2 + "L" + (x2 - ux * ah - uy * ah * 0.6) + "," + (y2 - uy * ah + ux * ah * 0.6) + "L" + (x2 - ux * ah + uy * ah * 0.6) + "," + (y2 - uy * ah - ux * ah * 0.6) + "Z", style: "fill:" + e.stroke + (e.opacity ? ";opacity:" + e.opacity : "") }));
      (e.halo ? hg : eg).appendChild(g);
      var hit = sv("line", { x1: x1, y1: y1, x2: x2, y2: y2, style: "stroke:transparent;stroke-width:12" });
      (e.halo ? hg : eg).appendChild(hit);
      if (e.tip) edgeMarks.push({ hit: hit, tip: e.tip });
    });
    var marks = [];
    spec.nodes.forEach(function (n) {
      if (!n) return;
      var p = px(n);
      var c = sv("circle", { class: "mark", cx: p[0], cy: p[1], r: n.r, style: n.style + ";stroke-linejoin:round" });
      ng.appendChild(c);
      var hit = sv("circle", { cx: p[0], cy: p[1], r: Math.max(12, n.r + 6), class: "hit" });
      ng.appendChild(hit);
      if (n.label) {
        var below = p[1] < 40;
        lg.appendChild(sv("text", { x: p[0], y: below ? p[1] + n.r + 14 : p[1] - n.r - 7, "text-anchor": p[0] < 70 ? "start" : p[0] > w - 70 ? "end" : "middle", class: "lbl-ink",
          style: "font-weight:600;paint-order:stroke;stroke:var(--card);stroke-width:3px;stroke-linejoin:round", text: n.label }));
      }
      marks.push({ hit: hit, hot: c, tip: n.tip, activate: n.activate });
    });
    edgeMarks.forEach(function (m) {
      m.hit.addEventListener("pointermove", function (ev) { showTip(m.tip(), ev.clientX, ev.clientY); });
      m.hit.addEventListener("pointerleave", hideTip);
    });
    bindMarks(svg, marks, spec.aria);
  }

  function buildGraph(gc, r, d) {
    var G = d.graph, maxHop = 0;
    G.nodes.forEach(function (n) { maxHop = Math.max(maxHop, n[3]); });
    var hasOrc = G.edges.some(function (e) { return e[6] !== null; });
    var ctr = el("div", { class: "cx-controls" });
    var box = el("div", { class: "cx-chart" });
    var lgBox = el("div");
    if (state.radius > Math.max(1, maxHop)) state.radius = Math.max(1, maxHop);
    if (state.edgeColor === "orc" && !hasOrc) state.edgeColor = "forman";
    ctr.appendChild(seg("Radius", [{ value: 1, label: "1 hop" }, { value: 2, label: "2 hops", disabled: maxHop < 2, title: maxHop < 2 ? "Built with radius 1 (rebuild with radius=2)" : null }], state.radius, function (v) { state.radius = v; redraw("detail", true); }));
    ctr.appendChild(seg("Nodes", [{ value: "role", label: "Role" }, { value: "field", label: "Field", disabled: !hasWorks }], state.nodeColor, function (v) { state.nodeColor = v; redraw("detail", true); }));
    ctr.appendChild(seg("Edges", [{ value: "forman", label: "Forman at t" }, { value: "orc", label: "ORC", disabled: !hasOrc, title: hasOrc ? null : "No escalated edge in this subgraph" }], state.edgeColor, function (v) { state.edgeColor = v; redraw("detail", true); }));
    gc.appendChild(ctr);
    gc.appendChild(box);
    gc.appendChild(lgBox);
    // global field -> colour mapping: the three most common fields among all embedded papers keep slots 1-3
    var counts = {};
    NC.field.forEach(function (fi) { if (fi !== null) counts[fi] = (counts[fi] || 0) + 1; });
    var topFields = Object.keys(counts).map(Number).sort(function (a, b) { return counts[b] - counts[a] || D.fields[a].localeCompare(D.fields[b]); }).slice(0, 3);
    chart("detail", box, function (bx, width) {
      var vis = {}, nodes = [];
      G.nodes.forEach(function (n, k) {
        if (n[3] > state.radius) { nodes.push(null); return; }
        vis[k] = true;
        var core = n[3] === 0, isSrc = core && n[0] === r.src, isDst = core && n[0] === r.dst;
        var style, rr = core ? 8 : n[3] === 1 ? 5.5 : 4.5;
        if (state.nodeColor === "role") {
          if (isSrc) style = "fill:var(--pick);stroke:var(--card);stroke-width:2";
          else if (isDst) style = "fill:var(--card);stroke:var(--pick);stroke-width:3";
          else style = "fill:" + (n[3] === 1 ? "var(--node1)" : "var(--node2)") + ";stroke:var(--card);stroke-width:2";
        } else {
          var fi = NC.field[n[0]], slot = topFields.indexOf(fi);
          var fill = slot >= 0 ? COMP_COLOR[slot] : "var(--node2)";
          style = "fill:" + fill + ";stroke:" + (core ? "var(--ink)" : "var(--card)") + ";stroke-width:2";
        }
        nodes.push({ x: n[1], y: n[2], r: rr, style: style, label: isSrc ? "u (citing)" : isDst ? "v (cited)" : null,
          activate: null,
          tip: function () {
            return { head: nodeTitle(n[0]), rows: [["paper id", nodeId(n[0])], ["field", nodeField(n[0]) || "unknown"], ["position", core ? (isSrc ? "citing paper u" : "cited paper v") : "ring " + n[3]],
              ["in-degree at t", fmtInt(n[4])], ["out-degree at t", fmtInt(n[5])]] };
          } });
      });
      var br = state.edgeColor === "orc" ? M.curv_scale.orc : M.curv_scale.forman;
      var edges = [];
      G.edges.forEach(function (e) {
        if (!vis[e[0]] || !vis[e[1]]) return;
        var v = state.edgeColor === "orc" ? e[6] : e[5], k = divClass(v, br), scored = !!e[7];
        var stroke = k < 0 ? "var(--node2)" : DIV[k];
        var outer = G.nodes[e[0]][3] === 2 || G.nodes[e[1]][3] === 2;  // ring-2 context recedes
        edges.push({ a: e[0], b: e[1], stroke: stroke, width: scored ? 2.5 : e[2] > 1 ? 2 : outer ? 1 : 1.4, halo: scored,
          opacity: scored ? null : outer ? 0.45 : k < 0 ? 0.7 : null,
          tip: function () {
            var A = G.nodes[e[0]][0], B = G.nodes[e[1]][0];
            return { head: short(nodeTitle(A), 40) + " → " + short(nodeTitle(B), 40), rows: (scored ? [["edge", "this citation"]] : [["events", fmtInt(e[2])], ["first", day(e[3])], ["last", day(e[4])]])
              .concat([["Forman" + (scored && d.curv ? " (cache)" : " at t"), fmt(e[5], 0), k >= 0 && state.edgeColor === "forman" ? DIV[k] : null], ["ORC", e[6] === null ? "not escalated" : fmt(e[6])]]) };
          } });
      });
      edges.sort(function (a, b) { return (a.halo ? 1 : 0) - (b.halo ? 1 : 0); });
      drawNet(bx, width, { nodes: nodes, edges: edges, aria: "Local citation graph around the scored citation; arrow keys read the nodes" });
      lgBox.textContent = "";
      var nodeItems = state.nodeColor === "role"
        ? [{ color: "var(--pick)", label: "citing paper u" }, { color: "var(--pick)", label: "cited paper v", kind: "ring" }, { color: "var(--node1)", label: "ring 1" }].concat(maxHop > 1 && state.radius > 1 ? [{ color: "var(--node2)", label: "ring 2" }] : [])
        : topFields.map(function (fi, s) { return { color: COMP_COLOR[s], label: D.fields[fi] }; }).concat([{ color: "var(--node2)", label: "other field" }]);
      lgBox.appendChild(legend(nodeItems));
      var dg = state.edgeColor === "orc" ? 2 : 0, b = br.map(function (x) { return fmt(x, dg); });
      var names = ["< −" + b[2], "−" + b[2] + " to −" + b[1], "−" + b[1] + " to −" + b[0], "−" + b[0] + " to " + b[0], b[0] + " to " + b[1], b[1] + " to " + b[2], "> " + b[2]];
      var shown = state.edgeColor === "orc" ? [0, 1, 2, 3, 4, 5, 6] : [0, 1, 2, 3];  // Forman F <= 2 by construction
      lgBox.appendChild(el("div", { class: "cx-legend" }, el("span", { text: (state.edgeColor === "orc" ? "ORC" : "Forman at t") + ":" }),
        shown.map(function (i) { return el("span", { class: "cx-legend__item" }, el("span", { class: "cx-swatch cx-swatch--line", style: "background:" + DIV[i] }), names[i]); }),
        state.edgeColor === "orc" ? el("span", { class: "cx-legend__item" }, el("span", { class: "cx-swatch cx-swatch--line", style: "background:var(--node2)" }), "not escalated") : null,
        el("span", { class: "cx-legend__item" }, el("span", { class: "cx-swatch cx-swatch--line", style: "background:var(--pick);opacity:.5;height:6px" }), "this citation")));
    });
    var caption = "Ring 1: " + fmtInt(G.ring1_shown) + " of " + fmtInt(G.ring1_total) + " neighbours shown" + (G.ring2_total ? "; ring 2: " + fmtInt(G.ring2_shown) + " of " + fmtInt(G.ring2_total) + " (neighbours of the shown ring-1 nodes)" : "") +
      ". Cap " + M.max_nodes + " nodes: kept by links to u and v, then recency. Edge colour: Forman F(a→b) = 2 − in_deg_t(a) − out_deg_t(b) recomputed at t (the scored edge uses the cache when given); classes: |F| ≤ 2 neutral, then breaks at the median (" + fmt(M.curv_scale.forman[1], 0) + ") and 90th percentile (" + fmt(M.curv_scale.forman[2], 0) + ") of the larger |F| over all drawn edges. ORC: cached value of the latest escalated instance at or before t (breaks 0.05 / 0.25 / 0.5).";
    gc.appendChild(el("p", { class: "cx-caption", text: caption }));
    gc.appendChild(dataTable("Data table: edges", [{ label: "from" }, { label: "to" }, { label: "events", num: true }, { label: "last", num: true }, { label: "Forman", num: true }, { label: "ORC", num: true }],
      G.edges.map(function (e) { return [nodeId(G.nodes[e[0]][0]), nodeId(G.nodes[e[1]][0]) + (e[7] ? " (this citation)" : ""), fmtInt(e[2]), day(e[4]), fmt(e[5], 0), e[6] === null ? "n/a" : fmt(e[6])]; })));
  }

  // ------------------------------------------------------------------ PAPERS & AUTHORS
  function buildRollup(p) {
    var mode = state.roll;
    if (mode === "authors" && !D.authors) mode = state.roll = "papers";
    var top = el("div", { class: "cx-controls" });
    top.appendChild(seg("Rollup", [{ value: "papers", label: "Papers" }, { value: "authors", label: "Authors", disabled: !D.authors, title: D.authors ? null : D.missing.authorships }], mode, function (v) { state.roll = v; state.rollPage = 0; activate("rollup", true); }));
    p.appendChild(top);
    if (!D.authors) p.appendChild(el("p", { class: "cx-note", style: "margin-bottom:10px", text: D.missing.authorships }));
    var items = mode === "papers" ? D.papers : D.authors;
    if (!items || !items.length) { p.appendChild(el("p", { class: "cx-card cx-empty", text: "No ranked events to roll up." })); return; }
    var g = el("div", { class: "cx-grid cx-grid--2", style: "margin-top:0" });
    p.appendChild(g);
    var lc = card(mode === "papers" ? "Papers by rollup score" : "Authors by rollup score",
      (mode === "papers" ? "Paper score = mean of its 5 highest incident edge totals (as citing or cited paper; fewer if it has fewer)." : "Author score = mean of the 5 highest totals over distinct events incident to any of the author's papers.") +
      " Count = incident events with total ≥ " + fmt(M.threshold) + " (top " + fmtPct(M.flag_share, 0) + "). Definition from S3.3.");
    g.appendChild(lc);
    var dc = el("div");
    g.appendChild(dc);
    var per = 25, pages = Math.ceil(items.length / per);
    var wrap = el("div"), pager = el("div", { class: "cx-pager" });
    lc.appendChild(wrap);
    lc.appendChild(pager);
    function render() {
      var slice = items.slice(state.rollPage * per, (state.rollPage + 1) * per);
      var cols = mode === "papers"
        ? [{ label: "#" }, { label: "score", num: true }, { label: "≥ thr", num: true }, { label: "Paper" }, { label: "events", num: true }]
        : [{ label: "#" }, { label: "score", num: true }, { label: "≥ thr", num: true }, { label: "Author" }, { label: "papers", num: true }];
      var tbody = el("tbody");
      slice.forEach(function (it, j) {
        var idx = state.rollPage * per + j;
        var btn = el("button", { type: "button", class: "cx-rankbtn", "aria-label": "Show flagged citations of entry " + (idx + 1), text: String(idx + 1) });
        var name = mode === "papers" ? el("div", { class: "cx-cite" }, el("div", { text: short(nodeTitle(it.p), 90) }), el("div", { class: "cx-cite__ids", text: nodeId(it.p) + (nodeField(it.p) ? " · " + nodeField(it.p) : "") }))
          : el("div", null, el("div", { class: "mono", text: it.a }));
        var tr = el("tr", { class: "is-clickable" + (idx === state.rollSel[mode] ? " is-current" : "") }, el("td", { class: "num" }, btn), el("td", { class: "num" }, el("b", { text: fmt(it.score) })),
          el("td", { class: "num", text: fmtInt(it.n_above) }), el("td", null, name), el("td", { class: "num", text: mode === "papers" ? fmtInt(it.n_events) + " (" + it.n_src + " / " + it.n_dst + ")" : fmtInt(it.n_papers) }));
        function pick() {
          state.rollSel[mode] = idx; render(); showSel();
          if (dc.getBoundingClientRect().top > lc.getBoundingClientRect().bottom - 4) dc.scrollIntoView({ block: "start", behavior: "smooth" });
        }
        btn.addEventListener("click", function (ev) { ev.stopPropagation(); pick(); });
        tr.addEventListener("click", pick);
        tbody.appendChild(tr);
      });
      var thead = el("thead", null, el("tr", null, cols.map(function (c) { return el("th", { scope: "col", class: c.num ? "num" : null, text: c.label }); })));
      wrap.textContent = "";
      wrap.appendChild(el("div", { class: "cx-tablewrap" }, el("table", null, thead, tbody)));
      if (mode === "papers") wrap.appendChild(el("p", { class: "cx-caption", text: "events = incident ranked events (as citing / as cited)." }));
      pager.textContent = "";
      var prev = el("button", { type: "button", class: "cx-btn", disabled: state.rollPage === 0 ? true : null, text: "Previous" });
      var next = el("button", { type: "button", class: "cx-btn", disabled: state.rollPage >= pages - 1 ? true : null, text: "Next" });
      prev.addEventListener("click", function () { state.rollPage -= 1; render(); });
      next.addEventListener("click", function () { state.rollPage += 1; render(); });
      pager.appendChild(el("span", { class: "muted", text: "Page " + (state.rollPage + 1) + " of " + pages + " · top " + items.length }));
      pager.appendChild(prev);
      pager.appendChild(next);
    }
    function showSel() {
      var it = items[state.rollSel[mode]] || items[0];
      dc.textContent = "";
      var sc = card(mode === "papers" ? short(nodeTitle(it.p), 120) : "Author " + it.a,
        mode === "papers" ? nodeId(it.p) + " · score " + fmt(it.score) + " · " + it.n_above + " events above threshold" : "score " + fmt(it.score) + " · " + it.n_papers + " papers · " + it.n_above + " events above threshold");
      if (mode === "authors" && it.papers.length) {
        sc.appendChild(el("h3", { text: "Top papers", style: "margin-top:4px" }));
        sc.appendChild(el("ul", { class: "cx-notes", style: "margin:4px 0 10px" }, it.papers.map(function (pi) { return el("li", { text: short(nodeTitle(pi), 100) + " (" + nodeId(pi) + ")" }); })));
      }
      sc.appendChild(el("h3", { text: "Highest-scoring citations", style: "margin-bottom:6px" }));
      var trs = it.ev.map(function (e) {
        var eid = e[0], has = !!D.details[String(eid)];
        var rk = e[4];
        var btn = has ? el("button", { type: "button", class: "cx-rankbtn", "aria-label": "Open details for event " + eid, text: "#" + fmtInt(rk) }) : el("span", { class: "muted", text: rk === null ? "cold" : "#" + fmtInt(rk) });
        if (has) btn.addEventListener("click", function () { openDetail(eid); });
        var what = mode === "papers"
          ? el("div", { class: "cx-cite" }, el("div", { class: "muted", style: "font-size:12px", text: e[2] ? "cites" : "cited by" }), el("div", { text: short(nodeTitle(e[1]), 80) }))
          : el("div", { class: "cx-cite" }, el("div", { text: short(nodeTitle(e[1]), 60) }), el("div", { text: "→ " + short(nodeTitle(e[2]), 60) }));
        var total = e[3], t = e[5];
        return [btn, el("b", { text: fmt(total) }), what, day(t)];
      });
      sc.appendChild(table([{ label: "rank" }, { label: "total", num: true }, { label: mode === "papers" ? "other paper" : "citation" }, { label: "date", num: true }], trs));
      sc.appendChild(el("p", { class: "cx-caption", text: "Rank links open the detail view when the event is in the top " + fmtInt(M.top_n) + "." }));
      dc.appendChild(sc);
    }
    render();
    showSel();
  }

  // ------------------------------------------------------------------ CARTELS (synthetic injection track)
  function buildCartels(p) {
    var C = D.cartels;
    if (state.k === null) state.k = C.ks.indexOf(100) >= 0 ? 100 : C.ks[0];
    var k = state.k;
    var ctr = el("div", { class: "cx-controls" });
    var sel = el("select", { "aria-label": "Cartel" }, C.cartels.map(function (c, i) { return el("option", { value: String(i), selected: i === state.cartel ? true : null, text: c.id + " · " + c.level + " · " + c.members.length + " members" }); }));
    sel.addEventListener("change", function () { state.cartel = +sel.value; activate("cartels", true); });
    ctr.appendChild(el("label", null, "Cartel", sel));
    ctr.appendChild(seg("Caught means rank ≤", C.ks.map(function (v) { return { value: v, label: fmtInt(v) }; }), k, function (v) { state.k = v; activate("cartels", true); }));
    p.appendChild(ctr);
    var all = [];
    C.cartels.forEach(function (c) { c.events.forEach(function (e) { all.push(e); }); });
    var caught = all.filter(function (e) { return e.rank !== null && e.rank <= k; }).length;
    p.appendChild(el("div", { class: "cx-tiles" },
      tile("Injected events", fmtInt(C.n_injected), C.cartels.length + " cartels"),
      tile("Caught at rank ≤ " + fmtInt(k), fmtInt(caught), fmtPct(all.length ? caught / all.length : 0) + " recall overall"),
      tile("Unranked (cold start)", fmtInt(all.filter(function (e) { return e.rank === null; }).length), "cannot be caught")));
    var g = el("div", { class: "cx-grid cx-grid--2" });
    p.appendChild(g);
    var rc = card("Recall by obviousness level", "Share of each level's injected events ranked within the top " + fmtInt(k) + ".");
    var levels = Object.keys(C.recall);
    var order = ["blatant", "moderate", "subtle"];
    levels.sort(function (a, b) { var ia = order.indexOf(a), ib = order.indexOf(b); return (ia < 0 ? 9 : ia) - (ib < 0 ? 9 : ib) || a.localeCompare(b); });
    rc.appendChild(chart("cartels", el("div", { class: "cx-chart" }), function (box, width) {
      var lw = 84, band = 34;
      var f = frame(box, width, band * levels.length + 34, { l: lw, r: 56, t: 4, b: 30 });
      var xs = lin(0, 1, 0, f.iw), tk = [0, 0.25, 0.5, 0.75, 1];
      var grid = sv("g", { class: "grid" }), ax = sv("g", { class: "ax" });
      tk.forEach(function (v) { grid.appendChild(sv("line", { x1: xs(v), x2: xs(v), y1: 0, y2: band * levels.length })); ax.appendChild(sv("text", { x: xs(v), y: band * levels.length + 16, "text-anchor": "middle", text: fmtPct(v, 0) })); });
      f.g.appendChild(grid); f.g.appendChild(ax);
      var marks = [];
      levels.forEach(function (lv, i) {
        var v = C.recall[lv][String(k)] || 0, y = i * band + 10;
        f.g.appendChild(sv("text", { class: "lbl-ink", x: -8, y: y + 7, "text-anchor": "end", "dominant-baseline": "middle", text: lv }));
        var bar = sv("path", { class: "mark", d: hbarPath(0, xs(v), y, 14, 4), style: "fill:var(--pick)" });
        f.g.appendChild(bar);
        f.g.appendChild(sv("text", { class: "lbl-ink", x: xs(v) + 6, y: y + 7, "dominant-baseline": "middle", style: "font-weight:600", text: fmtPct(v) }));
        var hit = sv("rect", { class: "hit", x: -lw, y: i * band, width: f.iw + lw, height: band });
        f.g.appendChild(hit);
        marks.push({ hit: hit, hot: bar, tip: function () { return { head: lv, rows: C.ks.map(function (kk) { return ["recall@" + fmtInt(kk), fmtPct(C.recall[lv][String(kk)])]; }) }; } });
      });
      bindMarks(f.svg, marks, "Recall by obviousness level");
    }));
    rc.appendChild(dataTable("Data table", [{ label: "level" }].concat(C.ks.map(function (kk) { return { label: "recall@" + fmtInt(kk), num: true }; })),
      levels.map(function (lv) { return [lv].concat(C.ks.map(function (kk) { return fmtPct(C.recall[lv][String(kk)]); })); })));
    if (C.metrics_levels.length) {
      rc.appendChild(el("h3", { text: "From run metrics", style: "margin-top:12px" }));
      rc.appendChild(table([{ label: "metric" }, { label: "value" }], C.metrics_levels.map(function (kv) { return [kv[0], typeof kv[1] === "number" ? fmt(kv[1], 4) : String(kv[1])]; }), { kv: true }));
    }
    g.appendChild(rc);

    var c = C.cartels[state.cartel];
    var mc = card("Cartel " + c.id + " (" + c.level + ")", c.members.length + " members; all citations among members over the whole stream. Injected edges are dark when caught at rank ≤ " + fmtInt(k) + ".");
    var roles = {};
    c.members.forEach(function (m) { roles[m[1]] = true; });
    var roleList = Object.keys(roles).sort();
    var roleSlot = function (role) { return role === "donor" ? 0 : role === "recipient" ? 1 : -1; };
    var injByPair = {};
    c.events.forEach(function (e) { injByPair[e.a + ">" + e.b] = e; });
    mc.appendChild(chart("cartels", el("div", { class: "cx-chart" }), function (box, width) {
      var nodes = c.members.map(function (m) {
        var slot = roleSlot(m[1]);
        return { x: m[2], y: m[3], r: 6.5, style: "fill:" + (slot >= 0 ? COMP_COLOR[slot] : "var(--node1)") + ";stroke:var(--card);stroke-width:2",
          tip: function () { return { head: nodeTitle(m[0]), rows: [["paper id", nodeId(m[0])], ["role", m[1]], ["field", nodeField(m[0]) || "unknown"]] }; } };
      });
      var edges = c.edges.map(function (e) {
        var inj = injByPair[e[0] + ">" + e[1]], isC = inj && inj.rank !== null && inj.rank <= k;
        return { a: e[0], b: e[1], stroke: inj ? (isC ? "var(--ink)" : "var(--muted)") : "var(--node2)", width: inj ? (isC ? 2 : 1.2) : 1, opacity: inj && !isC ? 0.75 : null,
          tip: function () {
            return { head: short(nodeTitle(c.members[e[0]][0]), 40) + " → " + short(nodeTitle(c.members[e[1]][0]), 40),
              rows: [["events", fmtInt(e[2])], ["injected", inj ? "yes" : "no"]].concat(inj ? [["rank", inj.rank === null ? "unranked" : "#" + fmtInt(inj.rank)], ["caught", isC ? "yes" : "no"]] : []) };
          } };
      });
      edges.sort(function (a, b) { return (a.width > 1.5 ? 1 : 0) - (b.width > 1.5 ? 1 : 0); });
      drawNet(box, width, { nodes: nodes, edges: edges, height: Math.round(Math.min(420, Math.max(280, width * 0.6))), aria: "Cartel member graph" });
    }));
    mc.appendChild(legend(roleList.map(function (rl) { var s = roleSlot(rl); return { color: s >= 0 ? COMP_COLOR[s] : "var(--node1)", label: rl }; })
      .concat([{ color: "var(--ink)", label: "injected, caught", kind: "line" }, { color: "var(--muted)", label: "injected, missed", kind: "line" }, { color: "var(--node2)", label: "other citation", kind: "line" }])));
    g.appendChild(mc);

    var ec = card("Injected events of " + c.id, "Sorted by rank. Rank links open the detail view when the event is in the top " + fmtInt(M.top_n) + ".");
    ec.style.marginTop = "16px";
    var evs = c.events.slice().sort(function (a, b) { return (a.rank === null ? 1e15 : a.rank) - (b.rank === null ? 1e15 : b.rank) || a.e - b.e; });
    ec.appendChild(table([{ label: "rank" }, { label: "caught" }, { label: "total", num: true }, { label: "date", num: true }, { label: "citation (donor \u2192 recipient)" }], evs.map(function (e) {
      var has = !!D.details[String(e.e)];
      var b = has ? el("button", { type: "button", class: "cx-rankbtn", "aria-label": "Open details for event " + e.e, text: "#" + fmtInt(e.rank) }) : el("span", { class: "muted", text: e.rank === null ? "unranked" : "#" + fmtInt(e.rank) });
      if (has) b.addEventListener("click", function () { openDetail(e.e); });
      var isC = e.rank !== null && e.rank <= k;
      return [b, isC ? statusLine(true, "caught") : el("span", { class: "muted", text: "missed" }), el("b", { text: fmt(e.total) }), day(e.t), citeCell(c.members[e.a][0], c.members[e.b][0])];
    })));
    p.appendChild(ec);
  }

  // ------------------------------------------------------------------ header, theme, start
  var run = M.run || {};
  root.querySelector('[data-slot="run"]').textContent = [run.model_variant ? "variant " + run.model_variant : null, run.seed !== undefined && run.seed !== null ? "run seed " + run.seed : null,
    run.config_hash ? "config " + run.config_hash : null, "layout seed " + M.seed].filter(Boolean).join(" · ");
  var themeBtn = root.querySelector('[data-action="theme"]');
  var THEMES = ["auto", "light", "dark"], theme = "auto";
  try { theme = window.localStorage.getItem("citecurv-explorer-theme") || "auto"; } catch (e) { theme = "auto"; }
  if (THEMES.indexOf(theme) < 0) theme = "auto";
  function applyTheme(t) {
    theme = t;
    if (t === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", t);
    themeBtn.textContent = "Theme: " + t;
    themeBtn.setAttribute("aria-label", "Colour theme: " + t + ". Activate to switch.");
    if (document.body && document.body.firstElementChild === root) document.documentElement.style.background = getComputedStyle(root).getPropertyValue("--bg");
  }
  themeBtn.addEventListener("click", function () {
    applyTheme(THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length]);
    try { window.localStorage.setItem("citecurv-explorer-theme", theme); } catch (e) { /* storage unavailable: keep in memory */ }
  });
  applyTheme(theme);
  var resizeTimer = null;
  window.addEventListener("resize", function () { clearTimeout(resizeTimer); resizeTimer = setTimeout(function () { hideTip(); redraw(state.tab); }, 150); });
  window.addEventListener("scroll", hideTip, { passive: true });
  activate("overview");
})();
