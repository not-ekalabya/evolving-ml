"""
visualize.py — Clean minimal live visualisation for model evolution.

Design language:
  • Light background, soft shadows, thin strokes
  • Nodes rendered as layered circles (like neurons) with subtle pulse on update
  • Edges as smooth bezier curves, thickness encodes relative "weight" (param count)
  • Animated signal dots travel along edges — small, monochrome, elegant
  • Right panel: clean typography, two micro sparklines
  • Evolution runs on daemon thread; UI on main thread via queue
"""

import json
import math
import os
import queue
import sys
import time
import tkinter as tk
from collections import deque
from typing import Optional

# ── palette (light, minimal) ──────────────────────────────────────────────────
BG          = "#f8f9fb"
CANVAS_BG   = "#ffffff"
PANEL_BG    = "#f0f2f5"
RULE        = "#e0e4ea"
TEXT_DK     = "#1a202c"
TEXT_MD     = "#4a5568"
TEXT_LT     = "#a0aec0"
EDGE_COL    = "#cbd5e0"
PARTICLE    = "#2b6cb0"

# Node accent colours — muted pastel fills with a darker stroke
NODE_PALETTE = {
    "input":          ("#ebf8ff", "#3182ce"),
    "MatMulNode":     ("#f0fff4", "#38a169"),
    "ActivationNode": ("#fffff0", "#d69e2e"),
    "ReshapeNode":    ("#ebf4ff", "#4299e1"),
    "AddNode":        ("#fff5f0", "#dd6b20"),
    "ConcatNode":     ("#faf5ff", "#805ad5"),
    "ReduceNode":     ("#fff5f5", "#e53e3e"),
    "IndexNode":      ("#f0fff4", "#2f855a"),
    "default":        ("#f7fafc", "#718096"),
}

FONT_FAMILY = "Helvetica"

def _fam(size, bold=False):
    return (FONT_FAMILY, size, "bold" if bold else "normal")


# ── node metadata ─────────────────────────────────────────────────────────────

def _node_type_key(node) -> str:
    if node is None:
        return "input"
    if isinstance(node, dict):
        return node.get("type", "default")
    return type(node).__name__

def _node_short(name: str, node):
    """Return (top_line, sub_line)."""
    if node is None:
        return ("input", "")
    if isinstance(node, dict):
        t = node.get("type", "Unknown")
        if t == "MatMulNode":
            return ("MatMul", f"{node.get('in_features','?')}→{node.get('out_features','?')}")
        if t == "ActivationNode":
            return (node.get("activation_type", "act"), "activation")
        if t == "ReshapeNode":
            return ("Reshape", str(node.get("operation", ""))[:12])
        if t == "AddNode":
            return ("Add", "")
        if t == "ConcatNode":
            return ("Concat", "")
        if t == "ReduceNode":
            return ("Reduce", str(node.get("operation", ""))[:12])
        if t == "IndexNode":
            return ("Index", str(node.get("mode", ""))[:12])
        return (t[:12], "")
    t = type(node).__name__
    if t == "MatMulNode":
        return ("MatMul", f"{node.W.shape[0]}→{node.W.shape[1]}")
    if t == "ActivationNode":
        return (node.activation_type, "activation")
    if t == "ReshapeNode":
        return ("Reshape", node.operation[:12])
    if t == "AddNode":
        return ("Add", "")
    if t == "ConcatNode":
        return ("Concat", "")
    if t == "ReduceNode":
        return ("Reduce", node.operation[:12])
    if t == "IndexNode":
        return ("Index", node.mode[:12])
    return (t[:12], "")


# ── bezier helpers ────────────────────────────────────────────────────────────

def _bez(p0, p1, p2, p3, t):
    mt = 1 - t
    return (
        mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0],
        mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1],
    )

def _bez_pts(p0, c1, c2, p3, steps=32):
    pts = []
    for i in range(steps + 1):
        pts.extend(_bez(p0, c1, c2, p3, i / steps))
    return pts


# ── layout ────────────────────────────────────────────────────────────────────

def _layout(model, W, H):
    names = ["input"] + list(model.execution_order)
    n = len(names)
    pad_x, pad_top, pad_bot = 90, 70, 70
    usable = H - pad_top - pad_bot
    step = usable / max(n - 1, 1)
    pos = {name: [W / 2, pad_top + i * step] for i, name in enumerate(names)}

    for name in model.execution_order:
        node = model.nodes[name]
        inputs = node.get("inputs", []) if isinstance(node, dict) else node.inputs
        if inputs:
            xs = [pos[s][0] for s in inputs if s in pos]
            if xs:
                pos[name][0] = sum(xs) / len(xs)

    for _ in range(12):
        for i, ni in enumerate(names):
            for nj in names[i+1:]:
                xi, yi = pos[ni]; xj, yj = pos[nj]
                if abs(yi - yj) < 4 and abs(xi - xj) < 96:
                    mid = (xi + xj) / 2
                    pos[ni][0] = mid - 52
                    pos[nj][0] = mid + 52

    for v in pos.values():
        v[0] = max(pad_x, min(W - pad_x, v[0]))

    return {k: (v[0], v[1]) for k, v in pos.items()}


# ── particles ─────────────────────────────────────────────────────────────────

class Particle:
    __slots__ = ("t", "speed", "key")
    def __init__(self, key, offset=0.0):
        self.key   = key
        self.t     = offset
        self.speed = 0.006 + 0.002 * (abs(hash(key)) % 5) / 5


class ModelView:
    def __init__(self, data: dict):
        self.execution_order = list(data.get("execution_order", []))
        self.nodes = dict(data.get("nodes", {}))
        self._last_mutation = data.get("last_mutation", None)


def describe_architecture_serialized(model: ModelView):
    parts = []
    for name in model.execution_order:
        node = model.nodes[name]
        t = node.get("type", "Unknown")
        if t == "ReshapeNode":
            parts.append(f"{name}:Reshape({node.get('operation', '')})")
        elif t == "MatMulNode":
            parts.append(f"{name}:MatMul({node.get('in_features','?')}->{node.get('out_features','?')})")
        elif t == "ActivationNode":
            parts.append(f"{name}:Act({node.get('activation_type', '')})")
        elif t == "AddNode":
            parts.append(f"{name}:Add")
        elif t == "ConcatNode":
            parts.append(f"{name}:Concat(dim={node.get('dim', '')})")
        elif t == "ReduceNode":
            parts.append(f"{name}:Reduce({node.get('operation', '')})")
        elif t == "IndexNode":
            parts.append(f"{name}:Index({node.get('mode', '')})")
        else:
            parts.append(f"{name}:{t}")
    return " -> ".join(parts)


def _find_latest_log(output_dir="output"):
    if not os.path.isdir(output_dir):
        return None
    candidates = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.lower().endswith(".json")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_stats(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gens = data.get("generations", [])
    stats = []
    for g in gens:
        best_model = ModelView(g["best_model"])
        max_nodes_model = ModelView(g.get("max_nodes_model", g["best_model"]))
        stats.append(
            {
                "generation": g.get("generation"),
                "best_acc": g.get("best_acc", 0.0),
                "avg_acc": g.get("avg_acc", 0.0),
                "best_params": g.get("best_params", 0),
                "best_nodes": g.get("best_nodes", 0),
                "max_nodes": g.get("max_nodes", 0),
                "train_steps": g.get("train_steps", 0),
                "train_steps_max": g.get("train_steps_max", 0),
                "best_model": best_model,
                "max_nodes_model": max_nodes_model,
                "mutation": g.get("best_model", {}).get("last_mutation"),
            }
        )
    return stats


# ── app ───────────────────────────────────────────────────────────────────────

class EvolutionViz:
    R      = 22   # node radius
    SHADOW = 3    # drop-shadow offset

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Model Evolution")
        self.root.configure(bg=BG)
        self.root.geometry("1280x740")
        self.root.minsize(900, 560)

        self._q: queue.Queue    = queue.Queue()
        self._stats: Optional[dict] = None
        self._model             = None
        self._pos: dict         = {}
        self._beziers: dict     = {}
        self._particles: list   = []
        self._tick              = 0
        self._fresh             = False
        self._acc_hist: deque   = deque(maxlen=100)
        self._param_hist: deque = deque(maxlen=100)
        self._playback: list    = []
        self._play_idx          = 0
        self._play_interval     = 0.9
        self._next_play         = 0.0

        self._build_ui()
        self._loop()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, minsize=270, weight=0)
        self.root.rowconfigure(0, weight=1)

        self.cv = tk.Canvas(self.root, bg=CANVAS_BG,
                            highlightthickness=0, bd=0)
        self.cv.grid(row=0, column=0, sticky="nsew",
                     padx=(12, 6), pady=12)

        pnl = tk.Frame(self.root, bg=PANEL_BG, bd=0)
        pnl.grid(row=0, column=1, sticky="nsew",
                 padx=(6, 12), pady=12)
        pnl.columnconfigure(0, weight=1)
        self._build_panel(pnl)

    def _build_panel(self, pnl):
        row = 0
        tk.Label(pnl, text="Evolution", font=_fam(15, True),
                 fg=TEXT_DK, bg=PANEL_BG
                 ).grid(row=row, column=0, sticky="w",
                        padx=18, pady=(18, 2)); row += 1

        self._gen_var = tk.StringVar(value="Generation —")
        tk.Label(pnl, textvariable=self._gen_var, font=_fam(9),
                 fg=TEXT_LT, bg=PANEL_BG
                 ).grid(row=row, column=0, sticky="w",
                        padx=18, pady=(0, 14)); row += 1

        self._divider(pnl, row); row += 1

        self._metric_vars = {}
        for key, label in [
            ("best_acc",    "Best accuracy"),
            ("avg_acc",     "Avg accuracy"),
            ("best_params", "Parameters"),
            ("best_nodes",  "Nodes"),
            ("max_nodes",   "Max nodes"),
            ("train_steps", "Train steps"),
            ("mutation",    "Last mutation"),
        ]:
            f = tk.Frame(pnl, bg=PANEL_BG)
            f.grid(row=row, column=0, sticky="ew", padx=18, pady=3)
            f.columnconfigure(0, weight=1)
            tk.Label(f, text=label, font=_fam(8),
                     fg=TEXT_MD, bg=PANEL_BG, anchor="w"
                     ).grid(row=0, column=0, sticky="w")
            var = tk.StringVar(value="—")
            self._metric_vars[key] = var
            tk.Label(f, textvariable=var, font=_fam(9, True),
                     fg=TEXT_DK, bg=PANEL_BG, anchor="e"
                     ).grid(row=0, column=1, sticky="e")
            row += 1

        self._divider(pnl, row); row += 1

        for title, attr, color in [
            ("Accuracy",   "spark_acc",    "#3182ce"),
            ("Parameters", "spark_params", "#805ad5"),
        ]:
            tk.Label(pnl, text=title, font=_fam(8), fg=TEXT_LT,
                     bg=PANEL_BG
                     ).grid(row=row, column=0, sticky="w",
                            padx=18); row += 1
            sp = tk.Canvas(pnl, bg=PANEL_BG, height=52,
                           highlightthickness=0)
            sp.grid(row=row, column=0, sticky="ew",
                    padx=18, pady=(0, 12)); row += 1
            setattr(self, attr, sp)

        self._divider(pnl, row); row += 1
        tk.Label(pnl, text="Architecture", font=_fam(8),
                 fg=TEXT_LT, bg=PANEL_BG
                 ).grid(row=row, column=0, sticky="w",
                        padx=18, pady=(8, 2)); row += 1
        self._arch_var = tk.StringVar(value="")
        tk.Label(pnl, textvariable=self._arch_var, font=_fam(8),
                 fg=TEXT_MD, bg=PANEL_BG, wraplength=220,
                 justify="left", anchor="nw"
                 ).grid(row=row, column=0, sticky="w",
                        padx=18, pady=(0, 18))

    def _divider(self, parent, row):
        tk.Frame(parent, bg=RULE, height=1
                 ).grid(row=row, column=0, sticky="ew",
                        padx=12, pady=4)

    # ── main loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        if self._playback and self._play_idx < len(self._playback):
            now = time.monotonic()
            if now >= self._next_play:
                self._apply_stats(self._playback[self._play_idx])
                self._play_idx += 1
                self._next_play = now + self._play_interval

        self._fresh = False
        try:
            while True:
                stats = self._q.get_nowait()
                self._apply_stats(stats)
                self._fresh = True
        except queue.Empty:
            pass

        self._tick += 1
        self._step_particles()
        self._draw()
        self.root.after(28, self._loop)

    # ── stats ─────────────────────────────────────────────────────────────────

    def _apply_stats(self, stats):
        self._stats = stats
        model = stats["best_model"]
        self._model = model

        W = self.cv.winfo_width() or 960
        H = self.cv.winfo_height() or 700
        self._pos = _layout(model, W, H)
        self._rebuild_edges(model, W, H)

        self._gen_var.set(f"Generation {stats.get('generation', '—')}")
        mv = self._metric_vars
        mv["best_acc"].set(f"{stats.get('best_acc', 0):.4f}")
        mv["avg_acc"].set(f"{stats.get('avg_acc', 0):.4f}")
        mv["best_params"].set(f"{stats.get('best_params', 0):,}")
        mv["best_nodes"].set(str(stats.get("best_nodes", "—")))
        mv["max_nodes"].set(str(stats.get("max_nodes", "—")))
        steps = stats.get("train_steps", 0)
        steps_max = stats.get("train_steps_max", 0)
        mv["train_steps"].set(f"{steps}/{steps_max}" if steps_max else str(steps))
        mutation = stats.get("mutation", getattr(model, "_last_mutation", "n/a"))
        mv["mutation"].set(str(mutation)[:24])
        self._arch_var.set(describe_architecture_serialized(model)[:200])

        self._acc_hist.append(stats.get("best_acc", 0))
        self._param_hist.append(stats.get("best_params", 0))
        self._redraw_sparkline(self.spark_acc,    self._acc_hist,   "#3182ce")
        self._redraw_sparkline(self.spark_params, self._param_hist, "#805ad5")

    def _rebuild_edges(self, model, W, H):
        self._beziers.clear()
        self._particles.clear()
        for name in model.execution_order:
            node = model.nodes[name]
            inputs = node.get("inputs", []) if isinstance(node, dict) else node.inputs
            for src in inputs:
                x1, y1 = self._pos.get(src, (W/2, 0))
                x2, y2 = self._pos.get(name, (W/2, H))
                dy = abs(y2 - y1) * 0.5
                key = (src, name)
                self._beziers[key] = (
                    (x1, y1), (x1, y1 + dy), (x2, y2 - dy), (x2, y2)
                )
                for i in range(2):
                    self._particles.append(Particle(key, offset=i * 0.5))

    # ── particles ─────────────────────────────────────────────────────────────

    def _step_particles(self):
        for p in self._particles:
            p.t += p.speed
            if p.t > 1.0:
                p.t -= 1.0

    # ── draw ──────────────────────────────────────────────────────────────────

    def _draw(self):
        cv = self.cv
        cv.delete("all")
        W = cv.winfo_width();  H = cv.winfo_height()
        if W < 20 or H < 20:
            return

        if self._model is None:
            cv.create_text(W//2, H//2,
                           text="Waiting for first generation…",
                           fill=TEXT_LT, font=_fam(13))
            return

        model = self._model

        # Re-layout on resize
        if self._pos:
            ox = max(v[0] for v in self._pos.values())
            oy = max(v[1] for v in self._pos.values())
            if abs(ox - W) > 60 or abs(oy - H) > 60:
                self._pos = _layout(model, W, H)
                self._rebuild_edges(model, W, H)

        # Edges
        for key, bez in self._beziers.items():
            pts = _bez_pts(*bez, steps=30)
            cv.create_line(*pts, fill=EDGE_COL, width=1, smooth=False)

        # Particles
        for p in self._particles:
            bez = self._beziers.get(p.key)
            if bez is None:
                continue
            hx, hy = _bez(*bez, p.t)
            # Tail
            for ti in range(1, 7):
                tt = p.t - ti * 0.028
                if tt < 0:
                    continue
                tx, ty = _bez(*bez, tt)
                a  = (7 - ti) / 7
                pr, pg, pb = 0x2b, 0x6c, 0xb0
                er, eg, eb = 0xcb, 0xd5, 0xe0
                rc = int(pr*a + er*(1-a))
                gc = int(pg*a + eg*(1-a))
                bc = int(pb*a + eb*(1-a))
                r2 = max(1, int(3 * a))
                cv.create_oval(tx-r2, ty-r2, tx+r2, ty+r2,
                               fill=f"#{rc:02x}{gc:02x}{bc:02x}", outline="")
            # Head
            cv.create_oval(hx-4, hy-4, hx+4, hy+4,
                           fill=PARTICLE, outline="")

        # Nodes
        names  = ["input"] + list(model.execution_order)
        newest = model.execution_order[-1] if model.execution_order else None
        pulse  = 0.5 + 0.5 * math.sin(self._tick * 0.12)

        for name in names:
            node = None if name == "input" else model.nodes[name]
            cx, cy = self._pos.get(name, (W/2, H/2))
            key   = _node_type_key(node)
            fill, stroke = NODE_PALETTE.get(key, NODE_PALETTE["default"])
            R     = self.R
            is_new = (name == newest) and self._fresh

            # Drop shadow
            s = self.SHADOW
            cv.create_oval(cx-R+s, cy-R+s, cx+R+s, cy+R+s,
                           fill="#dde3ed", outline="")

            # Pulsing outer ring on newest node
            ring_r = R + (3 + int(4 * pulse) if is_new else 3)
            ring_w = 2 if is_new else 1
            ring_d = () if is_new else (3, 3)
            cv.create_oval(cx-ring_r, cy-ring_r, cx+ring_r, cy+ring_r,
                           fill="", outline=stroke, width=ring_w, dash=ring_d)

            # Main circle
            cv.create_oval(cx-R, cy-R, cx+R, cy+R,
                           fill=fill, outline=stroke, width=2)

            # Soma dot
            cv.create_oval(cx-5, cy-5, cx+5, cy+5,
                           fill=stroke, outline="")

            # Labels above node
            top, sub = _node_short(name, node)
            cv.create_text(cx, cy - R - 9, text=top,
                           fill=TEXT_DK, font=_fam(8, True), anchor="s")
            if sub:
                cv.create_text(cx, cy - R - 1, text=sub,
                               fill=TEXT_LT, font=_fam(7), anchor="n")

        # Generation badge
        gen = self._stats.get("generation", "?") if self._stats else "?"
        bx, by = W - 16, 16
        cv.create_rectangle(bx-52, by-11, bx+2, by+11,
                            fill=PANEL_BG, outline=RULE)
        cv.create_text(bx-25, by, text=f"gen  {gen}",
                       fill=TEXT_MD, font=_fam(8), anchor="center")

    # ── sparklines ────────────────────────────────────────────────────────────

    def _redraw_sparkline(self, sp: tk.Canvas, data: deque, color: str):
        sp.delete("all")
        sp.update_idletasks()
        W = sp.winfo_width() or 220
        H = sp.winfo_height() or 52
        vals = list(data)
        if len(vals) < 2:
            return
        lo, hi = min(vals), max(vals)
        span   = hi - lo or 1e-9
        n      = len(vals)
        xs = [int(i / (n-1) * (W-4)) + 2 for i in range(n)]
        ys = [int((1 - (v-lo)/span) * (H-12)) + 6 for v in vals]

        # Area fill
        poly = [2, H] + [c for xy in zip(xs, ys) for c in xy] + [W-2, H]
        r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        fc = (f"#{int(r*.08+0xf0*.92):02x}"
              f"{int(g*.08+0xf2*.92):02x}"
              f"{int(b*.08+0xf5*.92):02x}")
        sp.create_polygon(*poly, fill=fc, outline="")

        # Line
        pts = [c for xy in zip(xs, ys) for c in xy]
        sp.create_line(*pts, fill=color, width=1.5, smooth=True)

        # End dot
        sp.create_oval(xs[-1]-3, ys[-1]-3, xs[-1]+3, ys[-1]+3,
                       fill=color, outline="")

        # Value
        sp.create_text(W-2, 2, text=f"{vals[-1]:.4g}",
                       fill=color, font=_fam(7, True), anchor="ne")

    # ── public callback (from evolution thread) ────────────────────────────────

    def on_generation(self, stats: dict):
        self._q.put(stats)

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self):
        path = sys.argv[1] if len(sys.argv) > 1 else _find_latest_log()
        if path is None:
            self._gen_var.set("No JSON log found")
        else:
            self._playback = _load_stats(path)
            self._play_idx = 0
            self._next_play = time.monotonic()
        self.root.mainloop()


if __name__ == "__main__":
    EvolutionViz().run()
