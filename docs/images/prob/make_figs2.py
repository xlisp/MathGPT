import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25,
})
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Fig 8: history timeline of probability -> large models
# ---------------------------------------------------------------------------
events = [
    (1654, "Pascal & Fermat\nclassical prob.\n(gambling)", +1),
    (1713, "Bernoulli\nLaw of\nLarge Numbers", -1),
    (1763, "Bayes\nP(A|B)", +1),
    (1809, "Gauss / Laplace\nNormal dist.\n+ least squares", -1),
    (1906, "Markov\nMarkov chains\n(LM ancestor)", +1),
    (1933, "Kolmogorov\naxioms\n(measure theory)", -1),
    (1948, "Shannon\nentropy /\ncross-entropy", +1),
    (1988, "Pearl\nBayes nets /\ncausality", -1),
    (2017, "Transformer\nP(next|context)", +1),
    (2022, "ChatGPT\nRLHF / RL", -1),
]
fig, ax = plt.subplots(figsize=(15, 5.2))
ax.axis("off")
xs = [e[0] for e in events]
ax.plot([min(xs)-8, max(xs)+8], [0, 0], color="#333", lw=2, zorder=1)
for yr, label, side in events:
    ax.scatter([yr], [0], s=90, color="#1971c2", zorder=3)
    y = 0.55 * side
    ax.annotate(label, (yr, 0), xytext=(yr, y), ha="center",
                va="bottom" if side > 0 else "top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#e7f5ff", ec="#1971c2"),
                arrowprops=dict(arrowstyle="-", color="#1971c2", lw=1))
    ax.text(yr, -0.07*side, str(yr), ha="center",
            va="top" if side > 0 else "bottom", fontsize=8, color="#555")
ax.set_xlim(min(xs)-12, max(xs)+12)
ax.set_ylim(-1.1, 1.1)
ax.set_title("400 years of probability: from the gambling table to ChatGPT",
             fontsize=14, pad=12)
fig.savefig(os.path.join(OUT, "fig8_history.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 9: Central Limit Theorem — sums of ANY distribution -> Gaussian
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
N = 200000
sources = {
    "Uniform": lambda n: rng.uniform(0, 1, (N, n)),
    "Exponential": lambda n: rng.exponential(1.0, (N, n)),
    "Bernoulli(0.3)": lambda n: (rng.random((N, n)) < 0.3).astype(float),
}
ns = [1, 2, 30]
fig, axes = plt.subplots(3, 3, figsize=(13, 8.5))
for r, (name, draw) in enumerate(sources.items()):
    for c, n in enumerate(ns):
        s = draw(n).mean(axis=1)
        s = (s - s.mean()) / s.std()
        ax = axes[r, c]
        ax.hist(s, bins=80, density=True, color="#74b9ff", alpha=0.8)
        xx = np.linspace(-4, 4, 200)
        ax.plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), "r-", lw=2)
        ax.set_title(f"{name}, avg of n={n}", fontsize=10)
        ax.set_xlim(-4, 4); ax.set_yticks([])
fig.suptitle("Central Limit Theorem: average enough of ANYTHING -> the same bell curve",
             fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig9_clt.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 10: Bayesian network (DAG) + ladder of causation
# ---------------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 4.8),
                               gridspec_kw={"width_ratios": [1.1, 1]})

# Left: a small causal DAG
axL.axis("off"); axL.set_xlim(0, 10); axL.set_ylim(0, 10)
nodes = {
    "Season": (2, 8),
    "Sprinkler": (1, 4.5),
    "Rain": (5, 6),
    "Wet grass": (3.5, 2),
}
pos = {}
for name, (x, y) in nodes.items():
    c = Circle((x, y), 1.05, fc="#fff3bf", ec="#e8590c", lw=1.8, zorder=2)
    axL.add_patch(c); axL.text(x, y, name, ha="center", va="center", fontsize=9.5, zorder=3)
    pos[name] = (x, y)
edges = [("Season", "Sprinkler"),
         ("Season", "Rain"),
         ("Sprinkler", "Wet grass"),
         ("Rain", "Wet grass")]
for a, b in edges:
    x1, y1 = pos[a]; x2, y2 = pos[b]
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                          shrinkA=34, shrinkB=34, color="#e8590c", lw=2, zorder=1)
    axL.add_patch(arr)
axL.set_title("Bayesian network: a DAG of conditional dependencies\n"
              "P(S,Sp,R,W)=P(S)P(Sp|S)P(R|S)P(W|Sp,R)", fontsize=11)

# Right: ladder of causation
axR.axis("off"); axR.set_xlim(0, 10); axR.set_ylim(0, 10)
rungs = [
    (1.5, "1. Association  P(y|x)", "'seeing x, prob of y?'  observe / correlate\nTransformer stops on THIS rung", "#74b9ff"),
    (5.0, "2. Intervention  P(y|do(x))", "'force-change x, then y?'  intervene\nRLHF / A-B test", "#ffd43b"),
    (8.3, "3. Counterfactual  P(y_x|x',y')", "'what if it had been otherwise?'\nhuman reflection & attribution", "#ff8787"),
]
for y, title, sub, col in rungs:
    axR.add_patch(plt.Rectangle((0.5, y-1.0), 9, 1.9, fc=col, alpha=0.35,
                                ec=col, lw=1.5))
    axR.text(0.8, y+0.4, title, fontsize=11, weight="bold", va="center")
    axR.text(0.8, y-0.45, sub, fontsize=8.5, va="center", color="#333")
for y0, y1 in [(2.4, 4.0), (5.9, 7.3)]:
    axR.add_patch(FancyArrowPatch((9.6, y0), (9.6, y1), arrowstyle="-|>",
                  mutation_scale=16, color="#333", lw=1.5))
axR.set_title("Pearl's Ladder of Causation\n(why correlation ≠ causation)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig10_causal.png"))
plt.close(fig)

print("done:", sorted(p for p in os.listdir(OUT) if p.startswith(("fig8","fig9","fig10"))))
