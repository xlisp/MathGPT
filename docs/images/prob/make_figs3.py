import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import beta as beta_dist

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 11,
})
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

fig, (axT, axB) = plt.subplots(1, 2, figsize=(15, 5.6),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ---------------------------------------------------------------------------
# Left: debugging version tree (jim0 -> jim1 -> ... with rollback)
# ---------------------------------------------------------------------------
axT.axis("off"); axT.set_xlim(0, 12); axT.set_ylim(0, 10)

def node(ax, x, y, label, belief, status):
    color = {"ok": "#b2f2bb", "fail": "#ffc9c9", "now": "#a5d8ff"}[status]
    edge  = {"ok": "#2b8a3e", "fail": "#c92a2a", "now": "#1971c2"}[status]
    box = FancyBboxPatch((x-0.78, y-0.42), 1.56, 0.84,
                         boxstyle="round,pad=0.04", fc=color, ec=edge, lw=1.8, zorder=3)
    ax.add_patch(box)
    ax.text(x, y+0.12, label, ha="center", va="center", fontsize=10, weight="bold", zorder=4)
    ax.text(x, y-0.22, belief, ha="center", va="center", fontsize=8, color="#444", zorder=4)
    return (x, y)

def edge(ax, p, q, color="#495057", style="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15,
                 shrinkA=22, shrinkB=22, color=color, lw=1.8,
                 linestyle=style, zorder=2))

# committed backbone
n0 = node(axT, 1.4, 5, "jim0", "✓ baseline", "now")
# direction A attempts
nA1 = node(axT, 4.2, 7.4, "jim1", "P(A)=.6", "fail")
nA2 = node(axT, 7.0, 7.4, "jim2", "P(A)=.35", "fail")
# direction B attempt (after abandoning A)
nB1 = node(axT, 4.2, 2.6, "jim1'", "P(B)=.3", "fail")
nB2 = node(axT, 7.0, 4.2, "jim2'", "P(B)=.55", "ok")
nB3 = node(axT, 9.9, 4.2, "jim3 ✓", "BREAK!", "ok")

edge(axT, n0, nA1)
edge(axT, nA1, nA2)
edge(axT, nA2, n0, color="#c92a2a", style="--")   # rollback A
edge(axT, n0, nB1)
edge(axT, nB1, n0, color="#c92a2a", style="--")    # rollback B'
edge(axT, n0, nB2, color="#2b8a3e")
edge(axT, nB2, nB3, color="#2b8a3e")

axT.text(4.2, 8.5, "direction A: belief drops after each fail -> abandon",
         ha="center", fontsize=8.5, color="#c92a2a")
axT.text(7.5, 3.0, "direction B: belief rises -> commit & break through",
         ha="center", fontsize=8.5, color="#2b8a3e")
axT.text(1.4, 6.3, "rollback\n(git reset)", ha="center", fontsize=8, color="#c92a2a")
axT.set_title("Debugging as Bayesian search:\nbranch, test, update belief, roll back, re-allocate",
              fontsize=12)

# ---------------------------------------------------------------------------
# Right: Beta posterior = quantified "belief in direction A succeeding"
# ---------------------------------------------------------------------------
x = np.linspace(0, 1, 400)
curves = [
    (1, 1,  "#adb5bd", "prior Beta(1,1): no idea"),
    (1, 4,  "#c92a2a", "after 3 fails Beta(1,4): belief collapses"),
    (5, 2,  "#2b8a3e", "after 4 wins/1 fail Beta(5,2): belief rises"),
]
for a, b, c, lab in curves:
    axB.plot(x, beta_dist.pdf(x, a, b), color=c, lw=2.4, label=lab)
    mean = a / (a + b)
    axB.axvline(mean, color=c, ls=":", lw=1)
axB.set_xlabel("p = belief that this direction will succeed")
axB.set_ylabel("posterior density")
axB.set_title("Quantify your belief: Beta posterior\nupdated by every success / failure", fontsize=12)
axB.legend(fontsize=8.5, loc="upper center")
axB.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig11_belief.png"))
plt.close(fig)
print("done: fig11_belief.png")
