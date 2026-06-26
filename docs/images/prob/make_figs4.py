import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import exp, factorial

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25,
})
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# ===========================================================================
# Fig 12: WHY softmax? same logits under different normalizations
# ===========================================================================
z = np.array([2.0, 1.0, 0.2, -1.0])
labels = ["A", "B", "C", "D"]

# (1) argmax / hardmax -> one-hot
hard = np.zeros_like(z); hard[np.argmax(z)] = 1.0
# (2) proportional / normalize: must clip negatives first -> info lost
relu = np.clip(z, 0, None); prop = relu / relu.sum()
# (3) softmax
e = np.exp(z - z.max()); soft = e / e.sum()

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))

axes[0].bar(labels, z, color="#868e96")
axes[0].axhline(0, color="#333", lw=0.8)
axes[0].set_title("raw logits z\n(can be negative, unbounded)")

axes[1].bar(labels, hard, color="#fa5252")
axes[1].set_title("argmax -> one-hot\n✗ no gradient: can't train")
axes[1].set_ylim(0, 1)

axes[2].bar(labels, prop, color="#fab005")
axes[2].set_title("clip(z,0)/sum\n✗ throws away negatives, ✗ kinks")
axes[2].set_ylim(0, 1)

axes[3].bar(labels, soft, color="#2f9e44")
axes[3].set_title("softmax = exp(z)/Σexp(z)\n✓ smooth ✓ any real ✓ log-linear")
axes[3].set_ylim(0, 1)
for i, v in enumerate(soft):
    axes[3].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

fig.suptitle("Why softmax and not something else? Same 4 logits, four ways to make a distribution",
             y=1.04, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig12_softmax_why.png"))
plt.close(fig)

# ===========================================================================
# Fig 13: Poisson origin — Bortkiewicz Prussian horse-kick deaths (1898)
# ===========================================================================
# Classic data: deaths per corps-year, 280 corps-years total
ks       = np.array([0, 1, 2, 3, 4])
observed = np.array([144, 91, 32, 11, 2])      # actual counts
total    = observed.sum()
lam      = (ks * observed).sum() / total        # MLE: lambda = mean = 0.70
expected = np.array([total * exp(-lam) * lam**k / factorial(k) for k in ks])

fig, ax = plt.subplots(figsize=(8.8, 4.6))
w = 0.4
ax.bar(ks - w/2, observed, width=w, color="#1971c2", label="observed (real army records)")
ax.bar(ks + w/2, expected, width=w, color="#f08c00", alpha=0.85,
       label=f"Poisson(λ={lam:.2f}) prediction")
for k, o, e in zip(ks, observed, expected):
    ax.text(k - w/2, o + 2, str(o), ha="center", fontsize=8.5, color="#1971c2")
    ax.text(k + w/2, e + 2, f"{e:.0f}", ha="center", fontsize=8.5, color="#d9480f")
ax.set_xlabel("soldiers killed by horse kick, per cavalry corps per year")
ax.set_ylabel("number of corps-years (out of 280)")
ax.set_title("Why Poisson exists: rare independent events.\n"
             "Bortkiewicz 1898 — Prussian horse-kick deaths fit Poisson almost perfectly")
ax.set_xticks(ks)
ax.legend()
fig.savefig(os.path.join(OUT, "fig13_poisson_horsekick.png"))
plt.close(fig)

print("done:", [f for f in sorted(os.listdir(OUT)) if f.startswith(("fig12","fig13"))])
