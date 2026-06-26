import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

def softmax(x, t=1.0):
    x = np.asarray(x, dtype=float) / t
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

# ---------------------------------------------------------------------------
# Fig 1: softmax temperature — logits -> probability distribution
# ---------------------------------------------------------------------------
toks = ["72", "12", "60", "cat", "the", "0", "and", "?"]
logits = np.array([8.0, 6.5, 5.0, 1.0, 3.0, 4.0, 2.5, 0.5])

fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
for ax, t in zip(axes, [0.3, 0.6, 1.0, 2.0]):
    p = softmax(logits, t)
    colors = plt.cm.viridis(p / p.max())
    ax.bar(toks, p, color=colors)
    ax.set_title(f"temperature = {t}")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=0)
axes[0].set_ylabel("probability")
fig.suptitle("softmax(logits / T): same logits, different 'confidence'", y=1.03, fontsize=13)
fig.savefig(os.path.join(OUT, "fig1_temperature.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2: top-k truncation
# ---------------------------------------------------------------------------
np.random.seed(0)
vocab = 60
big_logits = np.sort(np.random.randn(vocab) * 1.5)[::-1]
big_logits[0:3] += np.array([4.0, 2.5, 1.6])
p_full = softmax(big_logits, 1.0)

k = 8
idx = np.argsort(big_logits)[::-1]
keep = idx[:k]
p_topk = np.zeros_like(big_logits)
masked = big_logits.copy()
masked[idx[k:]] = -1e9
p_topk = softmax(masked, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
axes[0].bar(range(vocab), p_full, color="#888")
axes[0].set_title(f"full distribution over {vocab} tokens")
axes[0].set_xlabel("token id (sorted by logit)")
axes[0].set_ylabel("probability")

bar_colors = ["#2c7fb8" if i < k else "#e0e0e0" for i in range(vocab)]
axes[1].bar(range(vocab), p_topk, color=bar_colors)
axes[1].axvline(k - 0.5, color="crimson", ls="--", lw=1.5)
axes[1].set_title(f"top-k = {k}: tail zeroed out, renormalized")
axes[1].set_xlabel("token id (sorted by logit)")
fig.suptitle("top-k sampling: cut the long tail of nonsense tokens", y=1.04, fontsize=13)
fig.savefig(os.path.join(OUT, "fig2_topk.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3: cross entropy = -log(p) = surprise
# ---------------------------------------------------------------------------
p = np.linspace(0.001, 1.0, 500)
loss = -np.log(p)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(p, loss, color="#d6336c", lw=2.5)
for pv in [0.9, 0.5, 0.1, 0.02]:
    ax.plot([pv, pv], [0, -np.log(pv)], color="#888", ls=":", lw=1)
    ax.scatter([pv], [-np.log(pv)], color="#d6336c", zorder=5)
    ax.annotate(f"p={pv}\nloss={-np.log(pv):.2f}", (pv, -np.log(pv)),
                textcoords="offset points", xytext=(8, 6), fontsize=9)
ax.set_xlabel("probability the model assigned to the CORRECT next token")
ax.set_ylabel("cross-entropy loss  =  -log(p)")
ax.set_title("Loss is just 'surprise': confident & right -> ~0,  confident & wrong -> huge")
ax.set_ylim(0, 6)
fig.savefig(os.path.join(OUT, "fig3_cross_entropy.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 4: conditional probability chain (autoregressive)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 3.2))
ctx = ["<bos>", "Natalia", "sold", "48", "clips", "in", "May", "and"]
steps = list(range(len(ctx) + 1))
ax.axis("off")
x = 0
for i, w in enumerate(ctx):
    ax.add_patch(plt.Rectangle((x, 0.4), 1.4, 0.6, fc="#dbeafe", ec="#2563eb"))
    ax.text(x + 0.7, 0.7, w, ha="center", va="center", fontsize=10)
    x += 1.6
ax.add_patch(plt.Rectangle((x, 0.4), 1.4, 0.6, fc="#fde68a", ec="#d97706"))
ax.text(x + 0.7, 0.7, "?", ha="center", va="center", fontsize=14, weight="bold")
ax.annotate("", xy=(x, 0.35), xytext=(0.7, 0.35),
            arrowprops=dict(arrowstyle="->", color="#d97706", lw=2,
                            connectionstyle="arc3,rad=-0.18"))
ax.text(x / 2 + 0.7, -0.15,
        r"P( next token  |  all previous tokens )",
        ha="center", fontsize=13, color="#d97706")
ax.set_xlim(-0.3, x + 2)
ax.set_ylim(-0.4, 1.2)
ax.set_title("A transformer is one giant conditional-probability machine", fontsize=13)
fig.savefig(os.path.join(OUT, "fig4_conditional.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 5: REINFORCE advantage = reward - mean(reward)
# ---------------------------------------------------------------------------
np.random.seed(3)
rewards = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=float)
mean_r = rewards.mean()
adv = rewards - mean_r
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
xs = np.arange(len(rewards))
axes[0].bar(xs, rewards, color=["#2b8a3e" if r > 0 else "#c92a2a" for r in rewards])
axes[0].axhline(mean_r, color="#1971c2", ls="--", lw=2, label=f"baseline mean = {mean_r:.2f}")
axes[0].set_title("8 sampled answers to ONE math problem")
axes[0].set_xlabel("rollout #")
axes[0].set_ylabel("reward (1=correct, 0=wrong)")
axes[0].legend()
axes[1].bar(xs, adv, color=["#2b8a3e" if a > 0 else "#c92a2a" for a in adv])
axes[1].axhline(0, color="#333", lw=1)
axes[1].set_title("advantage = reward - baseline  ->  push up green, push down red")
axes[1].set_xlabel("rollout #")
axes[1].set_ylabel("advantage")
fig.suptitle("REINFORCE: 'be more like the answers that beat the average'", y=1.04, fontsize=13)
fig.savefig(os.path.join(OUT, "fig5_advantage.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 6: Poisson — WeChat messages per hour
# ---------------------------------------------------------------------------
from math import exp, factorial
def poisson(lmbda, kmax):
    ks = np.arange(0, kmax)
    return ks, np.array([exp(-lmbda) * lmbda**k / factorial(k) for k in ks])

fig, ax = plt.subplots(figsize=(8.5, 4.3))
for lmbda, c in [(2, "#1971c2"), (5, "#2b8a3e"), (10, "#d6336c")]:
    ks, ps = poisson(lmbda, 25)
    ax.plot(ks, ps, "o-", color=c, label=f"λ = {lmbda} msgs/hour")
ax.set_xlabel("number of WeChat messages in the next hour (k)")
ax.set_ylabel("P(X = k)")
ax.set_title("Poisson distribution: how many messages will the group send this hour?")
ax.legend()
fig.savefig(os.path.join(OUT, "fig6_poisson.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 7: debugging transformer via per-token entropy
# ---------------------------------------------------------------------------
np.random.seed(11)
positions = np.arange(40)
# simulate: low entropy on copied/forced tokens, spikes at reasoning decision points
ent = 0.4 + 0.3 * np.abs(np.random.randn(40))
ent[[5, 6, 18, 27]] += np.array([2.6, 2.2, 2.9, 1.8])  # decision points
ent[30:36] = 0.05  # forced calculator output tokens
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(positions, ent, color="#5f3dc4", lw=1.8)
ax.fill_between(positions, ent, color="#5f3dc4", alpha=0.15)
ax.axhspan(0, 0.2, color="#2b8a3e", alpha=0.08)
for px in [5, 6, 18, 27]:
    ax.annotate("high entropy:\nmodel 'unsure'", (px, ent[px]),
                textcoords="offset points", xytext=(2, 6), fontsize=8, color="#d6336c")
ax.annotate("entropy ~0:\nforced tool output", (32, 0.05),
            textcoords="offset points", xytext=(-10, 30), fontsize=8, color="#2b8a3e",
            arrowprops=dict(arrowstyle="->", color="#2b8a3e"))
ax.set_xlabel("token position in the generated answer")
ax.set_ylabel("entropy of next-token distribution (nats)")
ax.set_title("Debugging a transformer: plot per-token entropy to SEE where it hesitates")
fig.savefig(os.path.join(OUT, "fig7_entropy.png"))
plt.close(fig)

print("done:", sorted(os.listdir(OUT)))
