import matplotlib.pyplot as plt
import numpy as np

# Strategic Reward Data - Representative trajectories from testing refine loops
steps = [1, 2, 3]
easy_scores = [0.42, 0.58, 0.81]     # Multi-step refinement in Task Easy
medium_scores = [0.72, 0.82, 0.85]   # Strategic stability in Task Medium
hard_scores = [0.35, 0.61, 0.94]     # Major improvement in Task Hard

# Styling: High-Fidelity/Professional (Dark Voyager Theme)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(steps, easy_scores, marker='o', markersize=8, linewidth=3.5, label='Easy (Refining Ambiguity)', color='#00E676', alpha=0.9)
ax.plot(steps, medium_scores, marker='s', markersize=8, linewidth=3.5, label='Medium (Gap Detection)', color='#2979FF', alpha=0.9)
ax.plot(steps, hard_scores, marker='D', markersize=8, linewidth=3.5, label='Hard (Policy Evolution)', color='#FFD600', alpha=0.9)

# Enhancements: Title, Labels, Grids
ax.set_title('Strategic Reward Progression: PolicyEvolverEnv', fontsize=20, fontweight='bold', pad=25, color='#FFFFFF')
ax.set_xlabel('Execution Step (Iterative Refinement)', fontsize=14, labelpad=10)
ax.set_ylabel('Strategic Reward (Grader Score)', fontsize=14, labelpad=10)
ax.set_xticks(steps)
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle='--', alpha=0.15)

# Add Legend
legend = ax.legend(fontsize=12, loc='lower right', frameon=True, shadow=True, facecolor='#121212', edgecolor='#333333')
for text in legend.get_texts():
    text.set_color('#FFFFFF')

# Annotations for "RLVR/RLHF" Feedback
ax.annotate('Strategic Convergence (RLVR)', xy=(2.4, 0.88), xytext=(1.2, 0.25),
             arrowprops=dict(facecolor='#FFFFFF', shrink=0.05, alpha=0.4, headwidth=10, width=2),
             fontsize=13, style='italic', color='#B0BEC5')

plt.tight_layout()
plt.savefig('reward_progression.png', dpi=300)
print("Strategic reward progression diagram saved as reward_progression.png! 🚀🏆")
