import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke

# real Baseline Metrics from latest inference.py run (Llama-3.3-70B)
# These represent the 'Zero-Shot' or 'Basic COT' performance gap we aim to solve.
steps = [1, 2, 3]
base_easy = [0.67, 0.61, 0.32]   # Drop at Step 3: Anti-Repetition / Vagueness penalty triggered.
base_med = [0.85, 0.80, 0.70]    # Slight decline: Complexity-to-Stability gap.
base_hard = [0.50, 0.45, 0.45]   # Stagnation: Hallucination/Realism constraint prevents easy scoring.

# Theoretical RL-Optimized Convergent Trajectory (The Goal of PolicyEvolver training)
rl_target = [0.45, 0.78, 0.98]

colors = {
    'easy': '#00F5FF',    # Cyan
    'medium': '#FF00E5',  # Magenta
    'hard': '#FFD700',    # Gold
    'target': '#FFFFFF',  # White (Dashed)
    'bg': '#0D0F14',      # Obsidian
    'grid': '#1A1D23',
    'text': '#E0E0E0'
}

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
fig.patch.set_facecolor(colors['bg'])
ax.set_facecolor(colors['bg'])

def plot_line(x, y, label, color, marker, linestyle='-', alpha_base=1.0, is_target=False):
    if is_target:
        ax.plot(x, y, linestyle='--', linewidth=2, color=color, alpha=0.4, label=label, zorder=3)
        return
    
    # Glow layers
    for w in range(1, 12, 2):
        ax.plot(x, y, color=color, linewidth=w, alpha=0.03, zorder=2)
    # Main line
    ax.plot(x, y, marker=marker, markersize=8, linewidth=3, 
            label=label, color=color, zorder=5, alpha=alpha_base,
            path_effects=[withStroke(linewidth=3, foreground='black')])

# Plotting
plot_line(steps, base_easy, 'Agent: Easy (Clarification)', colors['easy'], 'o')
plot_line(steps, base_med, 'Agent: Medium (Gap Detection)', colors['medium'], 's')
plot_line(steps, base_hard, 'Agent: Hard (Evolution)', colors['hard'], 'D')
plot_line(steps, rl_target, 'RLVR Fine-Tuning Target', colors['target'], None, is_target=True)

# Strategic Annotations for the judges
ax.annotate(' Penalty: Repetition / Vagueness Hit', xy=(3, 0.32), xytext=(2.2, 0.15),
             arrowprops=dict(facecolor='#FF5252', shrink=0.05, width=1, headwidth=6),
             fontsize=10, fontweight='bold', color='#FF5252', bbox=dict(facecolor='#000', alpha=0.5))

ax.annotate(' Stagnation: Realism Constraint', xy=(3, 0.45), xytext=(2.0, 0.28),
             arrowprops=dict(facecolor='#FFAB00', shrink=0.05, width=1, headwidth=6),
             fontsize=10, fontweight='bold', color='#FFAB00', bbox=dict(facecolor='#000', alpha=0.5))

ax.set_title('Strategic Performance Gap: Baseline vs. Optimized', fontsize=20, fontweight='black', pad=30)
ax.set_xlabel('Iterative Strategy Step', fontsize=12, labelpad=10)
ax.set_ylabel('Grader Reward (Strategy Convergence)', fontsize=12, labelpad=10)

ax.set_xticks(steps)
ax.set_ylim(0, 1.1)
ax.grid(True, linestyle='-', color=colors['grid'], alpha=0.4, zorder=1)

# Style Overrides
for spine in ax.spines.values(): spine.set_visible(False)
legend = ax.legend(fontsize=10, loc='upper left', frameon=True, facecolor='#15181E', edgecolor='#2A2D35')
for text in legend.get_texts(): text.set_color(colors['text'])

# Branding
ax.text(0.98, 0.02, 'Meta x PyTorch x Scaler Hackathon | PolicyEvolver v2.0', 
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9, alpha=0.4, color=colors['text'])

plt.tight_layout()
plt.savefig('reward_progression.png', dpi=300, facecolor=colors['bg'], bbox_inches='tight')
print("High-Fidelity Strategic Gap Chart Generated! 🚀📊")
