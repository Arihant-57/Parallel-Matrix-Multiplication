import os
import matplotlib.pyplot as plt

# Output directories
os.makedirs('results/mpi/performance', exist_ok=True)
os.makedirs('results/sequential/performance', exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial']

# Benchmark results
models = [
    'Sequential CPU\n(Single-Threaded)',
    'MPI\n(4 Processes)',
    'OpenMP\n(8 Threads)',
    'CUDA\n(GPU)'
]

times = [
    321.28,
    226.17,
    104.49,
    0.273303
]

# Speedup relative to Sequential CPU
baseline = times[0]
speedups = [baseline / time for time in times]

# Neutral colors for all implementations
colors = ['#94A3B8', '#94A3B8', '#94A3B8', '#94A3B8']


# 1. Combined Performance Chart
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(16, 6.5),
    facecolor='#FFFFFF'
)

bars1 = ax1.bar(
    models,
    times,
    color=colors,
    width=0.55,
    edgecolor='#0F172A',
    linewidth=1
)

ax1.set_title(
    'Benchmark Execution Time',
    fontsize=14,
    fontweight='bold',
    color='#0F172A',
    pad=15
)

ax1.set_ylabel(
    'Execution Time (Seconds)',
    fontsize=11,
    fontweight='bold',
    color='#475569'
)

ax1.set_yscale('log')
ax1.grid(
    True,
    axis='y',
    linestyle='--',
    color='#E2E8F0',
    alpha=0.9
)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis='x', length=0)

for bar, time in zip(bars1, times):
    height = bar.get_height()
    label = f'{time:.3f} s' if time < 1 else f'{time:.2f} s'

    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height * 1.15,
        label,
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='#0F172A'
    )


bars2 = ax2.bar(
    models,
    speedups,
    color=colors,
    width=0.55,
    edgecolor='#0F172A',
    linewidth=1
)

ax2.set_title(
    'Parallel Speedup Factor',
    fontsize=14,
    fontweight='bold',
    color='#0F172A',
    pad=15
)

ax2.set_ylabel(
    'Speedup vs. Sequential',
    fontsize=11,
    fontweight='bold',
    color='#475569'
)

ax2.grid(
    True,
    axis='y',
    linestyle='--',
    color='#E2E8F0',
    alpha=0.9
)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.tick_params(axis='x', length=0)

for bar, speedup in zip(bars2, speedups):
    height = bar.get_height()

    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height * 1.05,
        f'{speedup:.2f}x',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='#0F172A'
    )

plt.suptitle(
    'Performance Analysis: Matrix Multiplication (4000x4000)',
    fontsize=16,
    fontweight='bold',
    color='#0F172A',
    y=1.05
)

plt.tight_layout()

plt.savefig(
    'results/mpi/performance/performance_comparison_charts.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='#FFFFFF'
)

plt.close()


# 2. Standalone Execution Time Chart
fig, ax = plt.subplots(
    figsize=(10, 5),
    facecolor='#FFFFFF'
)

bars = ax.bar(
    models,
    times,
    color=colors,
    width=0.55,
    edgecolor='#0F172A',
    linewidth=1
)

ax.set_title(
    'Matrix Multiplication (4000x4000) Execution Time Comparison',
    fontsize=14,
    fontweight='bold',
    color='#0F172A',
    pad=15
)

ax.set_ylabel(
    'Execution Time (Seconds)',
    fontsize=11,
    fontweight='bold',
    color='#475569'
)

ax.set_yscale('log')

ax.grid(
    True,
    axis='y',
    linestyle='--',
    color='#E2E8F0',
    alpha=0.9
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(axis='x', length=0)

for bar, time in zip(bars, times):
    height = bar.get_height()
    label = f'{time:.3f} s' if time < 1 else f'{time:.2f} s'

    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height * 1.15,
        label,
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='#0F172A'
    )

plt.tight_layout()

plt.savefig(
    'results/sequential/performance/execution_time_chart.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='#FFFFFF'
)

plt.close()


# 3. Standalone Speedup Chart
fig, ax = plt.subplots(
    figsize=(10, 5),
    facecolor='#FFFFFF'
)

bars = ax.bar(
    models,
    speedups,
    color=colors,
    width=0.55,
    edgecolor='#0F172A',
    linewidth=1
)

ax.set_title(
    'Parallel Speedup Factor Relative to Sequential Baseline',
    fontsize=14,
    fontweight='bold',
    color='#0F172A',
    pad=15
)

ax.set_ylabel(
    'Speedup Factor (x)',
    fontsize=11,
    fontweight='bold',
    color='#475569'
)

ax.grid(
    True,
    axis='y',
    linestyle='--',
    color='#E2E8F0',
    alpha=0.9
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(axis='x', length=0)

for bar, speedup in zip(bars, speedups):
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height * 1.05,
        f'{speedup:.2f}x',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='#0F172A'
    )

plt.tight_layout()

plt.savefig(
    'results/mpi/performance/speedup_chart.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='#FFFFFF'
)

plt.close()

print('Performance charts generated successfully.')