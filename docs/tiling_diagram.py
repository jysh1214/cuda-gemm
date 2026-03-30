import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# === Diagram 1: The Problem (Naive) ===
ax = axes[0]
ax.set_title("Naive: Redundant Global Memory Loads", fontsize=13, fontweight='bold')
ax.set_xlim(-1, 14)
ax.set_ylim(-1, 10)
ax.set_aspect('equal')
ax.axis('off')

# Matrix A
ax.add_patch(patches.Rectangle((0, 4), 4, 4, fill=True, facecolor='#ddeeff', edgecolor='black', lw=2))
ax.text(2, 8.5, 'A (M x K)', ha='center', fontsize=11, fontweight='bold')
# Highlight row 0
ax.add_patch(patches.Rectangle((0, 7), 4, 1, fill=True, facecolor='#ff9999', edgecolor='black', lw=2))
ax.text(-0.5, 7.5, 'row i', ha='right', fontsize=9, color='red')
# Highlight same row again (thread 1)
ax.add_patch(patches.Rectangle((0, 6), 4, 1, fill=True, facecolor='#ffcc99', edgecolor='black', lw=2))
ax.text(-0.5, 6.5, 'row i', ha='right', fontsize=9, color='orange')

# Matrix B
ax.add_patch(patches.Rectangle((6, 4), 4, 4, fill=True, facecolor='#ddeeff', edgecolor='black', lw=2))
ax.text(8, 8.5, 'B (K x N)', ha='center', fontsize=11, fontweight='bold')
# Highlight col 0
ax.add_patch(patches.Rectangle((6, 4), 1, 4, fill=True, facecolor='#ff9999', edgecolor='black', lw=2))
ax.text(6.5, 3.3, 'col j', ha='center', fontsize=9, color='red')
# Highlight col 1
ax.add_patch(patches.Rectangle((7, 4), 1, 4, fill=True, facecolor='#ffcc99', edgecolor='black', lw=2))
ax.text(7.5, 3.3, 'col j+1', ha='center', fontsize=9, color='orange')

# Labels
ax.text(2, 2.5, 'Thread (i,j) loads\nentire row i of A', ha='center', fontsize=9, color='red')
ax.text(8, 2.5, 'Thread (i,j+1) loads\nSAME row i of A again!', ha='center', fontsize=9, color='orange')
ax.text(5, 0.5, 'Same data loaded from global memory\nby EVERY thread in the same row = wasted bandwidth',
        ha='center', fontsize=10, style='italic', color='#cc0000')

# === Diagram 2: Tiling Concept ===
ax = axes[1]
ax.set_title("Tiling: Load Tile into Shared Memory", fontsize=13, fontweight='bold')
ax.set_xlim(-1, 16)
ax.set_ylim(-2, 11)
ax.set_aspect('equal')
ax.axis('off')

# Matrix A with tile highlighted
ax.add_patch(patches.Rectangle((0, 4), 6, 6, fill=True, facecolor='#ddeeff', edgecolor='black', lw=2))
ax.text(3, 10.5, 'A', ha='center', fontsize=12, fontweight='bold')
# Tile of A (TILE_SIZE rows x TILE_K cols)
ax.add_patch(patches.Rectangle((0, 8), 2, 2, fill=True, facecolor='#ff9999', edgecolor='black', lw=2, alpha=0.8))
ax.text(1, 9, 'Tile\nAs', ha='center', fontsize=8, fontweight='bold')
ax.annotate('', xy=(-0.3, 8), xytext=(-0.3, 10), arrowprops=dict(arrowstyle='<->', color='red'))
ax.text(-0.7, 9, 'TILE', ha='right', fontsize=8, color='red', rotation=90)

# Matrix B with tile highlighted
ax.add_patch(patches.Rectangle((8, 4), 6, 6, fill=True, facecolor='#ddeeff', edgecolor='black', lw=2))
ax.text(11, 10.5, 'B', ha='center', fontsize=12, fontweight='bold')
# Tile of B (TILE_K rows x TILE_SIZE cols)
ax.add_patch(patches.Rectangle((8, 8), 2, 2, fill=True, facecolor='#99ccff', edgecolor='black', lw=2, alpha=0.8))
ax.text(9, 9, 'Tile\nBs', ha='center', fontsize=8, fontweight='bold')

# Shared memory box
ax.add_patch(patches.FancyBboxPatch((3, -1), 8, 3, boxstyle="round,pad=0.3",
             facecolor='#ffffcc', edgecolor='#cc9900', lw=2))
ax.text(7, 0.5, 'Shared Memory\n(fast, on-chip)', ha='center', fontsize=10, fontweight='bold', color='#996600')

# Arrows: global -> shared
ax.annotate('', xy=(5, 2), xytext=(1, 7.5), arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.annotate('', xy=(9, 2), xytext=(9, 7.5), arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(2, 5, '1x load\nper tile', fontsize=8, color='red', fontweight='bold')

# === Diagram 3: Sliding Window ===
ax = axes[2]
ax.set_title("Slide Tiles Across K Dimension", fontsize=13, fontweight='bold')
ax.set_xlim(-1, 16)
ax.set_ylim(-2, 12)
ax.set_aspect('equal')
ax.axis('off')

# Matrix A
M_h, K_w = 4, 8
ax.add_patch(patches.Rectangle((0, 5), K_w, M_h, fill=True, facecolor='#eee', edgecolor='black', lw=2))
ax.text(K_w/2, 9.5, 'A (M x K)', ha='center', fontsize=11, fontweight='bold')
# Sliding tiles on A (along columns)
colors_a = ['#ff9999', '#ffcc66', '#99ff99']
for i, c in enumerate(colors_a):
    ax.add_patch(patches.Rectangle((i*2, 7), 2, 2, fill=True, facecolor=c, edgecolor='black', lw=2, alpha=0.7))
    ax.text(i*2+1, 8, f't={i}', ha='center', fontsize=9, fontweight='bold')
ax.annotate('', xy=(6.5, 8), xytext=(1, 8), arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(4, 6.5, 'slide along K  →', ha='center', fontsize=9)

# Matrix C = result
cx, cy = 10, 5
ax.add_patch(patches.Rectangle((cx, cy+2), 4, 2, fill=True, facecolor='#d4edda', edgecolor='black', lw=3))
ax.text(cx+2, cy+4.5, 'C tile', ha='center', fontsize=11, fontweight='bold')
ax.text(cx+2, cy+3, 'C += As * Bs\n(accumulate)', ha='center', fontsize=9)

# Matrix B
bx, by = 10, 0
ax.add_patch(patches.Rectangle((bx, by), 4, 6, fill=True, facecolor='#eee', edgecolor='black', lw=2))
ax.text(bx+2, -0.7, 'B (K x N)', ha='center', fontsize=11, fontweight='bold')
# Sliding tiles on B (along rows)
for i, c in enumerate(colors_a):
    ax.add_patch(patches.Rectangle((bx, by+4-i*2), 2, 2, fill=True, facecolor=c, edgecolor='black', lw=2, alpha=0.7))

# Summary text
ax.text(8, -1.5, 'Each step: load one tile of A & B into shared mem,\n'
        'all threads in block reuse them → fewer global loads',
        ha='center', fontsize=10, style='italic', color='#006600')

plt.tight_layout()
plt.savefig('/home/alex/cuda_gemm/docs/tiling_explained.png', dpi=150, bbox_inches='tight')
print("Saved docs/tiling_explained.png")
