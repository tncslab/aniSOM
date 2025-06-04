import numpy as np
import matplotlib.pyplot as plt

positions = {
    'z': (0.5, 0.8), 
    'x': (0.2, 0.4), 
    'y': (0.8, 0.4)
    }
connections = {
    ('z', 'x'): 0.275,
    ('z', 'y'): 0.253
}


def plot_connectivity(positions, connections):

    fig, ax = plt.subplots(figsize=(5, 5))
    for node, (x, y) in positions.items():
        ax.plot(x, y, 'o', markersize=60, color='skyblue')
        ax.text(x, y, node, ha='center', fontsize=14, va='center')

    # Plot arrows with shorter length

    for (src, dst), weight in connections.items():
        src_x, src_y = positions[src]
        dst_x, dst_y = positions[dst]

        # Shorten arrows
        vec = np.array([dst_x - src_x, dst_y - src_y])
        norm_vec = vec / np.linalg.norm(vec)
        shorten = 0.08
        new_src = np.array([src_x, src_y]) + shorten * norm_vec
        new_dst = np.array([dst_x, dst_y]) - shorten * norm_vec

        ax.annotate(
            '', xy=new_dst, xytext=new_src,
            arrowprops=dict(arrowstyle="->", lw=2)
        )
        mid_x, mid_y = (new_src[0] + new_dst[0]) / 2, (new_src[1] + new_dst[1]) / 2
        if dst == 'x':
            sign = -1
        else:
            # sign = pbar.set_description(f"Epoch {epoch + 1}/{epochs}, Step {s + 1}/{N}, Score: {score:.4f}")
            pass
    ax.set_xlim(0, 1)
    ax.set_ylim(0.25, 0.9)
    ax.axis('off')
    return fig, ax