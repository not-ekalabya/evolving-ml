import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import main


def _node_style(node):
    if node is None:
        return {"face": "#1d4ed8", "edge": "#0f172a", "text": "white"}
    if isinstance(node, main.MatMulNode):
        return {"face": "#10b981", "edge": "#064e3b", "text": "black"}
    if isinstance(node, main.ActivationNode):
        return {"face": "#f59e0b", "edge": "#92400e", "text": "black"}
    if isinstance(node, main.ReshapeNode):
        return {"face": "#60a5fa", "edge": "#1e3a8a", "text": "black"}
    if isinstance(node, main.AddNode):
        return {"face": "#f97316", "edge": "#9a3412", "text": "black"}
    if isinstance(node, main.ConcatNode):
        return {"face": "#a78bfa", "edge": "#4c1d95", "text": "black"}
    if isinstance(node, main.ReduceNode):
        return {"face": "#f43f5e", "edge": "#881337", "text": "white"}
    if isinstance(node, main.IndexNode):
        return {"face": "#22c55e", "edge": "#14532d", "text": "black"}
    return {"face": "#e2e8f0", "edge": "#334155", "text": "black"}


def _node_label(name, node):
    if node is None:
        return "input"
    if isinstance(node, main.MatMulNode):
        return f"{name}\nMatMul {node.W.shape[0]}→{node.W.shape[1]}"
    if isinstance(node, main.ActivationNode):
        return f"{name}\n{node.activation_type.upper()}"
    if isinstance(node, main.ReshapeNode):
        return f"{name}\nReshape {node.operation}"
    if isinstance(node, main.AddNode):
        return f"{name}\nAdd"
    if isinstance(node, main.ConcatNode):
        return f"{name}\nConcat"
    if isinstance(node, main.ReduceNode):
        return f"{name}\nReduce {node.operation}"
    if isinstance(node, main.IndexNode):
        return f"{name}\nIndex {node.mode}"
    return f"{name}\n{type(node).__name__}"


def _layout_positions(model):
    positions = {"input": (0.0, 0.0)}
    for idx, name in enumerate(model.execution_order, start=1):
        node = model.nodes[name]
        if node.inputs:
            xs = [positions.get(src, (0.0, 0.0))[0] for src in node.inputs]
            x = sum(xs) / len(xs)
        else:
            x = 0.0
        y = -float(idx)
        positions[name] = (x, y)
    return positions


def _draw_graph(ax, model):
    ax.clear()
    ax.set_axis_off()
    ax.set_facecolor("#0b1220")

    positions = _layout_positions(model)
    min_x = min(x for x, _ in positions.values())
    max_x = max(x for x, _ in positions.values())
    min_y = min(y for _, y in positions.values())
    padding_x = 1.6
    padding_y = 1.2

    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(min_y - padding_y, 1.5)

    # Draw edges
    for name in model.execution_order:
        node = model.nodes[name]
        for src in node.inputs:
            x1, y1 = positions.get(src, (0.0, 0.0))
            x2, y2 = positions[name]
            ax.annotate(
                "",
                xy=(x2, y2 + 0.1),
                xytext=(x1, y1 - 0.1),
                arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.3, shrinkA=6, shrinkB=6),
                zorder=1,
            )

    # Draw nodes
    for name in ["input"] + model.execution_order:
        node = None if name == "input" else model.nodes[name]
        x, y = positions[name]
        style = _node_style(node)
        width = 1.8 if name != "input" else 1.6
        height = 0.6 if name != "input" else 0.5
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=style["edge"],
            facecolor=style["face"],
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            _node_label(name, node),
            ha="center",
            va="center",
            fontsize=9,
            color=style["text"],
            zorder=3,
        )


def _format_stats(stats):
    arch = main.describe_architecture(stats["best_model"])
    wrapped_arch = "\n".join(textwrap.wrap(arch, width=38))
    mutation = getattr(stats["best_model"], "_last_mutation", "n/a")
    lines = [
        f"Generation: {stats['generation']}",
        f"Best Acc:   {stats['best_acc']:.4f}",
        f"Avg Acc:    {stats['avg_acc']:.4f}",
        f"Best Params:{stats['best_params']:,}",
        f"Best Nodes: {stats['best_nodes']}",
        f"Max Nodes:  {stats['max_nodes']}",
        f"Mutation:   {mutation}",
        "",
        "Best Architecture:",
        wrapped_arch,
    ]
    return "\n".join(lines)


def run_visual():
    plt.ion()
    fig = plt.figure(figsize=(13, 7), facecolor="#0b1220")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.3], wspace=0.05)
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.set_axis_off()
    ax_stats.set_facecolor("#0f172a")

    title = fig.suptitle("Model Evolution — Architecture View", color="white", fontsize=14, y=0.98)

    stat_text = ax_stats.text(
        0.04,
        0.97,
        "",
        ha="left",
        va="top",
        fontsize=10,
        color="#e2e8f0",
        family="monospace",
        transform=ax_stats.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#111827", edgecolor="#334155", linewidth=1.2),
    )

    def on_generation(stats):
        _draw_graph(ax_graph, stats["best_model"])
        stat_text.set_text(_format_stats(stats))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

    main.run_evolution(on_generation=on_generation)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_visual()
