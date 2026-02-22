import time
import matplotlib.pyplot as plt

import main


def run_visual():
    history_gen = []
    history_best = []
    history_avg = []
    history_params = []
    history_nodes = []
    history_max_nodes = []

    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle("Evolution Progress")

    line_best, = ax1.plot([], [], label="Best Acc")
    line_avg, = ax1.plot([], [], label="Avg Acc")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")

    line_params, = ax2.plot([], [], label="Best Params")
    ax2.set_ylabel("Params")
    ax2.legend(loc="upper left")

    line_nodes, = ax3.plot([], [], label="Best Nodes")
    line_max_nodes, = ax3.plot([], [], label="Max Nodes")
    ax3.set_ylabel("Node Count")
    ax3.set_xlabel("Generation")
    ax3.legend(loc="upper left")

    arch_text = fig.text(0.01, 0.01, "", ha="left", va="bottom")

    def on_generation(stats):
        history_gen.append(stats["generation"])
        history_best.append(stats["best_acc"])
        history_avg.append(stats["avg_acc"])
        history_params.append(stats["best_params"])
        history_nodes.append(stats["best_nodes"])
        history_max_nodes.append(stats["max_nodes"])

        line_best.set_data(history_gen, history_best)
        line_avg.set_data(history_gen, history_avg)
        line_params.set_data(history_gen, history_params)
        line_nodes.set_data(history_gen, history_nodes)
        line_max_nodes.set_data(history_gen, history_max_nodes)

        for ax in (ax1, ax2, ax3):
            ax.relim()
            ax.autoscale_view()

        arch = main.describe_architecture(stats["best_model"])
        arch_text.set_text(f"Best Arch: {arch}")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

    main.run_evolution(on_generation=on_generation)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_visual()
