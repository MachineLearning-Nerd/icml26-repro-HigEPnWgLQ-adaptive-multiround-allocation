# Standard library imports
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

# Local imports
from core.utils import load_pickle

const_palette = plt.cm.YlGn(np.linspace(0.3, 0.9, 4))
greedy_palette = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 4))
greedy_remainder_palette = plt.cm.YlGnBu(np.linspace(0.3, 0.9, 4))
policy_info = {
    ("OurPolicy", None): {"label": r"$\pi^{\mathrm{our}}$", "color": "black", "ls": "-", "marker": "*"},
    ("ConstantPolicy", 2): {"label": "Const(2)", "color": const_palette[0], "ls": ":", "marker": "x"},
    ("ConstantPolicy", 3): {"label": "Const(3)", "color": const_palette[1], "ls": ":", "marker": "x"},
    ("ConstantPolicy", 5): {"label": "Const(5)", "color": const_palette[2], "ls": ":", "marker": "x"},
    ("ConstantPolicy", 10): {"label": "Const(10)", "color": const_palette[3], "ls": ":", "marker": "x"},
    ("GreedyPolicy", 0.1): {"label": "Greedy(0.1)", "color": greedy_palette[0], "ls": "-.", "marker": "^"},
    ("GreedyPolicy", 0.2): {"label": "Greedy(0.2)", "color": greedy_palette[1], "ls": "-.", "marker": "^"},
    ("GreedyPolicy", 0.5): {"label": "Greedy(0.5)", "color": greedy_palette[2], "ls": "-.", "marker": "^"},
    ("GreedyPolicy", 1.0): {"label": "Greedy(1.0)", "color": greedy_palette[3], "ls": "-.", "marker": "^"},
    ("GreedyRemainderPolicy", 0.1): {"label": "GreedyRemainder(0.1)", "color": greedy_remainder_palette[0], "ls": "--", "marker": "o"},
    ("GreedyRemainderPolicy", 0.2): {"label": "GreedyRemainder(0.2)", "color": greedy_remainder_palette[1], "ls": "--", "marker": "o"},
    ("GreedyRemainderPolicy", 0.5): {"label": "GreedyRemainder(0.5)", "color": greedy_remainder_palette[2], "ls": "--", "marker": "o"},
    ("GreedyRemainderPolicy", 1.0): {"label": "GreedyRemainder(1.0)", "color": greedy_remainder_palette[3], "ls": "--", "marker": "o"},
}

legend_order = [
    ("ConstantPolicy", 2),
    ("ConstantPolicy", 3),
    ("ConstantPolicy", 5),
    ("ConstantPolicy", 10),
    ("GreedyPolicy", 0.1),
    ("GreedyPolicy", 0.2),
    ("OurPolicy", None),
    ("GreedyPolicy", 0.3),
    ("GreedyPolicy", 0.5),
    ("GreedyPolicy", 1.0),
    ("GreedyRemainderPolicy", 0.1),
    ("GreedyRemainderPolicy", 0.2),
    ("GreedyRemainderPolicy", 0.3),
    ("GreedyRemainderPolicy", 0.5),
    ("GreedyRemainderPolicy", 1.0)
]

def plot(rng_seed: int = 42) -> None:
    stds = ["Gonorrhea", "Chlamydia", "Syphilis", "HIV", "Hepatitis"]
    num_times = 30
    max_budget = 200
    all_gamma = [0.5, 0.7, 0.9]
    all_n = [5, 10, 15]

    for std_idx in range(5):
        std = stds[std_idx]
        for sim in [True, False]:
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(39, 26))
            plt.subplots_adjust(wspace=0.1, hspace=0.3)
            legend_handles = dict()

            for row_idx in range(len(all_gamma)):
                for col_idx in range(len(all_n)):
                    ax = axes[row_idx, col_idx]
                    gamma = all_gamma[row_idx]
                    n = all_n[col_idx]

                    # Collect results
                    result_fname = f"results/full_nt{num_times}_b{max_budget}_g{gamma}_n{n}_seed{rng_seed}.pkl"
                    all_results = load_pickle(result_fname)
                    ax.set_title(fr"$\gamma = {gamma}$, $n = {n}$", fontsize=35)
                    
                    overall_max_timestep = 0
                    all_trajs = all_results[(std, sim)]
                    for policy_key, trajs in all_trajs.items():
                        if policy_key in policy_info.keys():
                            overall_max_timestep = max(overall_max_timestep, max([len(traj) for traj in trajs]))
                    for policy_key, trajs in all_trajs.items():
                        if policy_key in policy_info.keys():
                            max_timestep = max([len(traj) for traj in trajs])
                            acc_disc_rewards = []
                            for traj in trajs:
                                padded = np.full(max_timestep, traj[-1], dtype=int)
                                padded[:len(traj)] = traj
                                acc_disc_reward = [0]
                                for t in range(1, len(padded)):
                                    val = acc_disc_reward[-1] + pow(gamma, t-1) * (padded[t] - padded[t-1])
                                    acc_disc_reward.append(val)
                                acc_disc_rewards.append(acc_disc_reward)
                            acc_disc_rewards = np.array(acc_disc_rewards)
                            mean_traj = np.mean(acc_disc_rewards, axis=0)
                            std_err_traj = np.std(acc_disc_rewards, axis=0) / np.sqrt(num_times)
                            X = np.arange(overall_max_timestep)
                            line_handle, = ax.plot(
                                X[:len(mean_traj)],
                                mean_traj,
                                color=policy_info[policy_key]['color'],
                                ls=policy_info[policy_key]['ls'],
                                label=policy_info[policy_key]['label'],
                                marker=policy_info[policy_key]['marker'],
                                markevery=[len(mean_traj)-1],
                                lw=5,
                                markerfacecolor="white",
                                markeredgewidth=2,
                                markersize=20 if policy_key != ("OurPolicy", None) else 30,
                                alpha=0.7 if policy_key != ("OurPolicy", None) else 1,
                                zorder=1 if policy_key != ("OurPolicy", None) else 2 # So that OurPolicy appears above others
                            )
                            base_color = to_rgb(line_handle.get_color())
                            fill_color = 0.5 * np.array(base_color) + 0.5 * np.array([1.0, 1.0, 1.0])
                            ax.fill_between(
                                X[:len(mean_traj)],
                                mean_traj - std_err_traj,
                                mean_traj + std_err_traj,
                                color=fill_color,
                                alpha=0.5
                            )
                            ax.plot(
                                X[len(mean_traj)-1:],
                                [mean_traj[-1]] * (overall_max_timestep-len(mean_traj)+1),
                                color=policy_info[policy_key]['color'],
                                ls=policy_info[policy_key]['ls'],
                                lw=5,
                                alpha=0.7 if policy_key != ("OurPolicy", None) else 1,
                                zorder=0 # Put continuation line below actual plot
                            )
                            legend_handles.setdefault(policy_key, line_handle)
                    axes[row_idx, col_idx].tick_params(labelbottom=True, labelleft=True, labelsize=30)

            # Build legend
            handles = [legend_handles[k] for k in legend_order if k in legend_handles]
            labels = [policy_info[k]['label'] for k in legend_order if k in legend_handles]
            for shift_idx in [0, 3, 6, 12, 15]:
                handles.insert(shift_idx, plt.plot([],[],color=(0,0,0,0), label=" ")[0])
                labels.insert(shift_idx, " ")
            fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 0.05), fontsize=30)

            # Annotate overall figure
            fig.supxlabel("Time step", y=0.05, fontsize=40)
            fig.supylabel("Accumulated discounted reward", x=0.07, fontsize=40)
            if sim:
                fig.text(0.5, 0.97, rf"Real-world inspired experiments on simulated distributions from ICPSR {std} disease network", ha="center", va='center', fontsize=40)
            else:
                fig.text(0.5, 0.97, rf"Real-world inspired experiments on actual ICPSR {std} disease network", ha="center", va='center', fontsize=40)
            fig.text(0.5, 0.93, rf"30 runs with maximum budget $b = {max_budget}$", ha="center", va='center', fontsize=40)

            # Save plot
            plot_fname = f"./figures/{std}_{sim}.png"
            os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
            plt.savefig(plot_fname, dpi=150, bbox_inches = 'tight')
            plt.close(fig)

if __name__ == "__main__":
    plot()
    