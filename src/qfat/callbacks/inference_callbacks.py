import logging
from collections import defaultdict
from typing import DefaultDict, List, Literal, Optional

import hydra
import imageio
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from scipy.stats import sem  # Standard error of the mean

import wandb
import wandb.plot
from qfat.callbacks.callbacks import Callback
from qfat.conf.configs import MultiPathTrajectoryDatasetCfg
from qfat.datasets.dataset import TrajectoryDataset
from qfat.entrypoints.inference import InferenceEntrypoint
from qfat.environments.pusht.env import PushTKeypointsEnv

logger = logging.getLogger(__name__)


class RewardLogger(Callback):
    """Logs rewards for each episode, with customizable reward reduction methods."""

    def __init__(
        self,
        reward_reduction: Literal["mean", "max", "last", "sum"] = "sum",
        **kwargs,
    ):
        """
        Args:
            reward_reduction (Literal["mean", "max", "last", "sum"]):
                How to aggregate the episode rewards.
        """
        self.cumulative_reward = 0
        self.counter = 0
        self.reward_reduction = reward_reduction

    def reduce_episode_rewards(self, episode_rewards: List[float]) -> float:
        """Applies the specified reward reduction method to episode rewards."""
        if self.reward_reduction == "mean":
            return sum(episode_rewards) / len(episode_rewards)
        elif self.reward_reduction == "max":
            return max(episode_rewards)
        elif self.reward_reduction == "last":
            return episode_rewards[-1]
        elif self.reward_reduction == "sum":
            return sum(episode_rewards)
        else:
            raise ValueError(
                f"Unknown reward reduction method: {self.reward_reduction}"
            )

    def __call__(self, inference_ep):
        reduced_reward = self.reduce_episode_rewards(inference_ep.episode_rewards)
        wandb.log(
            {
                "reward": reduced_reward,
                "episode": inference_ep.episode_counter,
            },
        )
        self.cumulative_reward += reduced_reward
        self.counter += 1

    def finalize(self) -> None:
        mean_reward = self.cumulative_reward / self.counter if self.counter > 0 else 0
        wandb.log({"mean_reward": mean_reward})


class EnvBehaviourLogger(Callback):
    """Logs the completed tasks sequences at the end of each episode"""

    def __init__(
        self,
        min_sequence_length: int,
        max_sequence_length: int,
        **kwargs,
    ):
        super().__init__()
        self.completed_tasks = {
            i: [] for i in range(min_sequence_length, max_sequence_length + 1)
        }
        self.completed_tasks_lengths = []
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length

    def __call__(self, inference_ep: InferenceEntrypoint):
        task = inference_ep.env.completed_tasks
        self.completed_tasks_lengths.append(len(task))
        if len(task) in range(self.min_sequence_length, self.max_sequence_length + 1):
            for i in range(self.min_sequence_length, len(task) + 1):
                task_i = task[:i]
                task_str = "->".join(task_i)
                self.completed_tasks[i].append(task_str)

    def finalize(self) -> None:
        for i in range(self.min_sequence_length, self.max_sequence_length + 1):
            df = pd.DataFrame({f"completed_task@{i}": self.completed_tasks[i]})
            df = df.value_counts().reset_index()
            df.columns = [f"completed_task@{i}", "count"]
            df["probability"] = df["count"] / df["count"].sum()
            entropy = (
                -(df["probability"].values * np.log2(df["probability"].values))
                .sum()
                .item()
            )
            wandb.log(
                {
                    f"task_statistics@{i}": wandb.Table(dataframe=df),
                    f"task_entropy@{i}": entropy,
                }
            )
        df = pd.DataFrame({"completed_task_lengths": self.completed_tasks_lengths})
        df = df.value_counts().reset_index()
        n_total_tasks = len(self.completed_tasks_lengths)
        df["probability"] = df["count"] / n_total_tasks
        df = df.sort_values(by="completed_task_lengths", ascending=False)
        df["cumulative_probability"] = df["probability"].cumsum()
        df = df[["cumulative_probability", "completed_task_lengths"]]
        wandb.log(
            {
                f"p{int(row['completed_task_lengths'])}": row["cumulative_probability"]
                for _, row in df.iterrows()
            }
        )


class UR3ActionProjectionVideo(Callback):
    """Visualize whether actions are closer to goal 1 or goal 2 based on y-axis distance."""

    def __init__(self, skip_n: int = 1, **kwargs):
        super().__init__()
        self.goal_1_y = -0.25  # y-coordinate of goal 1
        self.goal_2_y = -0.40  # y-coordinate of goal 2
        self.skip_n = skip_n
        self.steps = []
        self.closer_to_goal_1 = []
        self.closer_to_goal_2 = []

    def __call__(self, inference_ep: InferenceEntrypoint):
        meta = inference_ep.current_sample_metadata
        if inference_ep.step % self.skip_n == 0:
            dist = meta["distribution"]
            sampled_actions = dist.sample((1000,))
            actions_np = sampled_actions.cpu().numpy().squeeze(1)

            distances_to_goal_1_y = np.abs(actions_np[:, 1] - self.goal_1_y)
            distances_to_goal_2_y = np.abs(actions_np[:, 1] - self.goal_2_y)

            closer_to_goal_1_count = np.sum(
                distances_to_goal_1_y < distances_to_goal_2_y
            )
            closer_to_goal_2_count = np.sum(
                distances_to_goal_2_y < distances_to_goal_1_y
            )

            self.steps.append(inference_ep.step)
            self.closer_to_goal_1.append(closer_to_goal_1_count)
            self.closer_to_goal_2.append(closer_to_goal_2_count)

    def finalize(self):
        if not self.closer_to_goal_1 or not self.closer_to_goal_2 or not self.steps:
            print("No data to visualize. Ensure steps were skipped appropriately.")
            return

        assert (
            len(self.closer_to_goal_1) == len(self.closer_to_goal_2) == len(self.steps)
        ), "Inconsistent data lengths. Check projection and step collection logic."

        fig, ax = plt.subplots(figsize=(10, 6))

        def update(frame):
            ax.clear()
            step = self.steps[frame]

            ax.bar(
                ["Goal 1", "Goal 2"],
                [self.closer_to_goal_1[frame], self.closer_to_goal_2[frame]],
                color=["red", "blue"],
                alpha=0.7,
            )

            ax.set_ylim(0, 1000)
            ax.set_ylabel("Number of Actions")
            ax.set_title(f"Proximity to Goals Based on Y-Axis (Step {step})")
            ax.grid(axis="y")

        ani = FuncAnimation(fig, update, frames=len(self.steps), interval=300)

        wandb_dir = wandb.run.dir
        video_path = f"{wandb_dir}/goal_proximity_y.mp4"
        ani.save(video_path, writer="ffmpeg")

        wandb.log(
            {"Goal Proximity Video (Y-Axis)": wandb.Video(video_path, format="mp4")}
        )


class ModesCounter(Callback):
    def __init__(self):
        super().__init__()
        self.episode_data = defaultdict(list)

    def __call__(self, inference_ep):
        """Tracks the number of active modes per timestep in each episode."""
        modes = inference_ep.current_sample_metadata["valid_modes"]
        active_components = len(modes)
        self.episode_data[inference_ep.episode_counter].append(active_components)

    def finalize(self):
        """Compiles per-step data for each episode, logs to W&B, and saves to CSV."""
        max_timesteps = max(len(v) for v in self.episode_data.values())
        episode_list = list(self.episode_data.values())

        # Pad shorter trajectories with NaNs for alignment
        for ep in episode_list:
            ep.extend([np.nan] * (max_timesteps - len(ep)))

        # Create DataFrame with each column as an episode
        df = pd.DataFrame(episode_list).T
        df.columns = [f"episode_{i}" for i in range(len(episode_list))]
        df.to_csv("active_modes_trajectory.csv", index=False)
        print("Active modes per time step saved to 'active_modes_trajectory.csv'.")

        # ---- Multi-line plot: one line per episode ----
        melted_df = df.reset_index().melt(
            id_vars="index", var_name="episode", value_name="active_modes"
        )
        melted_df = melted_df.rename(columns={"index": "timestep"})

        table_all_episodes = wandb.Table(dataframe=melted_df)

        wandb.log(
            {
                "active_modes_trend_per_episode": wandb.plot.line(
                    table_all_episodes,
                    x="timestep",
                    y="active_modes",
                    stroke="episode",
                    title="Active Modes per Timestep (Each Episode)",
                )
            }
        )

        # ---- Mean & Confidence Interval Plot ----
        mean_active_modes = df.mean(axis=1)
        ci = sem(df, axis=1, nan_policy="omit")

        data_table = [
            {"step": step, "mean_active_modes": mean, "ci": error}
            for step, (mean, error) in enumerate(zip(mean_active_modes, ci))
        ]

        table_mean = wandb.Table(
            data=data_table, columns=["step", "mean_active_modes", "ci"]
        )

        wandb.log(
            {
                "mean_active_modes_trend": wandb.plot.line(
                    table_mean,
                    x="step",
                    y="mean_active_modes",
                    title="Mean Active Modes per Timestep",
                ),
                "active_modes_table_mean": table_mean,
            }
        )


# class ModesCounter(Callback):
#     def __init__(self):
#         super().__init__()
#         self.episode_data = defaultdict(lambda: [])

#     def __call__(self, inference_ep: InferenceEntrypoint):
#         modes = inference_ep.current_sample_metadata["valid_modes"]
#         active_components = len(modes)
#         self.episode_data[inference_ep.episode_counter].append(active_components)

#     def finalize(self):
#         all_active_counts = [
#             count
#             for episode_counts in self.episode_data.values()
#             for count in episode_counts
#         ]
#         mean_active_components = np.mean(all_active_counts)
#         wandb.log({"mean_active_modes": mean_active_components})
#         data = []
#         for episode_id, counts in self.episode_data.items():
#             for step, count in enumerate(counts):
#                 data.append(
#                     {"episode_id": episode_id, "step": step, "active_modes": count}
#                 )

#         df = pd.DataFrame(data)
#         df.to_csv("active_modes_distribution.csv", index=False)

#         print(
#             f"Mean active modes across all steps and episodes: {mean_active_components}"
#         )
#         print(
#             "Active components distribution saved to 'active_modes_distribution.csv'."
#         )


# class MixtureComponentsCounter(Callback):
#     def __init__(self):
#         super().__init__()
#         self.episode_data = defaultdict(list)

#     def __call__(self, inference_ep):
#         components = inference_ep.current_sample_metadata["mixture_probabilities"]
#         active_components = components / components.max() > 1e-2
#         active_count = active_components.sum()
#         self.episode_data[inference_ep.episode_counter].append(active_count)

#     def finalize(self):
#         """At the end of training or evaluation, compile the per-step data for each episode
#         into a DataFrame and log it to Weights & Biases."""
#         max_timesteps = max(len(v) for v in self.episode_data.values())
#         episode_list = list(self.episode_data.values())

#         # Pad shorter trajectories with NaNs so each episode has equal length
#         for ep in episode_list:
#             ep.extend([np.nan] * (max_timesteps - len(ep)))

#         # Each column in df = one episode; each row = one timestep
#         df = pd.DataFrame(episode_list).T
#         df.columns = [f"episode_{i}" for i in range(len(episode_list))]
#         df.to_csv("active_components_trajectory.csv", index=False)
#         print(
#             "Active components per time step saved to 'active_components_trajectory.csv'."
#         )

#         # ---- Option 1: Multi-line plot: one line per episode ----
#         # Reshape (melt) to long form so that each row = (timestep, episode, active_components)
#         melted_df = df.reset_index().melt(
#             id_vars="index", var_name="episode", value_name="active_components"
#         )
#         melted_df = melted_df.rename(columns={"index": "timestep"})

#         # Create a wandb.Table from the melted DataFrame
#         # wandb.plot.line allows specifying 'stroke' to group lines by an attribute
#         table_all_episodes = wandb.Table(dataframe=melted_df)

#         wandb.log(
#             {
#                 "active_components_trend_per_episode": wandb.plot.line(
#                     table_all_episodes,
#                     x="timestep",
#                     y="active_components",
#                     stroke="episode",  # each episode gets its own line
#                     title="Active Components per Timestep (Each Episode)",
#                 )
#             }
#         )

#         # ---- Option 2: (If you *also* want mean & standard error) ----
#         mean_active_components = df.mean(axis=1)
#         ci = sem(df, axis=1, nan_policy="omit")  # standard error

#         data_table = []
#         for step, (mean_val, std_err) in enumerate(zip(mean_active_components, ci)):
#             data_table.append(
#                 {"step": step, "mean_active_components": mean_val, "ci": std_err}
#             )
#         table_mean = wandb.Table(
#             data=data_table, columns=["step", "mean_active_components", "ci"]
#         )

#         # This logs the *average* number of active components as a single line
#         wandb.log(
#             {
#                 "mean_active_components_trend": wandb.plot.line(
#                     table_mean,
#                     x="step",
#                     y="mean_active_components",
#                     title="Mean Active Components per Timestep",
#                 ),
#                 "active_components_table_mean": table_mean,
#             }
#         )


class MixtureComponentsCounter(Callback):
    def __init__(self):
        super().__init__()
        self.episode_data = defaultdict(lambda: [])

    def __call__(self, inference_ep: InferenceEntrypoint):
        components = inference_ep.current_sample_metadata["mixture_probabilities"]
        active_components = components / components.max() > 1e-2
        active_count = active_components.sum()
        self.episode_data[inference_ep.episode_counter].append(active_count)

    def finalize(self):
        all_active_counts = [
            count
            for episode_counts in self.episode_data.values()
            for count in episode_counts
        ]
        mean_active_components = np.mean(all_active_counts)
        wandb.log({"mean_active_components": mean_active_components})
        data = []
        for episode_id, counts in self.episode_data.items():
            for step, count in enumerate(counts):
                data.append(
                    {"episode_id": episode_id, "step": step, "active_components": count}
                )

        df = pd.DataFrame(data)
        df.to_csv("active_components_distribution.csv", index=False)

        print(
            f"Mean active components across all steps and episodes: {mean_active_components}"
        )
        print(
            "Active components distribution saved to 'active_components_distribution.csv'."
        )


class PushTOverlayImage(Callback):
    """Overlays the trajectories of the PushT agent onto an image and logs it to wandb."""

    def __init__(self):
        super().__init__()
        self.trajectories = defaultdict(lambda: [])

    def __call__(self, inference_ep: InferenceEntrypoint):
        self.trajectories[inference_ep.episode_counter].append(
            np.array(inference_ep.env.env.agent.position) / 512
        )

    def finalize(self):
        env = PushTKeypointsEnv(
            reset_to_state=[
                298.42640687119285,
                213.57359312880715,
                213.57359312880715,
                298.42640687119285,
                0.7853981633974483,
            ],
            render_size=512,
        )
        env.reset()
        bg_img = env.render("rgb_array")  # shape ~ (render_size, render_size, 3)
        H, W = bg_img.shape[:2]  # height, width
        _, ax = plt.subplots()
        ax.imshow(bg_img)
        for traj in self.trajectories.values():
            agent_poses = np.array(traj)
            x_vals = agent_poses[:, 0] * H
            y_vals = agent_poses[:, 1] * W

            points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments, cmap="plasma", norm=plt.Normalize(0, len(segments))
            )
            lc.set_array(np.linspace(0, len(segments), len(segments)))
            lc.set_linewidth(2)
            ax.add_collection(lc)
            ax.set_xlim([0, W])
            ax.set_ylim([H, 0])
        plt.axis("off")
        wandb_dir = wandb.run.dir
        img_path = f"{wandb_dir}/pusht_trajectories.png"
        plt.savefig(img_path, dpi=1000, bbox_inches="tight", transparent=True)
        plt.close()
        wandb.log({"trajectories": wandb.Image(img_path)})


class QFATOutputStatsLogger(Callback):
    """Logs the sampled mixture component, corresponding mixture probabilities,
    and visualizes the time evolution of the PDF with marked means, mixture probabilities,
    and all valid modes discovered by the `sample_modes` function."""

    def __init__(
        self,
        skip_n: int = 1,
        lim: float = 0.5,
        dataset: Optional[TrajectoryDataset] = None,
        **kwargs,
    ):
        """
        Args:
            skip_n: Log/plot the PDF every N steps in the episode.
            lim: Range for plotting the PDF in each dimension ([-lim, lim]).
            dataset: (Optional) a dataset for comparing the predicted PDF
                against actual ground-truth action data.
        """
        super().__init__()
        # We'll keep track of time-evolution data per dimension index
        # Each dimension index maps to a dict containing lists of per-step info.
        self.time_evolution_data: DefaultDict[int, dict] = defaultdict(
            lambda: {
                "pdfs": [],
                "means": [],
                "mixture_probs": [],
                "modes": [],
            }
        )
        self.lim = lim
        self.skip_n = skip_n
        self.dataset = dataset

    def __call__(self, inference_ep: InferenceEntrypoint):
        """
        Invoked at each inference step/episode. Gathers metrics for logging and
        optionally stores the per-dimension PDF, mean, mixture probabilities,
        and all valid modes (if available).
        """
        meta = inference_ep.current_sample_metadata
        episode_name = f"episode_{inference_ep.episode_counter}"

        component_key = f"sampled_component@{episode_name}"
        proba_key = "prob_component_{}@{}"
        metrics = {
            component_key: meta["sampled_component"],
            "episode_step": inference_ep.step,
        }
        for i, prob in enumerate(meta["mixture_probabilities"]):
            metrics[proba_key.format(i, episode_name)] = prob.item()
        wandb.log(metrics)

        if inference_ep.step % self.skip_n == 0:
            dist = meta["distribution"]
            component_distribution = dist.component_distribution
            mixture_distribution = dist.mixture_distribution

            mixture_probs = mixture_distribution.probs.detach().cpu().numpy().squeeze()
            locs = component_distribution.loc.detach()
            scales = component_distribution.scale_tril.detach()
            valid_modes = meta.get("valid_modes", None)

            for dimension_index in range(locs.shape[-1]):
                selected_dimension_loc = locs[..., dimension_index]
                selected_dimension_scale = scales[..., dimension_index, dimension_index]
                selected_dimension_distribution = torch.distributions.MixtureSameFamily(
                    mixture_distribution,
                    torch.distributions.Normal(
                        selected_dimension_loc, selected_dimension_scale
                    ),
                )
                x_values = (
                    torch.linspace(-self.lim, self.lim, 10000)
                    .unsqueeze(-1)
                    .to(locs.device)
                )
                pdf_values = torch.exp(
                    selected_dimension_distribution.log_prob(x_values)
                )
                pdf_values_np = pdf_values.squeeze().detach().cpu().numpy()
                self.time_evolution_data[dimension_index]["pdfs"].append(pdf_values_np)
                self.time_evolution_data[dimension_index]["means"].append(
                    selected_dimension_loc.squeeze().cpu().numpy()
                )
                self.time_evolution_data[dimension_index]["mixture_probs"].append(
                    mixture_probs
                )

                if valid_modes is not None:
                    modes_1d = valid_modes[:, dimension_index].cpu().numpy()
                    self.time_evolution_data[dimension_index]["modes"].append(modes_1d)
                else:
                    self.time_evolution_data[dimension_index]["modes"].append(None)

    def finalize(self):
        """
        At the end of all episodes, create an animation per dimension that shows:
            - The evolving 1D PDF as steps progress
            - Vertical lines for each GMM component's mean
            - (Optional) Vertical lines for each valid mode from `sample_modes`
            - Overlay of the actual dataset histogram (if provided)
            - Legend, annotations, etc.
        """
        colors = list(mcolors.TABLEAU_COLORS.values())

        for dimension_index, data in self.time_evolution_data.items():
            pdfs = data["pdfs"]
            means = data["means"]
            mixture_probs = data["mixture_probs"]
            modes = data["modes"]
            if len(pdfs) == 0:
                continue

            num_components = mixture_probs[0].shape[0]
            colors_selected = colors[:num_components]

            pdfs_np = np.array(pdfs)
            means_np = np.array(means)
            mixture_probs_np = np.array(mixture_probs)
            x_values = np.linspace(-self.lim, self.lim, 10000)

            fig, ax = plt.subplots(figsize=(10, 6))

            def update(frame):
                ax.clear()
                ax.plot(x_values, pdfs_np[frame], color="black", label="PDF")

                handles = [plt.Line2D([0], [0], color="black", lw=2, label="PDF")]
                for comp_idx in range(num_components):
                    mean_val = means_np[frame, comp_idx]
                    prob_val = mixture_probs_np[frame, comp_idx]
                    color = colors_selected[comp_idx % len(colors_selected)]
                    ax.axvline(
                        mean_val,
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.5,
                    )
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=color,
                            linestyle="--",
                            lw=1.5,
                            label=f"Comp {comp_idx + 1}: μ={mean_val:.4f}, π={prob_val:.2f}",
                        )
                    )

                mode_vals = modes[frame]
                if mode_vals is not None:
                    for mode_idx, mode_val in enumerate(mode_vals):
                        ax.axvline(
                            mode_val,
                            color="red",
                            linestyle="-.",
                            linewidth=1.2,
                        )
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color="red",
                            linestyle="-.",
                            lw=1.2,
                            label="Valid Mode(s)",
                        )
                    )

                ax.set_xlabel(f"Output Values (Dimension {dimension_index})")
                ax.set_ylabel("PDF")
                # ax.set_yscale("log")
                ax.set_title(f"Time Evolution of PDF for Dimension {dimension_index}")
                ax.set_xlim(-self.lim, self.lim)
                ax.set_ylim(0, None)

                if self.dataset:
                    actual_data = [
                        traj.actions[frame, dimension_index]
                        for traj in self.dataset.trajectories
                        if traj.actions.shape[0] > frame
                    ]
                    if len(actual_data) > 0:
                        ax.hist(
                            actual_data,
                            bins=50,
                            density=True,
                            alpha=0.5,
                            label="Actual Data Distribution",
                            color="gray",
                        )
                        handles.append(
                            plt.Line2D(
                                [0],
                                [0],
                                color="gray",
                                lw=1.5,
                                label="Actual Data Distribution",
                            )
                        )
                ax.legend(handles=handles, loc="upper right")

            ani = FuncAnimation(
                fig, update, frames=len(pdfs_np), interval=200, repeat=False
            )

            video_path = f"{wandb.run.dir}/dimension_{dimension_index}_evolution.mp4"
            ani.save(video_path, writer="ffmpeg")

            wandb.log(
                {
                    f"Time Evolution Video for Dimension {dimension_index}": wandb.Video(
                        video_path, format="mp4"
                    )
                }
            )
            plt.close(fig)


class EpisodeVideoLogger(Callback):
    """Logs the episode frames as a wandb video"""

    def __init__(self, fps: int = 4, **kwargs):
        super().__init__()
        self.fps = fps

    def __call__(self, inference_ep: InferenceEntrypoint):
        if len(inference_ep.episode_frames) == 0:
            raise ValueError(
                "No frames where found in the inference InferenceEntrypoint. Ensure rendering mode is not None. "
            )
        video_frames = np.stack(inference_ep.episode_frames, axis=0)
        video_path = f"{wandb.run.dir}/episode_{inference_ep.episode_counter}.mp4"
        with imageio.get_writer(video_path, fps=self.fps, macro_block_size=1) as writer:
            for frame in video_frames:
                writer.append_data(frame.astype(np.uint8))
        wandb.log(
            {
                f"video@episode_{inference_ep.episode_counter}": wandb.Video(
                    video_path, format="mp4"
                )
            }
        )


class MRTrajectoryLogger(Callback):
    """
    Logs rendered trajectories from an inference episode and the original dataset.
    Saves visualizations as PNGs and logs to WandB.
    """

    def __init__(
        self,
        dataset_cfg: Optional[MultiPathTrajectoryDatasetCfg] = None,
    ):
        super().__init__()
        self.dataset = (
            hydra.utils.instantiate(dataset_cfg) if dataset_cfg is not None else None
        )

    def plot_trajectories(
        self,
        trajectories: List,
        ax,
        env,
        plot_final_positions: bool = True,
        buffer: float = 2,
    ):
        """
        Plot trajectories on the given Axes object with environment boundaries.

        Args:
            trajectories (List): List of trajectories to plot.
            ax: Matplotlib Axes object to plot on.
            env: The environment instance for boundary details.
            plot_final_positions (bool): Whether to plot final positions as red dots.
            buffer (float): Buffer around observation bounds to ensure consistent plot dimensions.
        """
        x_min = env.obs_low_bound - buffer
        x_max = env.obs_high_bound + buffer
        y_min = env.obs_low_bound - buffer
        y_max = env.obs_high_bound + buffer

        ax.set_aspect("equal", "box")
        ax.set_facecolor("white")
        ax.axis("off")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        obs_bounds = Rectangle(
            (env.obs_low_bound, env.obs_low_bound),
            env.obs_high_bound - env.obs_low_bound,
            env.obs_high_bound - env.obs_low_bound,
            fill=False,
            edgecolor="lightgray",
            linestyle="--",
            linewidth=1,
        )
        ax.add_patch(obs_bounds)

        target_bounds = Rectangle(
            (env._target_bounds.low[0], env._target_bounds.low[1]),
            env._target_bounds.high[0] - env._target_bounds.low[0],
            env._target_bounds.high[1] - env._target_bounds.low[1],
            fill=True,
            color="#A1D99B",
            alpha=0.7,
        )
        ax.add_patch(target_bounds)

        for traj in trajectories:
            color = env.get_trajectory_label(traj)
            for i in range(len(traj) - 1):
                point, next_point = traj[i], traj[i + 1]
                ax.plot(
                    [point[0], next_point[0]],
                    [point[1], next_point[1]],
                    color=color,
                    alpha=0.5,
                    linewidth=2,
                )

            if plot_final_positions:
                final_position = traj[-1]
                ax.plot(
                    final_position[0],
                    final_position[1],
                    marker="o",
                    markersize=8,
                    color="red",
                    markeredgecolor="black",
                    markeredgewidth=1,
                )

    def log_label_stats(self, trajectories: List, env):
        """
        Compute label statistics for the trajectories and log them to WandB.

        Args:
            trajectories (List): List of trajectories to analyze.
            env: The environment instance for label determination.
        """
        label_counts = {"lightsteelblue": 0, "rosybrown": 0, "gray": 0}
        for traj in trajectories:
            label = env.get_trajectory_label(traj)
            label_counts[label] += 1
        wandb.log({"label_stats": label_counts})

    def save_and_log_plot(self, fig, filename: str, wandb_key: str):
        """
        Save a matplotlib figure as a PNG and log it to WandB.

        Args:
            fig: Matplotlib figure object to save.
            filename (str): Filename to save the figure.
            wandb_key (str): WandB key for logging the image.
        """
        output_path = f"{wandb.run.dir}/{filename}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        wandb.log({wandb_key: wandb.Image(output_path)})

    def __call__(self, inference_ep: InferenceEntrypoint):
        """
        After an inference episode, visualize both inference and dataset trajectories.
        """
        if (
            not hasattr(inference_ep.env, "_trajectories")
            or not inference_ep.env._trajectories
        ):
            raise ValueError(
                "No trajectories found in the environment. Ensure `track_trajectories=True` "
                "is set when initializing the environment."
            )

        fig, ax = plt.subplots(figsize=(8, 8))
        self.plot_trajectories(
            trajectories=inference_ep.env._trajectories,
            ax=ax,
            env=inference_ep.env,
        )
        self.save_and_log_plot(
            fig, "inference_trajectories.png", "inference_trajectories_plot"
        )
        self.log_label_stats(inference_ep.env._trajectories, inference_ep.env)
        if self.dataset is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            self.plot_trajectories(
                trajectories=[traj.states for traj in self.dataset],
                ax=ax,
                env=inference_ep.env,
            )
            self.save_and_log_plot(
                fig,
                "original_dataset_trajectories.png",
                "dataset_trajectories_plot",
            )
