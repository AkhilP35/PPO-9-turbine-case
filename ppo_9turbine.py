import argparse
import numpy as np
import torch
import matlab.engine
import os
import csv
from ppo_agent_9turbine import PPO_Continuous, ReplayBuffer, Normalization


class WindFarmEnv:
    """
    State = [yaw1..yaw9, farm_power_mw, wind_speed, sin(wind_dir), cos(wind_dir)]
    wind_dir uses mathematical convention: angle of flow vector, CCW from +x.
    """
    def __init__(self, args):
        print("Starting MATLAB Engine...")
        self.eng = matlab.engine.start_matlab()

        base_path = r'/Users/akhilpatel/Desktop/Dissertation/WFSim-master'
        self.eng.addpath(os.path.join(base_path, 'layoutDefinitions'))
        self.eng.addpath(os.path.join(base_path, 'controlDefinitions'))
        self.eng.addpath(os.path.join(base_path, 'solverDefinitions'))
        self.eng.addpath(os.path.join('/Users/akhilpatel/Desktop/Dissertation', 'PPO-9-turbine-case'))
        self.eng.cd(base_path, nargout=0)
        print("Paths added to MATLAB...")

        self.sim_time = 0
        self.max_steps = args.episode_steps

        # Wind randomization ranges
        self.wind_speed_min = args.wind_speed_min
        self.wind_speed_max = args.wind_speed_max
        self.wind_dir_min = args.wind_dir_min
        self.wind_dir_max = args.wind_dir_max

        self.wind_speed = None
        self.wind_dir_deg = None

        self.action_dim = 9
        self.state_dim = 9 + 1 + 1 + 2  # yaw9 + power + speed + sin/cos(dir)

        # Initialize once
        self._sample_wind()
        self.eng.Initial_9(float(self.wind_speed), float(self.wind_dir_deg), nargout=0)

    def _sample_wind(self):
        self.wind_speed = float(np.random.uniform(self.wind_speed_min, self.wind_speed_max))
        self.wind_dir_deg = float(np.random.uniform(self.wind_dir_min, self.wind_dir_max))

    def _wind_features(self):
        theta = np.deg2rad(self.wind_dir_deg)
        return np.array([self.wind_speed, np.sin(theta), np.cos(theta)], dtype=np.float64)

    def reset(self):
        self.sim_time = 0
        self._sample_wind()
        self.eng.Initial_9(float(self.wind_speed), float(self.wind_dir_deg), nargout=0)

        yaw0 = np.zeros(9, dtype=np.float64)
        power0 = 0.0  # keep 0 at reset; first real power comes after step 0
        return np.concatenate([yaw0, [power0], self._wind_features()])

    def step(self, action):
        physical_yaw = action * 30.0

        phi = np.array(physical_yaw, dtype=np.float64)
        CT_prime = 2 * np.ones(9, dtype=np.float64)

        phi_matlab = matlab.double(phi.tolist())
        CT_prime_matlab = matlab.double(CT_prime.tolist())

        power = self.eng.Timestep_9(self.sim_time, phi_matlab, CT_prime_matlab, nargout=1)
        power_vals = np.array(power).flatten()

        total_power_mw = np.sum(power_vals) / 1e6
        reward = total_power_mw

        next_state = np.concatenate([physical_yaw, [total_power_mw], self._wind_features()])

        self.sim_time += 1
        done = self.sim_time >= self.max_steps
        return next_state, reward, done

    def close(self):
        self.eng.quit()


def evaluate_baseline(env, state_norm):
    s = env.reset()
    if state_norm is not None:
        s = state_norm(s, update=False)

    done = False
    powers = []
    while not done:
        a = np.zeros(env.action_dim, dtype=np.float64)
        s_, r, done = env.step(a)
        powers.append(s_[9])
        if state_norm is not None:
            s_ = state_norm(s_, update=False)
        s = s_
    arr = np.array(powers)
    return float(arr.mean())


def evaluate_agent(env, agent, state_norm):
    s = env.reset()
    if state_norm is not None:
        s = state_norm(s, update=False)

    done = False
    powers = []
    while not done:
        a = agent.choose_action_deterministic(s)
        s_, r, done = env.step(a)
        powers.append(s_[9])
        if state_norm is not None:
            s_ = state_norm(s_, update=False)
        s = s_
    arr = np.array(powers)
    return float(arr.mean())


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = WindFarmEnv(args)
    args.state_dim = env.state_dim
    args.action_dim = env.action_dim

    replay_buffer = ReplayBuffer(args)
    agent = PPO_Continuous(args)
    state_norm = Normalization(shape=args.state_dim) if args.use_state_norm else None

    total_steps = 0
    episode_idx = 0

    csv_path = "episode_metrics.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "wind_speed",
                "wind_dir_deg",
                "train_avg_power_mw",
                "train_sum_power_mw",
                "train_max_power_mw",
                "baseline_avg_power_mw",
                "eval_avg_power_mw",
                "eval_improvement_pct",
                "total_steps",
            ])

    while total_steps < args.max_train_steps:
        # ---- TRAIN EPISODE (stochastic) ----
        s = env.reset()
        if state_norm is not None:
            s = state_norm(s, update=True)

        done = False
        powers = []
        episode_reward = 0.0

        while not done:
            a, a_logprob = agent.choose_action(s)
            s_, r, done = env.step(a)
            powers.append(s_[9])

            if state_norm is not None:
                s_ = state_norm(s_, update=True)

            dw = False
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)

            s = s_
            episode_reward += r
            total_steps += 1

            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            if total_steps % 100 == 0:
                print(f"Step: {total_steps}, Episode Reward (Current): {episode_reward:.4f}")

        arr = np.array(powers)
        train_avg = float(arr.mean())
        train_sum = float(arr.sum())
        train_max = float(arr.max())

        # ---- PERIODIC EVAL (baseline + deterministic agent) ----
        baseline_avg = np.nan
        eval_avg = np.nan
        eval_impr = np.nan

        if episode_idx % args.eval_freq_episodes == 0:
            # Ensure baseline and eval see the SAME wind draw by reseeding before each reset
            eval_seed = args.seed + episode_idx
            np.random.seed(eval_seed)
            baseline_avg = evaluate_baseline(env, state_norm)

            np.random.seed(eval_seed)
            eval_avg = evaluate_agent(env, agent, state_norm)

            if baseline_avg > 1e-9:
                eval_impr = 100.0 * (eval_avg - baseline_avg) / baseline_avg

            print(
                f"[EVAL ep={episode_idx}] wind_speed={env.wind_speed:.2f} m/s, "
                f"wind_dir={env.wind_dir_deg:.1f} deg | "
                f"baseline={baseline_avg:.3f} MW, eval={eval_avg:.3f} MW, "
                f"impr={eval_impr:.2f}%"
            )

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode_idx,
                float(env.wind_speed),
                float(env.wind_dir_deg),
                train_avg,
                train_sum,
                train_max,
                baseline_avg,
                eval_avg,
                eval_impr,
                total_steps,
            ])

        episode_idx += 1

        if total_steps % args.save_freq == 0:
            torch.save(agent.actor.state_dict(), f"actor_step_{total_steps}.pth")
            torch.save(agent.critic.state_dict(), f"critic_step_{total_steps}.pth")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PPO for Wind Farm Control")

    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--save_freq", type=int, default=1000)

    parser.add_argument("--episode_steps", type=int, default=1000)
    parser.add_argument("--eval_freq_episodes", type=int, default=5)

    # Wind randomization
    parser.add_argument("--wind_speed_min", type=float, default=6.0)
    parser.add_argument("--wind_speed_max", type=float, default=12.0)
    parser.add_argument("--wind_dir_min", type=float, default=0.0)
    parser.add_argument("--wind_dir_max", type=float, default=360.0)

    # PPO hyperparameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--hidden_width", type=int, default=128)
    parser.add_argument("--lr_a", type=float, default=3e-4)
    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--K_epochs", type=int, default=10)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    parser.add_argument("--use_state_norm", type=bool, default=True)
    parser.add_argument("--use_lr_decay", type=bool, default=True)

    args = parser.parse_args()
    main(args)
