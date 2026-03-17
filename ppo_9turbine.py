import argparse
import numpy as np
import torch
import matlab.engine
import os
import csv
from ppo_agent_9turbine import PPO_Continuous, ReplayBuffer, Normalization


# -----------------------------------
# Environment Wrapper (MATLAB Interface)
# -----------------------------------
class WindFarmEnv:
    def __init__(self):
        print("Starting MATLAB Engine...")
        self.eng = matlab.engine.start_matlab()

        # Update these paths to your actual local paths
        base_path = r'/Users/akhilpatel/Desktop/Dissertation/WFSim-master'
        self.eng.addpath(os.path.join(base_path, 'layoutDefinitions'))
        self.eng.addpath(os.path.join(base_path, 'controlDefinitions'))
        self.eng.addpath(os.path.join(base_path, 'solverDefinitions'))
        self.eng.addpath(os.path.join('/Users/akhilpatel/Desktop/Dissertation', 'PPO-9-turbine-case'))
        self.eng.cd(base_path, nargout=0)
        print("Paths added to MATLAB...")

        self.sim_time = 0
        
        # NEW: The number of simulation steps to wait for the wake to plateau
        self.settling_steps = 400 
        
        # How many RL actions (choices) the agent gets per episode
        self.actions_per_episode = 5
        
        # Total MATLAB simulation steps per episode (400 * 5 = 2000)
        self.max_steps = self.settling_steps * self.actions_per_episode 

        # Initialize simulation
        self.eng.Initial_9(nargout=0)

        # State definition: [Yaw1..Yaw9, Power]
        self.state_dim = 10
        self.action_dim = 9  # Yaw1..Yaw9

        # Guard threshold for invalid solver outputs (total farm power per timestep, in MW)
        self.max_reasonable_power_mw = 60.0

    def reset(self):
        self.sim_time = 0
        self.eng.Initial_9(nargout=0)
        # Initial state: 0 yaw, 0 power
        return np.array([0.0] * 10, dtype=np.float64)

    def step(self, action):
        """
        Action is received in range [-1, 1] from PPO.
        We scale it to physical degrees (e.g., -30 to 30).
        Returns: next_state, reward, done, invalid
        """
        # 1. Scale Action
        physical_yaw = action * 30.0  # Map [-1,1] -> [-30,30] deg
        phi = np.array(physical_yaw, dtype=np.float64)
        CT_prime = 2 * np.ones(9, dtype=np.float64)

        phi_matlab = matlab.double(phi.tolist())
        CT_prime_matlab = matlab.double(CT_prime.tolist())

        total_power_mw = 0.0
        invalid = False

        # --- 2. WAKE SETTLING LOOP (Action Repeat) ---
        # Hold the yaw angles constant and step the simulator until the flow plateaus
        for _ in range(self.settling_steps):
            power = self.eng.Timestep_9(self.sim_time, phi_matlab, CT_prime_matlab, nargout=1)
            power_vals = np.array(power).flatten()

            # Guard 1: invalid raw solver output
            if power_vals.size == 0 or (not np.all(np.isfinite(power_vals))):
                invalid = True
                break

            total_power_mw = float(np.sum(power_vals) / 1e6)

            # Guard 2: invalid/absurd derived value
            if (not np.isfinite(total_power_mw)) or (total_power_mw < 0.0) or (total_power_mw > self.max_reasonable_power_mw):
                invalid = True
                break

            self.sim_time += 1
            
            # Prevent exceeding maximum episode length during the settling phase
            if self.sim_time >= self.max_steps:
                break

        # Handle Simulator Crash / Invalid Data
        if invalid:
            next_state = np.concatenate([physical_yaw, [0.0]])
            return next_state, -100.0, True, True

        # 3. Construct Reward & Next State based on the PLATEAUED power
        reward = total_power_mw
        next_state = np.concatenate([physical_yaw, [total_power_mw]])

        done = self.sim_time >= self.max_steps

        return next_state, reward, done, False

    def close(self):
        self.eng.quit()


# -----------------------------------
# Main Training Loop
# -----------------------------------

def main(args):
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize Environment and Agent
    env = WindFarmEnv()
    args.state_dim = env.state_dim
    args.action_dim = env.action_dim

    replay_buffer = ReplayBuffer(args)
    agent = PPO_Continuous(args)
    state_norm = Normalization(shape=args.state_dim)

    # Note: total_steps now counts the number of RL actions taken, NOT MATLAB timesteps.
    total_steps = 0
    episode_idx = 0
    invalid_episodes = 0

    # Always reset CSV at start of each run
    csv_path = "episode_power.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "avg_power_mw", "total_power_mw", "max_power_mw"])

    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)

        episode_reward = 0.0
        done = False
        episode_invalid = False
        power_history = []

        while not done and total_steps < args.max_train_steps:
            # Safety: avoid NaN/Inf state reaching policy
            if not np.all(np.isfinite(s)):
                s = np.zeros_like(s)

            # Choose action (returns value in [-1, 1])
            a, a_logprob = agent.choose_action(s)

            # Execute in environment (This now takes 400 MATLAB steps internally)
            s_raw, r, done, invalid = env.step(a)

            if invalid:
                # Do not normalize/store invalid transition
                episode_reward += r
                episode_invalid = True
                done = True
                break

            # Save power from raw next state (UN-normalized)
            power_history.append(s_raw[9])

            # Normalize Next State
            s_ = s_raw
            if args.use_state_norm:
                s_ = state_norm(s_)

            # Store Transition
            dw = False
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)

            s = s_
            episode_reward += r
            total_steps += 1

            # Update Policy
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Console logging
            if total_steps % 10 == 0:
                print(f"Agent Step: {total_steps}, Episode Reward (Current): {episode_reward:.4f}")

        if episode_invalid:
            invalid_episodes += 1
            print(f"[WARN] Episode {episode_idx} aborted due to invalid solver output. Aborted so far: {invalid_episodes}")

        print(f"Episode {episode_idx} Finished. Total Reward (Sum of Plateaus): {episode_reward:.4f}")

        # Episode-end power logging
        power_array = np.array(power_history, dtype=np.float64)
        avg_power_mw = float(power_array.mean()) if power_array.size > 0 else 0.0
        total_power_mw = float(power_array.sum()) if power_array.size > 0 else 0.0
        max_power_mw = float(power_array.max()) if power_array.size > 0 else 0.0

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode_idx, avg_power_mw, total_power_mw, max_power_mw])

        episode_idx += 1

        # Save Models
        if total_steps > 0 and total_steps % args.save_freq == 0:
            torch.save(agent.actor.state_dict(), f'actor_step_{total_steps}.pth')
            torch.save(agent.critic.state_dict(), f'critic_step_{total_steps}.pth')

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PPO for Wind Farm Control")
    parser.add_argument("--seed", type=int, default=10)
    
    # max_train_steps now refers to the number of times the AGENT makes a decision.
    # 30,000 steps = 6,000 episodes (at 5 actions/episode). Adjust as needed!
    parser.add_argument("--max_train_steps", type=int, default=30000)
    parser.add_argument("--save_freq", type=int, default=1000)
    
    # Run 2 Hyperparameters
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--mini_batch_size", type=int, default=128)
    parser.add_argument("--hidden_width", type=int, default=128)
    parser.add_argument("--lr_a", type=float, default=2e-4) # learning rate for actor
    parser.add_argument("--lr_c", type=float, default=1e-4) # learning rate for critic
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--K_epochs", type=int, default=8)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--use_state_norm", type=bool, default=True)
    parser.add_argument("--use_lr_decay", type=bool, default=True)

    args = parser.parse_args()
    main(args)
