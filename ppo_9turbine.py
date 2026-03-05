import argparse
import numpy as np
import torch
import matlab.engine
import os
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
        self.eng.cd(base_path, nargout=0)
        print("Paths added to MATLAB...")

        self.sim_time = 0
        self.max_steps = 1000
        # Initialize simulation
        self.eng.Initial_9(nargout=0)

        # State definition: [Yaw1..Yaw9, Normalized_Power]
        self.state_dim = 10
        self.action_dim = 9  # Yaw1..Yaw9

    def reset(self):
        self.sim_time = 0
        self.eng.Initial_9(nargout=0)
        # Initial state: 0 yaw, 0 power (or dummy value)
        return np.array([0.0] * 10)

    def step(self, action):
        """
        Action is received in range [-1, 1] from PPO.
        We scale it to physical degrees (e.g., -30 to 30).
        """
        # 1. Scale Action
        # Map [-1, 1] -> [-30, 30] degrees
        physical_yaw = action * 30.0

        # 2. Prepare MATLAB Inputs
        phi = np.array(physical_yaw, dtype=np.float64)
        CT_prime = 2 * np.ones(9, dtype=np.float64)  # Constant CT

        phi_matlab = matlab.double(phi.tolist())
        CT_prime_matlab = matlab.double(CT_prime.tolist())

        # 3. Step Simulation
        # Returns power in Watts
        power = self.eng.Timestep_9(self.sim_time, phi_matlab, CT_prime_matlab, nargout=1)
        power_vals = np.array(power).flatten()

        # 4. Construct Reward & Next State
        total_power_mw = np.sum(power_vals) / 1e6
        reward = total_power_mw
        # Construct State: [Yaw_Action_1..Yaw_Action_9, Current_Power_Output]
        next_state = np.concatenate([physical_yaw, [total_power_mw]])

        self.sim_time += 1
        done = False
        if self.sim_time >= self.max_steps:
            done = True

        return next_state, reward, done

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

    total_steps = 0

    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)

        episode_reward = 0
        done = False

        while not done:
            # Choose action (returns value in [-1, 1])
            a, a_logprob = agent.choose_action(s)

            # Execute in environment
            s_, r, done = env.step(a)

            # Normalize Next State
            if args.use_state_norm:
                s_ = state_norm(s_)

            # Store Transition
            # dw = True if dead/win, but here max_steps is just time limit, not failure
            dw = False
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)

            s = s_
            episode_reward += r
            total_steps += 1

            # Update Policy
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Logging
            if total_steps % 100 == 0:
                print(f"Step: {total_steps}, Episode Reward (Current): {episode_reward:.4f}")

        print(f"Episode Finished. Total Reward: {episode_reward:.4f}")

        # Save Models
        if total_steps % args.save_freq == 0:
            torch.save(agent.actor.state_dict(), f'actor_step_{total_steps}.pth')
            torch.save(agent.critic.state_dict(), f'critic_step_{total_steps}.pth')

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PPO for Wind Farm Control")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=50000, help="Increased for convergence")
    parser.add_argument("--save_freq", type=int, default=1000)
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