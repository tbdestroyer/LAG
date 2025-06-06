import sys
import os
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
#from config import get_config
#from runner.share_jsbsim_runner import ShareJSBSimRunner # Might not be needed, check later
from envs.JSBSim.envs import SingleControlEnv, SingleCombatEnv, MultipleCombatEnv
from envs.env_wrappers import DummyVecEnv
from runner.tacview import Tacview
import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level=logging.DEBUG)

# Corrected import path
from algorithms.ppo.ppo_actor import PPOActor


# Simple arguments placeholder - can be expanded if needed
class Args:
    def __init__(self):
        self.gain = 0.01
        self.env_name = "SingleControl"
        self.scenario_name = "1/heading"
        self.algorithm_name = "ppo"
        self.experiment_name = "v1"
        self.seed = 5
        self.n_training_threads = 1
        self.n_rollout_threads = 4
        self.cuda = False # Set to True if using GPU
        self.log_interval = 1
        self.save_interval = 1
        self.num_mini_batch = 5
        self.buffer_size = 3000
        self.num_env_steps = 1e8
        self.lr = 3e-4
        self.gamma = 0.99
        self.ppo_epoch = 4
        self.clip_params = 0.2
        self.max_grad_norm = 2
        self.entropy_coef = 1e-3
        self.hidden_size = "128 128"
        self.act_hidden_size = "128 128"
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.data_chunk_length = 8
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True
        self.model_dir = 'scripts/results/SingleControl/1/heading/ppo/v1/run2' # Path to your trained model


def make_render_env(all_args):
    # Use DummyVecEnv for rendering a single environment instance
    def get_env_fn():
        def init_env():
            env = SingleControlEnv(all_args.scenario_name)
            env.seed(all_args.seed)
            return env
        return init_env
    return DummyVecEnv([get_env_fn()])


if __name__ == "__main__":
    args = Args()

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create the environment
    envs = make_render_env(args)
    obs_space = envs.observation_space
    action_space = envs.action_space

    # Initialize policy
    # Assuming the heading task uses a single agent PPOActor
    policy = PPOActor(args, obs_space, action_space, device=torch.device("cpu"))

    # Load the trained model
    model_file_path = os.path.join(args.model_dir, "actor_latest.pt")
    if not os.path.exists(model_file_path):
        print(f"Error: Model file not found at {model_file_path}")
        print("Please ensure you have trained the heading control model and the path is correct.")
        sys.exit(1)

    print(f"Loading model from {model_file_path}")
    policy.load_state_dict(torch.load(model_file_path, map_location=torch.device("cpu")))
    policy.eval()

    print("Start rendering SingleControl heading environment")

    # Rendering setup (similar to render_1v1.py)
    tacview = Tacview()
    render = True # Set to False if you don't want rendering

    obs = envs.reset()
    rnn_states = np.zeros((envs.num_envs, 1, args.recurrent_hidden_size), dtype=np.float32) # num_envs is 1 for DummyVecEnv
    masks = np.ones((envs.num_envs, 1))

    episode_rewards = 0

    while True:
        # Get actions from the policy
        with torch.no_grad():
            # Need to reshape obs for the policy if it expects (num_agents, obs_dim) or similar
            # DummyVecEnv stacks envs, so obs shape is (num_envs, num_agents, obs_dim)
            # For SingleControl, num_agents is 1, num_envs is 1, so shape is (1, 1, obs_dim)
            # Policy might expect (batch_size, obs_dim) or (num_steps, num_envs, obs_dim)
            # Let's try (num_steps, num_envs, obs_dim) which is (1, 1, obs_dim)
            # If it's a recurrent policy, rnn_states and masks are needed

            # Assuming policy expects (num_steps, num_envs, ...), where num_steps=1 for a single step prediction
            obs_tensor = torch.from_numpy(obs).float()
            rnn_states_tensor = torch.from_numpy(rnn_states).float()
            masks_tensor = torch.from_numpy(masks).float()

            # Need to adjust the forward pass based on the PPOActor implementation
            # Common signature: forward(obs, rnn_states, masks)
            # Output: (value, action, action_log_prob, rnn_states)

            # Assuming PPOActor forward takes (num_steps * num_envs, obs_shape) for non-recurrent
            # or (num_steps, num_envs, obs_shape) for recurrent
            # And returns (num_steps * num_envs, action_shape) or (num_steps, num_envs, action_shape)

            # Let's assume a simple non-recurrent policy for now and adjust if needed after a run/error
            # Reshape obs from (1, 1, obs_dim) to (1, obs_dim)
            # And actions from (1, action_dim) to (1, 1, action_dim)

            # Correct approach for (num_steps, num_envs, ...)
            # obs_tensor shape: (1, 1, obs_dim)
            # rnn_states_tensor shape: (1, 1, rnn_hidden_size)
            # masks_tensor shape: (1, 1)

            value, action, action_log_prob, rnn_states = policy.forward(obs_tensor.unsqueeze(0), rnn_states_tensor.unsqueeze(0), masks_tensor.unsqueeze(0))
            # Squeeze the extra time step dimension (0) from action and rnn_states
            action = action.squeeze(0)
            rnn_states = rnn_states.squeeze(0)

        # Reshape action back to environment expected shape (num_envs, num_agents, action_dim) which is (1, 1, action_dim)
        # The policy's output 'action' should already be (num_envs, action_dim) or (1, action_dim) in this case
        # Need to confirm the exact output shape of PPOActor's forward method.
        # Assuming it returns actions per environment, shape (num_envs, action_dim)
        # But the env expects (num_envs, num_agents, action_dim)
        # For SingleControl, num_agents is 1, so need shape (1, 1, action_dim)

        # Let's assume policy.forward returns (num_envs, action_dim) and reshape to (num_envs, 1, action_dim)
        actions_np = action.squeeze(0).cpu().numpy() # Action is (1, action_dim) from squeeze(0) above
        actions_np = actions_np[:, np.newaxis, :]

        # Step the environment
        obs, rewards, dones, infos = envs.step(actions_np)

        episode_rewards += rewards[0, 0, 0] # Assuming reward is shape (num_envs, num_agents, 1)

        # Render the environment
        if render:
            # DummyVecEnv wraps the single env, need to access the unwrapped env for render
            envs.envs[0].render(mode='real_time', tacview=tacview)

        # Check if episode is done
        if dones[0, 0]: # Assuming done is shape (num_envs, num_agents)
            print(f"Episode finished. Total reward: {episode_rewards}")
            episode_rewards = 0
            # Reset the environment
            obs = envs.reset()
            # Reset RNN states and masks for the new episode
            rnn_states = np.zeros((envs.num_envs, 1, args.recurrent_hidden_size), dtype=np.float32)
            masks = np.ones((envs.num_envs, 1))

        # You might want a way to break the loop, e.g., after a number of episodes or a key press
        # For now, it will run indefinitely, you can stop it with Ctrl+C 