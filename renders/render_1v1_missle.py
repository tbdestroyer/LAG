import numpy as np
import torch
from runner.tacview import Tacview
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level=logging.WARNING)
#import matplotlib.pyplot as plt
from captum.attr._core.integrated_gradients import IntegratedGradients

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True



    
    
def _t2n(x):
    return x.detach().cpu().numpy()



def test_plane(model_path, number_of_episodes, algorithm):
    def model_forward(ego_obs):
        batch_size = ego_obs.shape[0]
        #obs: input tensor from Captum (shape should match ego_obs)
        # Use the latest ego_rnn_states and masks from the main loop
        rnn_states = torch.tensor(ego_rnn_states, dtype=torch.float32)
        masks_ = torch.tensor(masks, dtype=torch.float32)
        if rnn_states.shape[0] != batch_size:
            rnn_states = rnn_states.expand(batch_size, *rnn_states.shape[1:]).contiguous()
        if masks_.shape[0] != batch_size:
            masks_ = masks_.expand(batch_size, *masks_.shape[1:]).contiguous()
            ego_policy.to(ego_obs.device)
        # Get the action logits or probabilities from the policy
        _, action_log_probs, _ = ego_policy(ego_obs, rnn_states, masks_, deterministic=True)
        # Return the output you want to attribute (e.g., actions)
        return action_log_probs
    D = []
    values = []
    feature_labels = [
        "ego_altitude (5km)", "ego_roll_sin", "ego_roll_cos", "ego_pitch_sin", "ego_pitch_cos",
        "X Velocity (mh)", "Y Velocity (mh)", "Z Velocity (mh)", "Vertical Speed (mh)",
        "delta_v_body_x (mh)", "delta_altitude (km)", "Angle-OFf (rad)", "Target Aspect (rad)",
        "relative_distance (10km)", "side_flag", "missile delta_v_body_x", 
                "missile delta altitude",
                "missile ego_AO",
                "missile ego_TA",
                "missile relative distance",
                "missile side flag"
            
        ]
    tacview = Tacview()
    ego_num = [
        520, 520
    ]
    enm_num = [
        520, 1040
    ]
    for i in range(number_of_episodes):

        # Initialize live line plot
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(feature_labels))
        # Draw red background bars (full range)

        red_bars = ax.barh(y_pos, [2]*len(feature_labels), left=[-1]*len(feature_labels), color='red', alpha=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Attribution Score")
        ax.set_title("Live Integrated Gradients Attribution (Bar Progress Style)")
        plt.tight_layout()
        plt.show()

        # Draw green bars for live values (initialized at 0)
        green_bars = ax.barh(y_pos, [0]*len(feature_labels), left=[0]*len(feature_labels), color='green', alpha=0.8)
        """
        num_agents = 2
        render = True
        ego_policy_index = 1040
        enm_policy_index = 1040
        episode_rewards = 0
        ego_run_dir = "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/run3"
        enm_run_dir = "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/run3"
        experiment_name = ego_run_dir.split('/')[-4]

        env = SingleCombatEnv("1v1/ShootMissile/HierarchySelfplay")
        env.seed(0)
        args = Args()

        ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
        enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
        ego_policy.eval()
        enm_policy.eval()
        if number_of_episodes == 1:
            ego_policy.load_state_dict(torch.load(ego_run_dir+ f"/actor_520.pt"))
            enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_1040.pt"))
        else:
            ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_num[i]}.pt"))
            enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_num[i]}.pt"))

        print("Start render")
        obs = env.reset()
        if render:
            env.render(mode="real_time", tacview=tacview)
        ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
        masks = np.ones((num_agents // 2, 1))
        enm_obs =  obs[num_agents // 2:, :]
        ego_obs =  obs[:num_agents // 2, :]
        enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
        current_D = {'states': [], 'actions': [], 'entropy': [], 'dones': [], 'rewards': [], 'next_states': []}
        current_values = []
        while True:
            #Store current state before action
            prev_ego_obs = ego_obs.copy()
            
            # value is action_log_probs from PPOActor.forward()
            ego_actions, value, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
            ego_actions = _t2n(ego_actions)
            value = _t2n(value)
            ego_rnn_states = _t2n(ego_rnn_states)
            probabilities = np.exp(value)
            entropy = -np.sum(probabilities * value, axis=-1)

            enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
            action_dist = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=False)
            # action_log_probs, dist_entropy = ego_policy.evaluate_actions(obs, ego_rnn_states, ego_actions, masks, active_masks=None)
            # action_log_probs, dist_entropy = ego_policy.act.evaluate_actions(ego_policy.base(ego_obs), ego_actions)
            # ego_entropy = action_dist.entropy().detach().cpu().numpy()
            enm_actions = _t2n(enm_actions)
            enm_rnn_states = _t2n(enm_rnn_states)
            actions = np.concatenate((ego_actions, enm_actions), axis=0)
            
            # Obser reward and next obs
            obs, rewards, dones, infos = env.step(actions)
            rewards = rewards[:num_agents // 2, ...]
            episode_rewards += rewards
            bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
            next_ego_obs = obs[:num_agents // 2, :]

            if render:
                env.render(mode='real_time',tacview=tacview)
            if dones.all():
                print(infos)
                print(bloods)
                current_D["states"].append(prev_ego_obs[0].reshape(-1))
                current_D["actions"].append(ego_actions[0])
                current_D["entropy"].append(entropy[0])
                current_D["dones"].append(True)
                current_D["rewards"].append(rewards[0])
                current_D["next_states"].append(next_ego_obs[0].reshape(-1))
                print("Collected data lengths:")
                
                print("States:", len(current_D["states"]))
                print("Actions:", len(current_D["actions"]))
                print("Entropies:", len(current_D["entropy"]))
                print("Rewards:", len(current_D["rewards"]))
                break
            
            current_D["states"].append(prev_ego_obs[0].reshape(-1))
            current_D["actions"].append(ego_actions[0].reshape(-1))
            current_D["entropy"].append(entropy[0])
            current_D["dones"].append(False)
            current_D["rewards"].append(rewards[0])
            current_D["next_states"].append(next_ego_obs[0].reshape(-1))
            current_values.append(value)
            
            #----------------Integrated Gradients----------------
            input = torch.tensor(ego_obs, dtype=torch.float32, device=torch.device("cpu"), requires_grad=True)
            baseline = torch.zeros_like(input)
            ig = IntegratedGradients(model_forward, False)
            attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
        #--------------------------------------
            attr = attributions[0].detach().cpu().numpy()  # shape: (15,)
            """
            # Update green bars
            for i, bar in enumerate(green_bars):
                bar.set_width(attr[i])
                bar.set_x(0 if attr[i] >= 0 else attr[i])
                # Optional: set color based on sign
                bar.set_color('green' if attr[i] >= 0 else 'lime')
            fig.canvas.draw_idle()
            plt.pause(0.01)
            """
            bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
            print(f"step:{env.current_step}, bloods:{bloods} Attributions:{attributions} Entropy:{entropy}")
            enm_obs =  obs[num_agents // 2:, ...]
            ego_obs =  obs[:num_agents // 2, ...]

        print(episode_rewards)
        print(bloods)
        D.append(current_D)
        values.append(current_values)

    return D, ego_policy, len(feature_labels), 4, values, feature_labels