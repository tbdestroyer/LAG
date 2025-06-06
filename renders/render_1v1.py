import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
import captum
from captum.attr._core.integrated_gradients import IntegratedGradients
from runner.tacview import Tacview
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.WARNING)
 
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

def _t2n(x):
    return x.detach().cpu().numpy()

num_agents = 2
render = True
ego_policy_index = 44
enm_policy_index = 44
episode_rewards = 0
ego_run_dir = "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1"
enm_run_dir = "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1"
experiment_name = ego_run_dir.split('/')[-4]

env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
env.seed(0)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
ego_policy.eval()
enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))

#------------------Graphical rendering setup------------------
feature_labels = [
    "ego_altitude (5km)", "ego_roll_sin", "ego_roll_cos", "ego_pitch_sin", "ego_pitch_cos",
    "X Velocity (mh)", "Y Velocity (mh)", "Z Velocity (mh)", "Vertical Speed (mh)",
    "delta_v_body_x (mh)", "delta_altitude (km)", "Angle-OFf (rad)", "Target Aspect (rad)",
    "relative_distance (10km)", "side_flag"
]

# Initialize live line plot
plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(feature_labels))

# Draw red background bars (full range)
red_bars = ax.barh(y_pos, [2]*len(feature_labels), left=[-1]*len(feature_labels), color='red', alpha=0.3)

# Draw green bars for live values (initialized at 0)
green_bars = ax.barh(y_pos, [0]*len(feature_labels), left=[0]*len(feature_labels), color='green', alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(feature_labels)
ax.set_xlim(-1, 1)
ax.set_xlabel("Attribution Score")
ax.set_title("Live Integrated Gradients Attribution (Bar Progress Style)")
plt.tight_layout()
plt.show()

tacview = Tacview()
print("Start render")
obs = env.reset()
if render:
    env.render(mode='real_time', tacview=tacview)
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents // 2, 1))
enm_obs =  obs[num_agents // 2:, :]
ego_obs =  obs[:num_agents // 2, :]
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
while True:
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
    enm_actions = _t2n(enm_actions)
    enm_rnn_states = _t2n(enm_rnn_states)
    actions = np.concatenate((ego_actions, enm_actions), axis=0)
    # Obser reward and next obs
    obs, rewards, dones, infos = env.step(actions)
    rewards = rewards[:num_agents // 2, ...]
    episode_rewards += rewards
    if render:
        env.render(mode='real_time', tacview=tacview)
    if dones.all():
        print(infos)
        break
     #----------------Integrated Gradients----------------
    input = torch.tensor(ego_obs, dtype=torch.float32, device=torch.device("cpu"), requires_grad=True)
    baseline = torch.zeros_like(input)
    ig = IntegratedGradients(model_forward, False)
    attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
#--------------------------------------
    attr = attributions[0].detach().cpu().numpy()  # shape: (15,)
    # Update green bars
    for i, bar in enumerate(green_bars):
        bar.set_width(attr[i])
        bar.set_x(0 if attr[i] >= 0 else attr[i])
        # Optional: set color based on sign
        bar.set_color('green' if attr[i] >= 0 else 'lime')
    fig.canvas.draw_idle()
    plt.pause(0.01)
    
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    #print(f"step:{env.current_step}, bloods:{bloods}, episode_rewards:{episode_rewards}")
    print(f"step:{env.current_step}, bloods:{bloods}, episode_rewards:{episode_rewards}, Attributions:{attributions} Delta:{delta} Obs:{obs}")
    enm_obs =  obs[num_agents // 2:, ...]
    enm_obs =  obs[num_agents // 2:, ...]
    ego_obs =  obs[:num_agents // 2, ...]
    