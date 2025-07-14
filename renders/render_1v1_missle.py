import numpy as np
import torch
from runner.tacview import Tacview
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level=logging.WARNING)
import matplotlib.pyplot as plt
from captum.attr._core.integrated_gradients import IntegratedGradients
from sklearn.decomposition import PCA 
import hdbscan
from renders.render_utils import *
import PySimpleGUI as sg
import time
from scipy.stats import mode

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

# Initialize live line plot
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
num_agents = 2
render = True
ego_policy_index = 1040
enm_policy_index = 1040
episode_rewards = 0
ego_run_dir = "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/run4"
enm_run_dir = "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/run4"
#ego_run_dir = "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/run3"
#enm_run_dir = "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/run3"
experiment_name = ego_run_dir.split('/')[-4]

env = SingleCombatEnv("1v1/ShootMissile/HierarchySelfplay")
env.seed(0)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
ego_policy.eval()
enm_policy.eval()
#ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_4141.pt"))
#enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_4156.pt"))
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_520.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_600.pt"))


print("Start render")
obs = env.reset()
tacview = Tacview()
if render:
    env.render(mode='real_time', tacview=tacview)
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents // 2, 1))
enm_obs =  obs[num_agents // 2:, :]
ego_obs =  obs[:num_agents // 2, :]
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
data_rows = []
enm_data_rows = []

def extract_facts(obs, prev_obs=None):
    facts = []
    # Adjust indices to match your feature_labels order!
    # Example indices (update as needed):
    """
    "ego_altitude (5km)", "ego_roll_sin", "ego_roll_cos", "ego_pitch_sin", "ego_pitch_cos",
    "X Velocity (mh)", "Y Velocity (mh)", "Z Velocity (mh)", "Vertical Speed (mh)",
    "delta_v_body_x (mh)", "delta_altitude (km)", "Angle-OFf (rad)", "Target Aspect (rad)",
    "relative_distance (10km)", "side_flag", "missile delta_v_body_x", 
            "missile delta altitude",
            "missile ego_AO",
            "missile ego_TA",
            "missile relative distance",
            "missile side flag"
            """
    missile_distance_idx = feature_labels.index("missile relative distance")
    angle_off_idx = feature_labels.index("Angle-OFf (rad)")
    altitude_idx = feature_labels.index("ego_altitude (5km)")
    x_vel_idx = feature_labels.index("X Velocity (mh)")
    y_vel_idx = feature_labels.index("Y Velocity (mh)")
    z_vel_idx = feature_labels.index("Z Velocity (mh)")
    x_vel = obs[x_vel_idx] * 2000  # adjust scaling if needed
    y_vel = obs[y_vel_idx] * 2000
    z_vel = obs[z_vel_idx] * 2000
    delta_altitude_idx = feature_labels.index("delta_altitude (km)")
    speed = (x_vel**2 + y_vel**2 + z_vel**2) ** 0.5
    relative_distance = feature_labels.index("relative_distance (10km)")
    target_aspect = feature_labels.index("Target Aspect (rad)")
    missile_relative_distance = feature_labels.index("missile relative distance")
    missile_delta_v_body_x = feature_labels.index("missile delta_v_body_x")
    missile_side_flag = feature_labels.index("missile side flag")
    # Example thresholds (update as needed):
    
    '''
    if obs[missile_distance_idx] < 0.08:  # 0.08*10km = 800m
        facts.append("missile_threat")
    if abs(obs[angle_off_idx]) < 0.175:  # ~10 degrees in radians
        facts.append("enemy_in_sights")
    else:
        facts.append("enemy_out_of_sights")
    if obs[altitude_idx] < 0.5:  # 0.5*10km = 5000m
        facts.append("altitude_low")
    else:
        facts.append("altitude_fine")
    if speed < 206:  # 0.103*2000mh = 206mh ≈ 200 knots
        facts.append("low_speed")
    elif speed > 300:  # 0.15*2000mh = 300mh ≈ 300 knots
        facts.append("good_speed")
    if obs[z_vel_idx] > 0: # z velocity > 0 means climbing
        facts.append("climbing for energy advantage")
    if -np.pi/2 <= obs[angle_off_idx] < np.pi/2:
         facts.append("enemy_in_front")
    else:
         facts.append("enemy_behind")
    if prev_obs is not None:
        if obs[delta_altitude_idx] > prev_obs[delta_altitude_idx]:
            facts.append("delta_altitude_increasing")
        elif obs[delta_altitude_idx] < prev_obs[delta_altitude_idx]:
            facts.append("delta_altitude_decreasing")
    
    '''
    '''
    if obs[relative_distance] < 0.15 and obs[angle_off_idx] < 0.52 and obs[target_aspect] > 2.8:
        facts.append("Offesnive Kill Oppurtunity")
    if obs[relative_distance] < 0.15 and obs[angle_off_idx] > 2.5 and obs[target_aspect] < .3 :
        facts.append("Defensive Threat")
    if obs[missile_relative_distance] < 0.2 and obs[missile_delta_v_body_x] > 0 and obs[missile_side_flag] > 0:
        facts.append("missile threat from right")
    if obs[missile_relative_distance] < 0.2 and obs[missile_delta_v_body_x] > 0 and obs[missile_side_flag] < 0:
        facts.append("missile threat from left")
    '''
    
    return facts


# Initialize the dataset for storing observations, actions, and attributions
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
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    if render:
        env.render(mode='real_time',tacview=tacview)
    if dones.all():
        print(infos)
        print(bloods)
        break
    #----------------Integrated Gradients----------------
    input = torch.tensor(ego_obs, dtype=torch.float32, device=torch.device("cpu"), requires_grad=True)
    baseline = torch.zeros_like(input)
    ig = IntegratedGradients(model_forward, False)
    attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
#--------------------------------------

    attr = attributions[0].detach().cpu().numpy()  # shape: (15,)
    all_attributions = []
    all_attributions.append(attr)
    
    # Update green bars
    for i, bar in enumerate(green_bars):
        bar.set_width(attr[i])
        bar.set_x(0)
        # Optional: set color based on sign
        bar.set_color('green' if attr[i] >= 0 else 'lime')
    fig.canvas.draw_idle()
    plt.pause(0.01)
    
        
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    print(f"step:{env.current_step}, bloods:{bloods} Attributions:{attributions} Delta:{delta}")
    enm_obs =  obs[num_agents // 2:, ...]
    ego_obs =  obs[:num_agents // 2, ...]
    print(ego_obs)
    print(ego_actions)
    print(attr)
    data_rows.append(np.hstack((ego_obs[0], ego_actions[0], attr)))
    enm_data_rows.append(enm_obs[0])
    data_matrix = np.vstack(data_rows)

    

print(episode_rewards)
print(bloods)
pca = PCA(n_components=5).fit(data_matrix)
#print("Cumulative explained variance ratio:", np.cumsum(pca.explained_variance_ratio_))
data_pca = pca.transform(data_matrix)

# After you have data_pca from PCA
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)  # You can tune min_cluster_size
cluster_labels = clusterer.fit_predict(data_pca)

# Plot cluster assignments over time
plt.figure(figsize=(12, 3))
plt.plot(cluster_labels, drawstyle='steps-post')
plt.xlabel('Timestep')
plt.ylabel('Behavioral Stage (Cluster)')
plt.title('HDBSCAN Behavioral Stages Over Time')
plt.show()
plt.savefig("behavioral_stages_over_time.png")

# Optional: Visualize clusters in PCA space
plt.figure()
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster_labels, cmap='tab10', s=10)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('HDBSCAN Clusters in PCA Space')
plt.colorbar(label='Cluster')
plt.show()
plt.savefig("clusters_in_pca_space.png")

explanation_log = []
step_size = 0.2 
prev_obs = None 

for i in range(len(data_matrix)):
    t = i * step_size
    obs = data_matrix[i][:len(feature_labels)]
    action_idx = np.argmax(data_matrix[i][len(feature_labels):len(feature_labels)+4])  # adjust if not one-hot
    action_name = f"action_{action_idx}"  # Replace with your action mapping if available
    attr_start = len(feature_labels) + 4
    attr = data_matrix[i][attr_start:]
    # Get top 2 IG features
    # attr = all_attributions[i]
    top_ig_idx = np.argsort(np.abs(attr))[::-1]
    print("Top integrated gradients are ", top_ig_idx)
    top_ig_features = [feature_labels[j] for j in top_ig_idx[:2]]
    facts = extract_facts(obs,prev_obs)
    cluster = cluster_labels[i]
    explanation_log.append({
        "t": t,
        "action": action_name,
        "top_IG_features": top_ig_features,
        "facts": facts,
        "cluster": cluster
    })
    prev_obs = obs
    all_attributions.append(attr)
# Calculate step size in seconds
total_steps = len(cluster_labels)
step_size = 0.2

# Generate time labels in seconds
seconds = np.arange(total_steps) * step_size
time_labels = [sec_to_minsec(sec) for sec in seconds]
def mode_filter(sequence, window_size=5):
    half = window_size // 2
    padded = np.pad(sequence, (half, half), mode='edge')
    smoothed = [
        mode(padded[i:i + window_size], keepdims=False).mode
        for i in range(len(sequence))
    ]
    return np.array(smoothed)

smoothed_mode = mode_filter(cluster_labels, window_size=30)
#50 was good
# Plot with minute:second x-axis labels
plt.figure(figsize=(14, 3))
plt.plot(cluster_labels, label = 'Original', drawstyle='steps-post')
plt.plot(smoothed_mode, label='Mode Filter', linewidth=2)
plt.xlabel('Time (min:sec)')
plt.ylabel('Behavioral Stage (Cluster)')
plt.title('HDBSCAN Behavioral Stages Over Time (min:sec)')
plt.legend()

# Set x-ticks to every Nth label for readability
N = max(1, total_steps // 15)
plt.xticks(np.arange(0, total_steps, N), [time_labels[i] for i in range(0, total_steps, N)], rotation=45)

plt.tight_layout()
plt.show()
plt.savefig("behavioral_stages_over_time_minsec.png")

# Output cluster information and relevant explainability
for i, entry in enumerate(explanation_log):
    if i == 0 or entry["cluster"] != explanation_log[i-1]["cluster"]:
        prev = explanation_log[i-1] if i > 0 else None
        print(f'Agent entered behavior stage {entry["cluster"]} at {entry["t"]:.1f}s '
              f'due to spikes in {", ".join(entry["top_IG_features"])}.')
    print(rationale(entry))

cluster_times = get_cluster_times(explanation_log)
top_integrated_gradients_in_cluster = get_most_frequent_integrated_gradients_in_cluster(explanation_log)
#layout = [[sg.Text('', key='-WORD-', font=('Arial', 48), size=(15,1),justification='center')]]
#window = sg.Window('Live Status',layout, finalize=True, keep_on_top=True, no_titlebar=True)  

for cluster, times in cluster_times.items():
    print(f'Cluster {cluster} is from {times[0]} to {times[1]}')

for cluster in top_integrated_gradients_in_cluster:
    first = top_integrated_gradients_in_cluster[cluster]["first"]
    second = top_integrated_gradients_in_cluster[cluster]["second"]
    print(f'Cluster {cluster} has top features {first} (first) and {second} (second)')
    print()
    if (first) == 'missile relative distance' and (second) == 'missile ego_AO' :
        print('Navigating...')
    if (first) == 'missile relative distance' and (second) == 'Z Velocity (mh)' :
        print('Dodging...')
    if (first) == 'ego_pitch_sin' and (second) == 'Z Velocity (mh)' :
        print('Positioning...')
    if (first) == 'ego_pitch_sin' and (second) =='delta_altitude (km)' :
        print('Attacking...')
    

label = feature_labels.index("missile relative distance")
print("Enemy missile launch at start" if data_matrix[0][label] != 0 else "")
for i in range(1, len(data_matrix)):
    label = feature_labels.index("missile relative distance")
    if data_matrix[i - 1][label] == 0 and data_matrix[i][label] != 0:
        print("Enemy missile launch at ", steps_to_minsec(i, step_size))
    if data_matrix[i - 1][label] != 0 and data_matrix[i][label] == 0:
        print("Enemy missile detonation at ", steps_to_minsec(i, step_size))

for i in range(1, len(enm_data_rows)):
    label = feature_labels.index("missile relative distance")
    if enm_data_rows[i - 1][label] == 0 and enm_data_rows[i][label] != 0:
        print("Self missile launch at ", steps_to_minsec(i, step_size))
    if enm_data_rows[i - 1][label] != 0 and enm_data_rows[i][label] == 0:
        print("Self missile detonation at ", steps_to_minsec(i, step_size))
#Plot Label for Top-IG Features
