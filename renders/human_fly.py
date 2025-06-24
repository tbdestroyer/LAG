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
import numpy as np
from gymnasium import spaces
from envs.JSBSim.tasks.task_base import BaseTask
from envs.JSBSim.reward_functions import AltitudeReward, HeadingReward
from envs.JSBSim.termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading
 
class HumanSingleCombatTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            HeadingReward(self.config),
            AltitudeReward(self.config),
        ]
        self.termination_conditions = [
            # UnreachHeading(self.config),
            # ExtremeState(self.config),
            # Overload(self.config),
            # LowAltitude(self.config),
            Timeout(self.config)
        ]

    @property
    def num_agents(self):
        return 2

    def load_variables(self):
        self.state_var = [
            c.delta_altitude,                   # 0. delta_h   (unit: m)
            c.delta_heading,                    # 1. delta_heading  (unit: °)
            c.delta_velocities_u,               # 2. delta_v   (unit: m/s)
            c.position_h_sl_m,                  # 3. altitude  (unit: m)
            c.attitude_roll_rad,                # 4. roll      (unit: rad)
            c.attitude_pitch_rad,               # 5. pitch     (unit: rad)
            c.velocities_u_mps,                 # 6. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 7. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 8. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                # 9. vc        (unit: m/s)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(12,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.

        observation(dim 12):
            0. ego delta altitude      (unit: km)
            1. ego delta heading       (unit rad)
            2. ego delta velocities_u  (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego v_body_x            (unit: mh)
            9. ego v_body_y            (unit: mh)
            10. ego v_body_z           (unit: mh)
            11. ego_vc                 (unit: mh)
        """
        obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
        norm_obs = np.zeros(12)
        norm_obs[0] = obs[0] / 1000         # 0. ego delta altitude (unit: 1km)
        norm_obs[1] = obs[1] / 180 * np.pi  # 1. ego delta heading  (unit rad)
        norm_obs[2] = obs[2] / 340          # 2. ego delta velocities_u (unit: mh)
        norm_obs[3] = obs[3] / 5000         # 3. ego_altitude   (unit: 5km)
        norm_obs[4] = np.sin(obs[4])        # 4. ego_roll_sin
        norm_obs[5] = np.cos(obs[4])        # 5. ego_roll_cos
        norm_obs[6] = np.sin(obs[5])        # 6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[5])        # 7. ego_pitch_cos
        norm_obs[8] = obs[6] / 340          # 8. ego_v_north    (unit: mh)
        norm_obs[9] = obs[7] / 340          # 9. ego_v_east     (unit: mh)
        norm_obs[10] = obs[8] / 340         # 10. ego_v_down    (unit: mh)
        norm_obs[11] = obs[9] / 340         # 11. ego_vc        (unit: mh)
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        print(action)
        print(self.action_space.nvec)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act
    
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
        rnn_states = torch.tensor(enm_rnn_states, dtype=torch.float32)
        masks_ = torch.tensor(masks, dtype=torch.float32)
        if rnn_states.shape[0] != batch_size:
            rnn_states = rnn_states.expand(batch_size, *rnn_states.shape[1:]).contiguous()
        if masks_.shape[0] != batch_size:
            masks_ = masks_.expand(batch_size, *masks_.shape[1:]).contiguous()
            enm_policy.to(ego_obs.device)
        # Get the action logits or probabilities from the policy
        _, action_log_probs, _ = enm_policy(ego_obs, rnn_states, masks_, deterministic=True)
        # Return the output you want to attribute (e.g., actions)
        return action_log_probs
    #D = []
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
    #ego_run_dir = "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/run3"
    enm_run_dir = "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/run3"
    experiment_name = enm_run_dir.split('/')[-4]

    env = SingleCombatEnv("1v1/ShootMissile/HierarchySelfplay")
    env.seed(0)
    args = Args()

    #ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
    enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
   # ego_policy.eval()
    enm_policy.eval()
   # ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_520.pt"))
    enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_1040.pt"))


    print("Start render")
    obs = env.reset()
    tacview = Tacview()
    if render:
        env.render(mode="real_time", tacview=tacview)
    #ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((num_agents // 2, 1))
    enm_obs =  obs[num_agents // 2:, :]
    #ego_obs =  obs[:num_agents // 2, :]
    enm_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    #D = [{'states': [], 'actions': [], 'entropy': [], 'dones': [], 'rewards': [], 'next_states': []}]
    #values = []
    while True:
        #Store current state before action
        #prev_ego_obs = ego_obs.copy()
        
        # value is action_log_probs from PPOActor.forward()
       # ego_actions, value, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
       # ego_actions = _t2n(ego_actions)
        #value = _t2n(value)
        #ego_rnn_states = _t2n(ego_rnn_states)
        #probabilities = np.exp(value)
       # entropy = -np.sum(probabilities * value, axis=-1)

        enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
        #action_dist = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=False)
        # action_log_probs, dist_entropy = ego_policy.evaluate_actions(obs, ego_rnn_states, ego_actions, masks, active_masks=None)
        # action_log_probs, dist_entropy = ego_policy.act.evaluate_actions(ego_policy.base(ego_obs), ego_actions)
        # ego_entropy = action_dist.entropy().detach().cpu().numpy()
        '''
        '''
        enm_actions = _t2n(enm_actions)
        enm_rnn_states = _t2n(enm_rnn_states)
        actions = (enm_actions)
        
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
            '''
            #D[0]["states"].append(prev_ego_obs[0].reshape(-1))
            #D[0]["actions"].append(ego_actions[0])
            #D[0]["entropy"].append(entropy[0])
            #D[0]["dones"].append(True)
            #D[0]["rewards"].append(rewards[0])
            #D[0]["next_states"].append(next_ego_obs[0].reshape(-1))
            #print("Collected data lengths:")
            
           # print("States:", len(D[0]["states"]))
           # print("Actions:", len(D[0]["actions"]))
            #print("Entropies:", len(D[0]["entropy"]))
            #print("Rewards:", len(D[0]["rewards"]))
            break
        
        D[0]["states"].append(prev_ego_obs[0].reshape(-1))
        D[0]["actions"].append(ego_actions[0].reshape(-1))
        D[0]["entropy"].append(entropy[0])
        D[0]["dones"].append(False)
        D[0]["rewards"].append(rewards[0])
        D[0]["next_states"].append(next_ego_obs[0].reshape(-1))
        values.append(value)
        
        #----------------Integrated Gradients----------------
        input = torch.tensor(ego_obs, dtype=torch.float32, device=torch.device("cpu"), requires_grad=True)
        baseline = torch.zeros_like(input)
        ig = IntegratedGradients(model_forward, False)
        attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
        
    #--------------------------------------
        #attr = attributions[0].detach().cpu().numpy()  # shape: (15,)
        
        # Update green bars
        for i, bar in enumerate(green_bars):
            bar.set_width(attr[i])
            bar.set_x(0 if attr[i] >= 0 else attr[i])
            # Optional: set color based on sign
            bar.set_color('green' if attr[i] >= 0 else 'lime')
        fig.canvas.draw_idle()
        plt.pause(0.01)
    '''
        bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
        print(f"step:{env.current_step}, bloods:{bloods} ")
        enm_obs =  obs[num_agents // 2:, ...]
        ego_obs =  obs[:num_agents // 2, ...]

   # print(episode_rewards)
   # print(bloods)

    return D, len(feature_labels), 4, feature_labels

def main():
    test_plane(None, None, None)
    
if __name__ == '__main__':
    main()