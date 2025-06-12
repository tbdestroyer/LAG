import sys
import torch
import gymnasium as gym
from torch.autograd import Variable
import numpy as np
import random
import ray
from CAPS import explain
from topin_baseline import gen_apg
from CAPSconfig import argparser
from data import Data
from abstract import APG
# from translation import CartpolePredicates
# from Gridworld.test_gridworld import calculate_fidelity as calculate_fidelity_grid
# from Gridworld.test_gridworld import test as test_grid
# from translation import GridworldPredicates
# from MountainCar.test_mountaincar import calculate_fidelity as calculate_fidelity_mountain
# from MountainCar.test_mountaincar import test as test_mountain
# from translation import MountainCarPredicates
from zahavy_baseline import explain_zahavy
# from LunarLander.test_lunarlander import calculate_fidelity as calculate_fidelity_lunar
# from LunarLander.test_lunarlander import test as test_lunar
# from translation import LunarLanderPredicates
# from Blackjack.test_blackjack import calculate_fidelity as calculate_fidelity_blackjack
# from Blackjack.test_blackjack import test as test_blackjack
# from translation import BlackjackPredicates

from renders.render_1v1_missle import test_plane as test_plane
from translation import PlanePredicates
from algorithms.ppo.ppo_critic import PPOCritic



if __name__ == '__main__':

    args = argparser()
    model_path = args.path
    assert model_path != ''
    fidelity_fn = None

    
    if args.env == "plane":
        pass
        data, model, num_feats, num_actions, values,feature_labels = test_plane(None, None, None)
        # if args.calc_fidelity
        #     fidelity_fn = calculate_fidelity_plane
        def load_critic():
            obs_shape = [       # obs_space
                gym.spaces.Box(low=-1, high=1, shape=(18,))
            ]
            critic = PPOCritic(args, obs_shape)
            critic_file = "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/run3/critic_latest.pt"
            critic.load_state_dict(torch.load(critic_file))
            critic.eval()
            critic.to("cpu")

            return critic
        
        values_dict = {}
        for i in range(len(values)):
            values_dict[data[0]["states"][i][0].tobytes()] = values[i]

        def value_fn(obs):
            global values_dict
            # obs = np.reshape(obs, [1, -1])
            # obs = Variable(torch.from_numpy(obs))
            # # _,_ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            # #value = model.value_function().detach().numpy()[0]
            
            # args.hidden_size = 128
            # args.act_hidden_size = 128
            # args.use_recurrent_policy = True
            # args.use_feature_normalization = True
            # args.activation_id = 1
            # args.recurrent_hidden_size = 128
            # args.recurrent_hidden_layers = 1
            
            # critic = load_critic()
            # rnn_states = torch.zeros((1, 1, args.recurrent_hidden_size))  # (num_layers, batch, hidden_size)
            # masks = torch.ones((1, 1))  # (batch, 1)
            # values, _ = critic(obs, rnn_states, masks)
            # return values.detach().cpu().numpy()[0, 0]
            if obs.tobytes() not in values_dict:
                return [[1]]
            return values_dict[obs.tobytes()]
            # Ensure obs is a torch tensor of shape (1, 21)
            obs = np.reshape(obs, [1, -1]).astype(np.float32)
            obs = torch.from_numpy(obs)
            # Prepare RNN states and masks (batch size 1)
            rnn_states = torch.zeros((1, 1, args.recurrent_hidden_size))
            masks = torch.ones((1, 1))
            # Load the critic (if not already loaded, move this outside for efficiency)
            critic = load_critic()
            values, _ = critic(obs, rnn_states, masks)
            # Return the scalar value
            return values.detach().cpu().numpy()[0, 0]
        
        print("States: ", len(data[0]["states"]), data[0]["states"][0].shape)
        print("Actions: ", len(data[0]["actions"]), data[0]["actions"][0].shape)
        print("Nex States: ", len(data[0]["next_states"]), data[0]["next_states"][0].shape)
        print("Entropy: ", len(data[0]["entropy"]), data[0]["entropy"][0].shape)
        # print("Actions: ")
        # print(data[0]["actions"])
        def print_shapes(tag, lst):
            print(f"{tag}: {len(lst)}")
            for i, x in enumerate(lst[:5]):
                print(f"  [{i}] shape: {np.array(x).shape}")
        for i in range(len(data[0]["states"])):
            data[0]["states"][i] = np.array(data[0]["states"][i]).reshape(-1)
            data[0]["next_states"][i] = np.array(data[0]["next_states"][i]).reshape(-1)
            data[0]["actions"][i] = np.array(data[0]["actions"][i]).reshape(-1)
            data[0]["entropy"][i] = np.array(data[0]["entropy"][i]).reshape(())  # scalar
            data[0]["rewards"][i] = np.array(data[0]["rewards"][i]).reshape(())  # scalar
            data[0]["dones"][i] = np.array(data[0]["dones"][i]).reshape(())      # scalar
        print_shapes("states", data[0]["states"])
        print_shapes("next_states", data[0]["next_states"])
        print_shapes("actions", data[0]["actions"])
        print_shapes("entropy", data[0]["entropy"])
        print_shapes("dones", data[0]["dones"])
        print_shapes("rewards", data[0]["rewards"])
        dataset = Data(data, value_fn)
        translator = PlanePredicates(num_feats=len(feature_labels))
        '''
    elif args.env == 'mountain':
        data, model, num_feats, num_actions = test_mountain(model_path, args.num_episodes, mode=args.alg)
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_mountain
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        dataset = Data(data, value_fn)
        translator = MountainCarPredicates(num_feats=num_feats)
        '''
    '''
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
    '''
    '''
    elif args.env == 'cart':
        data, model, num_feats, num_actions = test_cart(model_path, args.num_episodes, mode=args.alg)

        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_cart
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        
        dataset = Data(data, value_fn)
        translator = CartpolePredicates(num_fea,ts=num_feats)
    
    elif args.env == 'grid':
        data, model, num_feats, num_actions = test_grid(model_path, args.num_episodes, mode=args.alg)
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_grid
        def value_fn(obs):
            int_state = obs[0]
            obs = np.zeros(48)
            obs[int_state] = 1
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        
        dataset = Data(data, value_fn)
        translator = GridworldPredicates(num_feats=num_feats)
    '''
    '''
    elif args.env == 'lunar':
        data, model, num_feats, num_actions = test_lunar(model_path, args.num_episodes, mode=args.alg)
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_lunar
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        dataset = Data(data, value_fn)
        translator = LunarLanderPredicates(num_feats=num_feats)

    elif args.env == 'blackjack':
        data, model, num_feats, num_actions = test_blackjack(model_path, args.num_episodes, mode=args.alg)
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_blackjack
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = np.squeeze(obs)
            p = obs[0]
            d = obs[1]
            a = obs[2]
            s = np.zeros(45)
            s[p] = 1
            s[32+d] = 1
            s[43+a] = 1
            s = np.reshape(s, [1, -1])
            obs = Variable(torch.from_numpy(s))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        dataset = Data(data, value_fn)
        translator = BlackjackPredicates(num_feats=num_feats)


    else:
        raise ValueError('Enter valid environment')
'''
    if args.zahavy_baseline:
        abstract_baseline = APG(num_actions, value_fn, translator)
        explain_zahavy(args, dataset, translator, abstract_baseline, num_actions, fidelity_fn, model_path, mode=args.alg)
    elif args.topin_baseline:
        info = {'states': dataset.states, 'actions': dataset.actions, 'next_states': dataset.next_states, 'dones': dataset.dones, 'entropies': dataset.entropies}
        abstract_baseline = APG(num_actions, value_fn, translator, info=info)
        gen_apg(abstract_baseline, model_path, fidelity_fn, mode=args.alg)
    else:
        abstract_baseline = APG(num_actions, value_fn, translator)
        explain(args, dataset, model_path, translator, num_feats, num_actions, fidelity_fn, abstract_baseline, mode=args.alg)
    







