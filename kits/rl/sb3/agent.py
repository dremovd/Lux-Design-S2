"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

import os.path as osp
import sys
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO
from lux.config import EnvConfig
from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary. Make sure to exclude the .zip extension here
MODEL_WEIGHTS_RELATIVE_PATH = "./best_model"

class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        self.policy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))

        self.controller = SimpleUnitDiscreteController(self.env_cfg)

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        return dict(faction="AlphaStrike", bid=0)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        if obs["teams"][self.player]["metal"] == 0:
            return dict()
        potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
        
        water_mask = [
            (-2, -1),
            (-2, 0),
            (-2, 1),
            (2, -1),
            (2, 0),
            (2, 1),
            (-1, -2),
            (0, -2),
            (1, -2),
            (-1, 2),
            (0, 2),
            (1, 2),
        ]
        lichen_mask = water_mask
        
        options = []
        for spawn_position in potential_spawns:
            count_ice = 0
            count_rubble_free = 0
            # left_x, top_y = spawn_position
            center_x, center_y = spawn_position # left_x + 1, top_y + 1
            for dx, dy in water_mask:
                x = center_x + dx
                y = center_y + dy
                if 0 < x < obs["board"]["ice"].shape[0] and 0 < y < obs["board"]["ice"].shape[1]: 
                    count_ice += obs["board"]["ice"][x, y]
            
            for dx, dy in lichen_mask:
                x = center_x + dx
                y = center_y + dy
                if 0 < x < obs["board"]["rubble"].shape[0] and 0 < y < obs["board"]["rubble"].shape[1]: 
                    count_rubble_free += obs["board"]["rubble"][x, y] == 0
            score = (count_ice > 0, count_rubble_free)
            options.append((score, spawn_position))
            
        options = sorted(options, key=lambda x: x[0], reverse=True)
        for count_ice, pos in options[:5]:
            print(count_ice, pos, file=sys.stderr)
            
        _, pos = options[0]
        metal = obs["teams"][self.player]["metal"]
        water = obs["teams"][self.player]["water"]
        return dict(spawn=pos, metal=metal, water=water)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players
        raw_obs = dict(player_0=obs, player_1=obs)
        obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        obs = th.from_numpy(obs).float()
        with th.no_grad():

            # to improve performance, we have a rule based action mask generator for the controller used
            # which will force the agent to generate actions that are valid only.
            action_mask = (
                th.from_numpy(self.controller.action_masks(self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )
            
            # SB3 doesn't support invalid action masking. So we do it ourselves here
            features = self.policy.policy.features_extractor(obs.unsqueeze(0))
            x = self.policy.policy.mlp_extractor.shared_net(features)
            logits = self.policy.policy.action_net(x) # shape (1, N) where N=12 for the default controller

            logits[~action_mask] = -1e8 # mask out invalid actions
            dist = th.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy() # shape (1, 1)

        # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        lux_action = self.controller.action_to_lux_action(
            self.player, raw_obs, actions[0]
        )

        shared_obs = raw_obs[self.player]
        factories = shared_obs["factories"][self.player]
        for unit_id in factories.keys():
            factory = factories[unit_id]
            if 1000 - step < 50 and factory["cargo"]["water"] > 100:
                lux_action[unit_id] = 2 # water and grow lichen at the very end of the game

        return lux_action
