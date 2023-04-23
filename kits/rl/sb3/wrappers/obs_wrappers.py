from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces

position_mask = [
    (dx, dy)
    for dx in range(-2, 3)
    for dy in range(-2, 3)
]

class SimpleUnitObservationWrapper(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(14 + 2 * len(position_mask),))

    def observation(self, obs):
        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    
    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any, player: str='player_0') -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs[player]
        ice_map = shared_obs["board"]["ice"]
        rubble_map = shared_obs["board"]["rubble"]
        ice_tile_locations = np.argwhere(ice_map == 1)
        def access(item_map, pos, dx, dy):
            dpos = pos.copy()
            dpos[0] += dx
            dpos[1] += dy
            if 0 <= dpos[0] < item_map.shape[0] and 0 <= dpos[1] < item_map.shape[1]:
                return item_map[dpos[0]][dpos[1]]
            else: 
                return -100
            

        for agent in obs.keys():
            obs_vec = np.zeros(
                14 
                + len(position_mask) * 2,
            )

            factories = shared_obs["factories"][agent]
            factory_vec = np.zeros(2)
            for k in factories.keys():
                # here we track a normalized position of the first friendly factory
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / env_cfg.map_size
                break
            units = shared_obs["units"][agent]
            for k in units.keys():
                unit = units[k]

                # store cargo+power values scaled to [0, 1]
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                cargo = unit["cargo"]
                cargo_vec = np.array(
                    [
                        unit["power"] / battery_cap,
                        cargo["ice"] / cargo_space,
                        cargo["ore"] / cargo_space,
                        cargo["water"] / cargo_space,
                        cargo["metal"] / cargo_space,
                        sum(cargo.get(resource, 0) for resource in ("ice", "ore", "water", "metal")) / cargo_space,
                    ]
                )
                unit_type = (
                    0 if unit["unit_type"] == "LIGHT" else 1
                )  # note that build actions use 0 to encode Light
                # normalize the unit position
                pos = np.array(unit["pos"]) / env_cfg.map_size
                unit_vec = np.concatenate([
                    pos, 
                    [unit_type], 
                    cargo_vec, 
                    [unit["team_id"]],
                ], axis=-1)

                # we add some engineered features down here
                # compute closest ice tile
                ice_tile_distances = np.mean(
                    (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
                )
                # normalize the ice tile location
                closest_ice_tile = (
                    ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size
                )
                obs_ice = np.array([
                    access(ice_map, unit["pos"], dx, dy)
                    for dx, dy in position_mask
                ])
                obs_rubble = np.array([
                    access(rubble_map, unit["pos"], dx, dy)
                    for dx, dy in position_mask                    
                ])
                assert len(obs_ice) == 25
                assert len(obs_rubble) == 25
                
                obs_vec = np.concatenate(
                    [
                        unit_vec,
                        factory_vec - pos,
                        closest_ice_tile - pos,
                        obs_ice,
                        obs_rubble,
                    ], 
                    axis=-1
                )
                break
            observation[agent] = obs_vec

        return observation
