import numpy as np
from .reward_function_base import BaseRewardFunction

class TerrainReward(BaseRewardFunction):
    """
    TerrainReward
    Rewards for terrain-aware positioning:
    - Height advantage over opponent
    - Terrain masking
    - Terrain-based tactical positioning
    """
    def __init__(self, config):
        super().__init__(config)
        self.advantage_height = getattr(self.config, f'{self.__class__.__name__}_advantage_height', 1000.0)
        self.mask_distance = getattr(self.config, f'{self.__class__.__name__}_mask_distance', 5000.0)
        
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_height', '_mask', '_position']]

    def get_reward(self, task, env, agent_id):
        """
        Calculate terrain-based rewards

        Args:
            task: task instance
            env: environment instance
            agent_id: ID of the agent

        Returns:
            (float): reward
        """
        # Get agent and opponent positions
        agent_pos = env.agents[agent_id].get_position()
        agent_alt = agent_pos[-1]
        
        # Find opponent
        opponent_id = None
        for other_id in env.agents:
            if other_id != agent_id:
                opponent_id = other_id
                break
        
        if opponent_id is None:
            return 0.0
            
        opponent_pos = env.agents[opponent_id].get_position()
        opponent_alt = opponent_pos[-1]
        
        # Height advantage reward
        height_diff = agent_alt - opponent_alt
        height_reward = np.clip(height_diff / self.advantage_height, -1.0, 1.0)
        
        # Terrain masking reward
        terrain_alt = env.get_terrain_elevation(agent_pos[0], agent_pos[1])
        mask_reward = 0.0
        if terrain_alt > opponent_alt:
            # Calculate if terrain blocks line of sight
            distance = np.linalg.norm(agent_pos[:2] - opponent_pos[:2])
            if distance < self.mask_distance:
                mask_reward = 1.0
        
        # Tactical positioning reward
        position_reward = 0.0
        if height_diff > 0:  # If we have height advantage
            # Reward for maintaining distance
            distance = np.linalg.norm(agent_pos[:2] - opponent_pos[:2])
            position_reward = np.exp(-distance / self.mask_distance)
        
        # Combine rewards
        total_reward = (height_reward + mask_reward + position_reward) * self.reward_scale
        
        self.reward_trajectory[agent_id].append([total_reward, height_reward, mask_reward, position_reward])
        
        return total_reward 