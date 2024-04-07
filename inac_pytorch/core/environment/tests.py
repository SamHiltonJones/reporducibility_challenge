import unittest
from generate_data_reward2 import GridWorldEnv
import pickle

class TestGridWorldEnv(unittest.TestCase):
    def setUp(self):
        self.env = GridWorldEnv(grid_matrix)
        self.starting_state = (1, 11)  # Update if the starting state is different
        self.goal_coords = (11, 1)  # Update if the goal coordinates are different

    def test_initialization(self):
        state = self.env.reset()
        self.assertEqual(state, self.starting_state, "Initial state should be the starting state")

    def test_step_function_different_actions(self):
        self.env.reset()
        actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
        for action in actions:
            _, _, _, _ = self.env.step(action)
            # Here you can add assertions specific to each action

    def test_boundary_conditions(self):
        self.env.reset()
        self.env.state = (0, 0)  # Corner of the grid
        next_state, _, _, _ = self.env.step(0)  # Move up
        self.assertEqual(next_state, (0, 0), "Should not move outside the grid")

    def test_termination_condition(self):
        self.env.reset()
        self.env.state = (10, 1)  # One step away from the goal
        _, _, done, _ = self.env.step(3)  # Move right towards the goal
        self.assertTrue(done, "Should be done when the goal is reached")

    def test_reward_structure(self):
        self.env.reset()
        # Move towards the goal
        next_state, reward, _, _ = self.env.step(3)  # Assuming right direction is towards the goal
        self.assertGreater(reward, 0, "Reward should be positive when moving towards the goal")

        # Move away from the goal
        self.env.state = next_state
        next_state, reward, _, _ = self.env.step(2)  # Assuming left direction is away from the goal
        self.assertLess(reward, 0, "Reward should be negative when moving away from the goal")

    def test_reset_function(self):
        self.env.reset()
        self.env.state = (5, 5)  # Arbitrary non-starting state
        state = self.env.reset()
        self.assertEqual(state, self.starting_state, "State should reset to the starting state")


grid_matrix = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,1,0,1,1,1,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,1,1,0,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1]
]

if __name__ == '__main__':
    unittest.main()
