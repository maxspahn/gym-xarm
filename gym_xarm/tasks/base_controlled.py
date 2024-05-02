import os

import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import _ALL_RENDERERS

from gym_xarm.tasks.base import Base
from gym_xarm.tasks import mocap

RENDER_MODES = ["rgb_array"]
if os.environ.get("MUJOCO_GL") == "glfw":
    RENDER_MODES.append("human")
elif os.environ.get("MUJOCO_GL") not in _ALL_RENDERERS:
    os.environ["MUJOCO_GL"] = "egl"


class BaseControlled(Base):
    """
    Superclass for all gym-xarm environments that uses actual joint controllers.
    """
    def __init__(
        self,
        task,
        obs_type="state",
        render_mode="rgb_array",
        gripper_rotation=None,
        observation_width=84,
        observation_height=84,
        visualization_width=680,
        visualization_height=680,
    ):
        # Coordinates
        if gripper_rotation is None:
            gripper_rotation = [0, 1, 0, 0]
        self.gripper_rotation = np.array(gripper_rotation, dtype=np.float32)
        self.center_of_table = np.array([1.655, 0.3, 0.63625])
        self.max_z = 1.2
        self.min_z = 0.2

        # Observations
        self.obs_type = obs_type

        # Rendering
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        # Assets
        self.xml_path = os.path.join(os.path.dirname(__file__), "assets", f"{task}_controlled.xml")
        if not os.path.exists(self.xml_path):
            raise OSError(f"File {self.xml_path} does not exist")

        # Initialize sim, spaces & renderers
        self._initialize_simulation()
        self.observation_renderer = self._initialize_renderer(renderer_type="observation")
        self.visualization_renderer = self._initialize_renderer(renderer_type="visualization")
        self.observation_space = self._initialize_observation_space()
        self._dof = 8
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._dof,))
        self.action_padding = np.zeros(self._dof - self._dof, dtype=np.float32)

        assert (
            int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
        ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'


    def step(self, action):
        assert action.shape == (self._dof,)
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        joint_action = np.concatenate((np.zeros(7), action[-1:]))
        self._apply_action(joint_action)
        self._mujoco.mj_step(self.model, self.data, nstep=2)
        self._step_callback()
        observation = self.get_obs()
        reward = self.get_reward()
        terminated = is_success = self.is_success()
        truncated = False
        info = {"is_success": is_success}

        return observation, reward, terminated, truncated, info


    def _apply_action(self, action):
        assert action.shape == (8,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:7], action[7]
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        mocap.apply_controller_action(
            self.model,
            self._model_names,
            self.data,
            np.concatenate([pos_ctrl, gripper_ctrl]),
        )
        print('---')
        print(np.round(self.data.qvel[0:7], 2))
        print(np.round(self.data.qpos[0:7], 2))
