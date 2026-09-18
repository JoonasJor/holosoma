from types import SimpleNamespace

import pytest

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.curriculum import CurriculumTermCfg
from holosoma.managers.command.terms.locomotion import LocomotionCommand
from holosoma.managers.curriculum.terms.locomotion import LowSpeedCommandCurriculum
from holosoma.managers.reward.terms.locomotion import tracking_ang_vel, tracking_lin_vel
from holosoma.utils.safe_torch_import import torch


class FakeGait:
    def __init__(self):
        self.resampled_env_ids = []

    def resample_frequency(self, env_ids):
        self.resampled_env_ids.append(env_ids.clone())


class FakeCommandManager:
    def __init__(self, gait=None, command_term=None):
        self.gait = gait
        self.command_term = command_term

    def get_state(self, name):
        if name == "locomotion_gait":
            return self.gait
        if name == "locomotion_command":
            return self.command_term
        return None


class FakeCommandEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.device = "cpu"
        self.is_evaluating = False


def make_command_term(*, stand_prob=0.0, low_speed_prob=0.0, low_speed_ranges=None, num_envs=4096):
    params = {
        "command_ranges": {
            "lin_vel_x": [-1.0, 1.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-1.0, 1.0],
        },
        "stand_prob": stand_prob,
        "low_speed_prob": low_speed_prob,
    }
    if low_speed_ranges is not None:
        params["low_speed_command_ranges"] = low_speed_ranges

    term = LocomotionCommand(CommandTermCfg(func="unused", params=params), FakeCommandEnv(num_envs))
    term.manager = FakeCommandManager(gait=FakeGait())
    term.setup()
    return term


def test_low_speed_sampler_generates_signed_nonzero_commands_in_range():
    torch.manual_seed(4)
    term = make_command_term(
        stand_prob=0.25,
        low_speed_prob=1.0,
        low_speed_ranges={
            "lin_vel_x": [0.05, 0.25],
            "lin_vel_y": [0.05, 0.25],
            "ang_vel_yaw": [0.05, 0.25],
        },
    )

    term._resample(torch.arange(term.env.num_envs))
    commands = term.commands
    assert commands is not None

    standing = torch.all(commands == 0.0, dim=1)
    moving = commands[~standing]
    magnitudes = torch.abs(moving)
    assert standing.any()
    assert moving.shape[0] > 0
    assert torch.all(torch.any(magnitudes > 0.0, dim=1))
    assert torch.all((magnitudes == 0.0) | ((magnitudes >= 0.05) & (magnitudes <= 0.25)))
    assert torch.any(moving > 0.0)
    assert torch.any(moving < 0.0)


def test_legacy_command_sampler_keeps_low_speed_sampling_disabled():
    term = make_command_term()

    assert term.low_speed_prob == 0.0
    assert term.low_speed_command_ranges is None


def test_low_speed_sampler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="low_speed_prob"):
        make_command_term(low_speed_prob=1.1)

    with pytest.raises(ValueError, match="low_speed_command_ranges"):
        make_command_term(low_speed_prob=0.1)

    with pytest.raises(ValueError, match="must satisfy"):
        make_command_term(
            low_speed_prob=0.1,
            low_speed_ranges={
                "lin_vel_x": [0.25, 0.05],
                "lin_vel_y": [0.05, 0.25],
                "ang_vel_yaw": [0.05, 0.25],
            },
        )


def make_reward_env(commands, linear_velocities=None, angular_velocities=None):
    num_envs = commands.shape[0]
    root_states = torch.zeros(num_envs, 13)
    if linear_velocities is not None:
        root_states[:, 7:10] = linear_velocities
    if angular_velocities is not None:
        root_states[:, 10:13] = angular_velocities
    return SimpleNamespace(
        base_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1),
        command_manager=SimpleNamespace(commands=commands),
        simulator=SimpleNamespace(robot_root_states=root_states),
    )


def test_low_speed_tracking_is_stricter_and_zero_commands_are_unchanged():
    commands = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.1],
            [0.3, 0.0, 0.3],
        ]
    )
    env = make_reward_env(commands)

    linear_rewards = tracking_lin_vel(
        env,
        tracking_sigma=0.25,
        low_speed_tracking_sigma=0.01,
        low_speed_min_command=0.01,
        low_speed_max_command=0.25,
    )
    angular_rewards = tracking_ang_vel(
        env,
        tracking_sigma=0.25,
        low_speed_tracking_sigma=0.01,
        low_speed_min_command=0.01,
        low_speed_max_command=0.25,
    )

    torch.testing.assert_close(linear_rewards[0], torch.tensor(1.0))
    torch.testing.assert_close(linear_rewards[1], torch.exp(torch.tensor(-1.0)))
    torch.testing.assert_close(linear_rewards[2], torch.exp(torch.tensor(-0.09 / 0.25)))
    torch.testing.assert_close(angular_rewards[0], torch.tensor(1.0))
    torch.testing.assert_close(angular_rewards[1], torch.exp(torch.tensor(-1.0)))
    torch.testing.assert_close(angular_rewards[2], torch.exp(torch.tensor(-0.09 / 0.25)))


class FakeLowSpeedCommandTerm:
    def __init__(self):
        self.low_speed_prob = None
        self.stand_prob = 0.2

    def set_low_speed_probability(self, probability):
        self.low_speed_prob = probability


class FakeCurriculumEnv:
    def __init__(self):
        self.average_episode_length = 0.0
        self.low_speed_command_term = FakeLowSpeedCommandTerm()
        self.command_manager = FakeCommandManager(command_term=self.low_speed_command_term)
        self.log_dict = {}


def test_low_speed_command_curriculum_respects_bounds_and_restores_state():
    env = FakeCurriculumEnv()
    curriculum = LowSpeedCommandCurriculum(
        CurriculumTermCfg(
            func="unused",
            params={
                "initial_probability": 0.1,
                "min_probability": 0.1,
                "max_probability": 0.5,
                "level_down_threshold": 150.0,
                "level_up_threshold": 750.0,
                "degree": 0.1,
            },
        ),
        env,
    )

    curriculum.setup()
    assert env.low_speed_command_term.low_speed_prob == 0.1

    curriculum.reset(None)
    assert env.low_speed_command_term.low_speed_prob == 0.1

    env.average_episode_length = 800.0
    curriculum.reset(None)
    assert env.low_speed_command_term.low_speed_prob == pytest.approx(0.11)

    curriculum.load_state_dict({"current_probability": 2.0})
    assert env.low_speed_command_term.low_speed_prob == 0.5
    assert env.low_speed_command_term.stand_prob == 0.2
