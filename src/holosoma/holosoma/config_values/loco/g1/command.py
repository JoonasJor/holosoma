"""Locomotion command presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg

g1_29dof_command = CommandManagerCfg(
    params={
        "locomotion_command_resampling_time": 10.0,
    },
    setup_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
            params={
                "gait_period": 1.0,
                "gait_period_randomization_width": 0.2,
            },
        ),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            params={
                "command_ranges": {
                    "lin_vel_x": [-1.0, 1.0],
                    "lin_vel_y": [-1.0, 1.0],
                    "ang_vel_yaw": [-1.0, 1.0],
                    "heading": [-3.14, 3.14],
                },
                "stand_prob": 0.2,
            },
        ),
    },
    reset_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionCommand"),
    },
    step_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionCommand"),
    },
)

g1_29dof_low_speed_command = replace(
    g1_29dof_command,
    setup_terms={
        **g1_29dof_command.setup_terms,
        "locomotion_command": replace(
            g1_29dof_command.setup_terms["locomotion_command"],
            params={
                **g1_29dof_command.setup_terms["locomotion_command"].params,
                "low_speed_prob": 0.1,
                "low_speed_command_ranges": {
                    "lin_vel_x": [0.05, 0.25],
                    "lin_vel_y": [0.05, 0.25],
                    "ang_vel_yaw": [0.05, 0.25],
                },
            },
        ),
    },
)

__all__ = ["g1_29dof_command", "g1_29dof_low_speed_command"]
