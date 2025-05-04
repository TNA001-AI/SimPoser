
import logging
import time
from dataclasses import asdict
from pprint import pformat

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser

########################################################################################
# Control modes
########################################################################################


@safe_disconnect
def calibrate(robot: Robot, cfg: CalibrateControlConfig):
    # TODO(aliberts): move this code in robots' classes
    if robot.robot_type.startswith("stretch"):
        if not robot.is_connected:
            robot.connect()
        if not robot.is_homed():
            robot.home()
        return

    arms = robot.available_arms if cfg.arms is None else cfg.arms
    unknown_arms = [arm_id for arm_id in arms if arm_id not in robot.available_arms]
    available_arms_str = " ".join(robot.available_arms)
    unknown_arms_str = " ".join(unknown_arms)

    if arms is None or len(arms) == 0:
        raise ValueError(
            "No arm provided. Use `--arms` as argument with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    if len(unknown_arms) > 0:
        raise ValueError(
            f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`."
        )

    for arm_id in arms:
        arm_calib_path = robot.calibration_dir / f"{arm_id}.json"
        if arm_calib_path.exists():
            print(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()
        else:
            print(f"Calibration file not found '{arm_calib_path}'")

    if robot.is_connected:
        robot.disconnect()

    # Calling `connect` automatically runs calibration
    # when the calibration file is missing
    robot.connect()
    robot.disconnect()
    print("Calibration is done! You can now teleoperate and record datasets!")


@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,
        display_cameras=cfg.display_cameras,
    )


@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    # TODO(rcadene): Add option to record logs
    print("cfg", cfg.resume)
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
            local_files_only=cfg.local_files_only,
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )
    print("dataset", dataset)
    # Load pretrained policy
    print("cfg.policy", cfg.policy)
    policy = None if cfg.policy is None else make_policy(cfg.policy, cfg.device, ds_meta=None)
    print("ds_meta", dataset.meta.stats)
    breakpoint()
    # if not robot.is_connected:
    #     robot.connect()

    # listener, events = init_keyboard_listener()

    # # Execute a few seconds without recording to:
    # # 1. teleoperate the robot to move it in starting position if no policy provided,
    # # 2. give times to the robot devices to connect and start synchronizing,
    # # 3. place the cameras windows on screen
    # enable_teleoperation = policy is None
    # log_say("Warmup record", cfg.play_sounds)
    # warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_cameras, cfg.fps)

    # if has_method(robot, "teleop_safety_stop"):
    #     robot.teleop_safety_stop()

    # recorded_episodes = 0
    # while True:
    #     if recorded_episodes >= cfg.num_episodes:
    #         break

    #     log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
    #     record_episode(
    #         dataset=dataset,
    #         robot=robot,
    #         events=events,
    #         episode_time_s=cfg.episode_time_s,
    #         display_cameras=cfg.display_cameras,
    #         policy=policy,
    #         device=cfg.device,
    #         use_amp=cfg.use_amp,
    #         fps=cfg.fps,
    #     )

    #     # Execute a few seconds without recording to give time to manually reset the environment
    #     # Current code logic doesn't allow to teleoperate during this time.
    #     # TODO(rcadene): add an option to enable teleoperation during reset
    #     # Skip reset for the last episode to be recorded
    #     if not events["stop_recording"] and (
    #         (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
    #     ):
    #         log_say("Reset the environment", cfg.play_sounds)
    #         reset_environment(robot, events, cfg.reset_time_s)

    #     if events["rerecord_episode"]:
    #         log_say("Re-record episode", cfg.play_sounds)
    #         events["rerecord_episode"] = False
    #         events["exit_early"] = False
    #         dataset.clear_episode_buffer()
    #         continue

    #     dataset.save_episode(cfg.single_task)
    #     recorded_episodes += 1

    #     if events["stop_recording"]:
    #         break

    # log_say("Stop recording", cfg.play_sounds, blocking=True)
    # stop_recording(robot, listener, cfg.display_cameras)

    # if cfg.run_compute_stats:
    #     logging.info("Computing dataset statistics")

    # dataset.consolidate(cfg.run_compute_stats)

    # if cfg.push_to_hub:
    #     dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    # log_say("Exiting", cfg.play_sounds)
    # return dataset


@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
):
    # TODO(rcadene, aliberts): refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO(rcadene): Add option to record logs

    dataset = LeRobotDataset(
        cfg.repo_id, root=cfg.root, episodes=[cfg.episode], local_files_only=cfg.local_files_only
    )
    actions = dataset.hf_dataset.select_columns("action")

    if not robot.is_connected:
        robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    if isinstance(cfg.control, CalibrateControlConfig):
        calibrate(robot, cfg.control)
    elif isinstance(cfg.control, TeleoperateControlConfig):
        teleoperate(robot, cfg.control)
    elif isinstance(cfg.control, RecordControlConfig):
        record(robot, cfg.control)
    elif isinstance(cfg.control, ReplayControlConfig):
        replay(robot, cfg.control)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_robot()
