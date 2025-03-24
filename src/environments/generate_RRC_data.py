from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from tqdm import tqdm
import remote_control_gym as gym
import math


def generate_actions(seq_length: int, biased: bool = False) -> np.ndarray:
    """generate 2d actions for Remote control gym env

    Args:
        seq_length (int): sequence length

    Returns:
        np.ndarray: (seq_length, 2)
    """
    if biased:
        angles = np.random.rand(seq_length) * 2 * math.pi
        random_x = np.sin(angles)
        random_y = np.cos(angles)
        random_xy = np.stack([random_x, random_y])
        factor = np.linspace(0.0001, 1, seq_length)
        actions = random_xy * factor
        actions = actions.T
    else:
        actions = np.random.rand(seq_length, 2) * 2 - 1.0
    return actions


def generate_dataset(
    n_samples: int,
    seq_length: int,
    start_seed: int = 0,
    controlled_robot: bool = False,
    biased: bool = False,
) -> np.ndarray:
    # No control of robot training data
    rrc_states = []
    progress_bar = tqdm(desc="Sample trajectories: trial=0", total=n_samples)
    iter_counter = 0
    collected_samples = len(rrc_states)
    while collected_samples < n_samples:
        iter_counter += 1
        progress_bar.set_description(f"Sample trajectories: trial={iter_counter}")

        seed = iter_counter + start_seed
        np.random.seed(seed)
        env = gym.RemoteControlGym(seed)
        robot_controlled = False

        o_t, info = env.reset(seed=seed, options={"with_info": True})
        actions = generate_actions(seq_length=seq_length, biased=biased)
        states = np.zeros((seq_length, 15))
        for t in range(seq_length):
            info_array = list(
                map(
                    lambda ele: np.array([ele])
                    if not isinstance(ele, np.ndarray)
                    else ele,
                    info.values(),
                )
            )
            state = np.concat([o_t, actions[t], np.concat(info_array)])
            states[t] = state
            o_t, _, _, info = env.step(actions[t])

            # check if robot was controlled by the actions
            if info["robot_controlled"]:
                robot_controlled = True
                # compare to target
                if not controlled_robot:
                    # print("break")
                    break
        env.close()
        # print(int(controlled_robot) + int(robot_controlled), controlled_robot, robot_controlled)
        if (int(controlled_robot) + int(robot_controlled)) == 1:
            continue
        if (int(controlled_robot) + int(robot_controlled)) == 0 or (
            int(controlled_robot) + int(robot_controlled)
        ) == 2:
            # print("add sample")
            rrc_states.append(states)
            collected_samples = len(rrc_states)
            progress_bar.update(1)
        # print(collected_samples < n_samples, collected_samples, n_samples)

    progress_bar.close()

    rrc_states = np.stack(rrc_states)
    return rrc_states


def setup_parser():
    parser = ArgumentParser(
        "Sample controlled and uncontrolled train and test datasets on the remote controlled gym environment"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/RRC2/",
        help="where to store the datasets",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8000,
        help="How many samples per dataset.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=50,
        help="Length of each sampled trajectory.",
    )
    parser.add_argument(
        "--biased",
        action="store_true",
        help="Sample biased actions.",
    )

    return parser

def get_file_name(controlled_robot: bool, biased: bool, train: bool) -> str:
    base_filename = "ControlData"
    if not controlled_robot:
        filename = "No" + base_filename
    else:
        filename = base_filename

    if biased:
        filename = "Biased" + filename
    else:
        filename = "Generalized"  + filename
    
    if train:
        filename += "Train"
    else:
        filename += "Test"

    filename += ".npy"
    return filename


def main(target_dir: str, n_samples: int, seq_length: int, biased: bool):
    target_dir: Path = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    controlled_robot = False
    
    file_name = get_file_name(controlled_robot, biased, True)
    print("Sample data for: ", file_name)
    rrc_states = generate_dataset(
        n_samples=n_samples,
        seq_length=seq_length,
        start_seed= 1900000,
        controlled_robot=controlled_robot,
        biased=biased,
    )
    np.save(target_dir / file_name, rrc_states)
    
    file_name = get_file_name(controlled_robot, biased, False)
    print("Sample data for: ", file_name)
    rrc_states = generate_dataset(
        n_samples=n_samples,
        seq_length=seq_length,
        start_seed= 2900000,
        controlled_robot=controlled_robot,
        biased=biased,
    )
    np.save(target_dir / file_name, rrc_states)

    
    controlled_robot = True
    file_name = get_file_name(controlled_robot, biased, True)
    print("Sample data for: ", file_name)
    rrc_states = generate_dataset(
        n_samples=n_samples,
        seq_length=seq_length,
        start_seed= 3900000,
        controlled_robot=controlled_robot,
        biased=biased,
    )
    np.save(target_dir / file_name, rrc_states)
    
    file_name = get_file_name(controlled_robot, biased, False)
    print("Sample data for: ", file_name)
    rrc_states = generate_dataset(
        n_samples=n_samples,
        seq_length=seq_length,
        start_seed= 4900000,
        controlled_robot=controlled_robot,
        biased=biased,
    )
    np.save(target_dir / file_name, rrc_states)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(**vars(args))
