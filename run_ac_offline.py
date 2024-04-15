import os
import argparse
import pickle

import core.environment.env_factory as environment
from core.utils import torch_utils, logger, run_funcs
from core.agent.in_sample import *

def load_data(is_training):
    base_path = "/home/sam/jack_and_sam/reproducibility_challenge/inac_pytorch/core/"
    dataset_file = 'train_data.pkl' if is_training else 'test_data.pkl'
    file_path = os.path.join(base_path, dataset_file)
    
    with open(file_path, 'rb') as file:
        loaded_experiences, loaded_data_dict = pickle.load(file)

    if 'pkl' in loaded_data_dict:
        data_dict = loaded_data_dict['pkl']
    else:
        data_dict = loaded_data_dict  

    for key, value in data_dict.items():
        if hasattr(value, 'shape'):
            print(f"Loaded data - {key} shape: {value.shape}")
        else:
            print(f"Loaded data - {key} is not an array")

    return loaded_experiences, data_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--env_name', default='grid', type=str)
    parser.add_argument('--dataset', default='expert', type=str)
    parser.add_argument('--discrete_control', default=1, type=int)
    parser.add_argument('--state_dim', default=2, type=int)
    parser.add_argument('--action_dim', default=4, type=int)
    parser.add_argument('--tau', default=0.01, type=float)
    
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--timeout', default=100, type=int)
    parser.add_argument('--gamma', default=0.90, type=float)
    parser.add_argument('--use_target_network', default=1, type=int)
    parser.add_argument('--target_network_update_freq', default=1, type=int)
    parser.add_argument('--polyak', default=0.995, type=float)
    parser.add_argument('--evaluation_criteria', default='return', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--info', default='0', type=str)
    cfg = parser.parse_args()

    torch_utils.set_one_thread()
    torch_utils.random_seed(cfg.seed)

    project_root = os.path.abspath(os.path.dirname(__file__))
    exp_path = f"data/output/{cfg.env_name}/{cfg.dataset}/{cfg.info}/{cfg.seed}_run"
    cfg.exp_path = os.path.join(project_root, exp_path)
    torch_utils.ensure_dir(cfg.exp_path)

    train_experience, train_data_dict = load_data(is_training=True)
    test_experience, test_data_dict = load_data(is_training=False)

    cfg.offline_data = train_data_dict.get('pkl', None)
    cfg.env_fn = environment.EnvFactory.create_env_fn(cfg)

    cfg.logger = logger.Logger(cfg, cfg.exp_path)
    try:
        logger.log_config(cfg)
    except KeyError as e:
        print(f"KeyError encountered in logger configuration: {e}")

    print("Initializing agent...")
    agent_obj = InSampleAC(
        device=cfg.device,
        discrete_control=cfg.discrete_control,
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        hidden_units=cfg.hidden_units,
        learning_rate=cfg.learning_rate,
        tau=cfg.tau,
        polyak=cfg.polyak,
        exp_path=cfg.exp_path,
        seed=cfg.seed,
        env_fn=cfg.env_fn,
        timeout=cfg.timeout,
        gamma=cfg.gamma,
        offline_data=train_data_dict['pkl'],
        batch_size=cfg.batch_size,
        use_target_network=cfg.use_target_network,
        target_network_update_freq=cfg.target_network_update_freq,
        evaluation_criteria=cfg.evaluation_criteria,
        logger=cfg.logger
    )

    print("Agent initialized.")
    run_funcs.run_steps(agent_obj, cfg.max_steps, cfg.log_interval, cfg.exp_path, train_experience, test_data_dict['pkl'])