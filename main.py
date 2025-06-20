from pathlib import Path
import argparse
import yaml
import torch
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import logging

from train import BasicTrain
from model.DHGFormer import DHGFormer
from dataloader import init_dataloader
from util import Logger_main


def main(args, current_seed):
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        random.seed(current_seed)
        np.random.seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        torch.manual_seed(current_seed)
        torch.cuda.manual_seed(current_seed)
        torch.cuda.manual_seed_all(current_seed)

        dataloaders, node_size, node_feature_size, timeseries_size = \
            init_dataloader(config['data'])
        config['train']["seq_len"] = timeseries_size
        config['train']["node_size"] = node_size

        model = DHGFormer(config['model'], node_size,
                         node_feature_size, timeseries_size)
        use_train = BasicTrain

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay'])
        opts = (optimizer,)

        loss_name = 'loss'
        if config['train']["group_loss"]:
            loss_name = f"{loss_name}_group_loss"
        if config['train']["sparsity_loss"]:
            loss_name = f"{loss_name}_sparsity_loss"

        save_folder_name = Path(config['train']['log_folder']) / Path(config['model']['type']) / Path(
            f"{config['data']['dataset']}_{config['data']['atlas']}")

        train_process = use_train(
            config['train'], model, opts, dataloaders, save_folder_name)

        train_process.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='setting/abide_DHGFormer.yaml', type=str,
                        help='Configuration filename for training the model.')
    parser.add_argument('--repeat_time', default=5, type=int)
    parser.add_argument('--seed', default=21, type=int)
    parser.add_argument('--device', default=4, type=int)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    logger = Logger_main()

    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        logger.info(f"Model {config['model']['type']} on {config['data']['dataset']} Dataset")
    for i in range(args.repeat_time):
        current_seed = seed + i
        logger.info(f"Fold {i + 1}/{args.repeat_time}, SEED:{current_seed}, device:{args.device}")
        main(args, current_seed)
        logger.info(f"Fold {i + 1} is done!")
    logging.info(f"Done!")