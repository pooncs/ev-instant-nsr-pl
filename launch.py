import sys
import argparse
import os
import time
import logging
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--resume', default=None, help='path to the weights to be resumed')
    parser.add_argument(
        '--resume_weights_only',
        action='store_true',
        help='specify this argument to restore only the weights (w/o training states), e.g. --resume path/to/resume --resume_weights_only'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--validate', action='store_true')
    group.add_argument('--test', action='store_true')
    group.add_argument('--predict', action='store_true')

    parser.add_argument('--exp_dir', default='exp')
    parser.add_argument('--runs_dir', default='runs')
    parser.add_argument('--verbose', action='store_true', help='if true, set logging level to DEBUG')

    args, extras = parser.parse_known_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    import systems
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    from utils.callbacks import CodeSnapshotCallback, ConfigSnapshotCallback, CustomProgressBar
    from utils.misc import load_config    

    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    print(config.tag)
    root_data = '/home/ubuntu/src/data'
    config.trial_name = config.get('trial_name') or (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    config.exp_dir = os.path.join(root_data, config.dataset.scene, args.exp_dir)
    config.save_dir = os.path.join(config.exp_dir, config.trial_name, 'save')
    config.ckpt_dir = os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = os.path.join(config.exp_dir, config.trial_name, 'code')
    config.config_dir = os.path.join(config.exp_dir, config.trial_name, 'config')

    logger = logging.getLogger('pytorch_lightning')
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if 'seed' not in config:
        config.seed = int(time.time() * 1000) % 1000
    pl.seed_everything(config.seed)

    # dm = datasets.make(config.dataset.name, config.dataset)
    system = systems.make(config.system.name, config, load_from_checkpoint=None if not args.resume_weights_only else args.resume)

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=config.ckpt_dir,
                **config.checkpoint
            ),
            LearningRateMonitor(logging_interval='step'),
            CodeSnapshotCallback(
                config.code_dir, use_version=False
            ),
            ConfigSnapshotCallback(
                config, config.config_dir, use_version=False
            ),
            CustomProgressBar(refresh_rate=1),
        ]

    loggers = []
    if args.train:
        args.runs_dir = os.path.join(root_data, args.runs_dir)
        loggers += [
            TensorBoardLogger(args.runs_dir, name=config.name, version=config.trial_name),
            CSVLogger(config.exp_dir, name=config.trial_name, version='csv_logs')
        ]
    
    if sys.platform == 'win32':
        # does not support multi-gpu on windows
        strategy = 'dp'
        assert n_gpus == 1
    else:
        strategy = 'ddp_find_unused_parameters_false'
    
    trainer = Trainer(
        devices=n_gpus,
        accelerator='gpu',
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        auto_scale_batch_size=True,
        **config.trainer
    )

    if args.train:
        if args.resume and not args.resume_weights_only:
            trainer.fit(system, datamodule=system.dm, ckpt_path=args.resume)
        else:
            trainer.fit(system, datamodule=system.dm)
        trainer.test(system, datamodule=system.dm)
    elif args.validate:
        trainer.validate(system, datamodule=system.dm, ckpt_path=args.resume)
    elif args.test:
        trainer.test(system, datamodule=system.dm, ckpt_path=args.resume)
    elif args.predict:
        trainer.predict(system, datamodule=system.dm, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
