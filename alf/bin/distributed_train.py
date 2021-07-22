from absl import app
from absl import flags
from absl import logging
import os
import pathlib

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback

from alf.utils import common
import alf.utils.external_configurables
from alf.trainers import policy_trainer


def _define_flags():
    flags.DEFINE_string(
        'root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
        'Root directory for writing logs/summaries/checkpoints.')
    flags.DEFINE_string('gin_file', None, 'Path to the gin-config file.')
    flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
    flags.DEFINE_string('conf', None, 'Path to the alf config file.')
    flags.DEFINE_multi_string('conf_param', None, 'Config binding parameters.')
    flags.DEFINE_bool('store_snapshot', True,
                      'Whether store an ALF snapshot before training')


FLAGS = flags.FLAGS


def single_worker(rank: int, conf_file: str, root_dir: str, store_snapshot: bool):
    try:
        dist.init_process_group('nccl', rank=rank, world_size=2)
        alf.PerProcessContext().set_rank(rank)
        if rank == 0:
            FLAGS.alsologtostderr = True
            logging.set_verbosity(logging.INFO)
            logging.get_absl_handler().use_absl_log_file(log_dir=root_dir)

            if store_snapshot:
                # ../<ALF_REPO>/alf/bin/train.py
                file_path = os.path.abspath(__file__)
                alf_root = str(pathlib.Path(file_path).parent.parent.parent.absolute())
                # generate a snapshot of ALF repo as ``<root_dir>/alf``
                common.generate_alf_root_snapshot(alf_root, root_dir)
        common.parse_conf_file(conf_file)
        trainer_conf = policy_trainer.TrainerConfig(root_dir=root_dir)

        if torch.cuda.is_available():
            alf.set_default_device("cuda")
            torch.cuda.set_device(rank)

        print(f'Initialize trainer on device {alf.PerProcessContext().rank}')
        trainer = policy_trainer.RLTrainer(trainer_conf)
        trainer.train()
    except Exception as e:
        tb = traceback.format_exc()
        print(e)
        print(tb)
    finally:
        alf.close_env()


def main(_):
    FLAGS.alsologtostderr = True
    root_dir = common.abs_path(FLAGS.root_dir)
    os.makedirs(root_dir, exist_ok=True)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    conf_file = common.get_conf_file()

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    processes = []
    for i in range(2):
        processes.append(
            mp.Process(target=single_worker, args=(i, conf_file, root_dir, FLAGS.store_snapshot)))
        processes[i].start()

    for proc in processes:
        print('wait for join ..')
        proc.join()



if __name__ == '__main__':
    _define_flags()
    flags.mark_flag_as_required('root_dir')
    # if torch.cuda.is_available():
    #     alf.set_default_device("cuda")
    app.run(main)
