from src.utils import timed_main, set_seed
from src.parser import main_parser
from src.data_pre_process import Trajectory_Data_Pre_Process
from src.trainer import trainer
from src.models.goal_pretrain import goal_pretrainer
from src.trajectory_bank_builder import TrajectoryBankCacheBuilder
from src.joint_dependency_v2_cache import JointDependencyV2CacheBuilder
import os
# os.environ["WANDB_MODE"]="offline"

@timed_main(use_git=True)
def main():
    # Parse input parameters and save/load config file
    args = main_parser()

    # Set seed for reproducibility
    if args.reproducibility:
        set_seed(seed_value=args.seed, use_cuda=args.use_cuda)
        print('set random seed')

    # Frozen trajectory-bank construction is a read-only consumer of the
    # synchronized batch cache.  In particular, debug limits for cache smoke
    # tests must never invalidate or rebuild the expensive source batches.
    if args.phase != 'trajectory_cache':
        Trajectory_Data_Pre_Process(args)

    if args.phase == 'pre-process':
        processor = None
    elif args.phase == 'trajectory_cache':
        processor = TrajectoryBankCacheBuilder(args)
    elif args.phase == 'build-jdv2-cache':
        processor = JointDependencyV2CacheBuilder(args)
    elif args.phase == 'goal_pretrain':
        processor = goal_pretrainer(args)
    else:
        # Initialize data-loader and model
        processor = trainer(args)

    if args.phase == 'pre-process':
        print("Data pre-processing and batches creation finished.")
    elif args.phase == 'trajectory_cache':
        processor.build()
    elif args.phase == 'build-jdv2-cache':
        processor.build()
    elif args.phase == 'goal_pretrain':
        processor.train_test()
    elif args.phase == 'train':
        processor.train()
    elif args.phase == 'test':
        processor.test(load_checkpoint=args.load_checkpoint)
    elif args.phase == 'train_test':
        processor.train_test()
    else:
        raise ValueError(
            f"Unsupported phase {args.phase}! args.phase can only take the "
            f"following values: 'train', 'test', 'train_test', "
            "'trajectory_cache', 'build-jdv2-cache' or 'pre-process'")


if __name__ == '__main__':
    main()
