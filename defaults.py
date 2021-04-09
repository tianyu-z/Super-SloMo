import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_root",
    type=str,
    required=True,
    help="path to dataset folder containing train-test-validation folders",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help="path to folder for saving checkpoints",
)
parser.add_argument(
    "--checkpoint", type=str, help="path of checkpoint for pretrained model"
)
parser.add_argument(
    "--train_continue", action="store_true", help="resuming from checkpoint."
)
parser.add_argument(
    "-it",
    "--init_type",
    default="",
    type=str,
    help="the name of an initialization method: normal | xavier | kaiming | orthogonal",
)

parser.add_argument(
    "--epochs", type=int, default=200, help="number of epochs to train. Default: 200."
)
parser.add_argument(
    "-tbs",
    "--train_batch_size",
    type=int,
    default=384,
    help="batch size for training. Default: 6.",
)
parser.add_argument(
    "-nw", "--num_workers", default=4, type=int, help="number of CPU you get"
)
parser.add_argument(
    "-vbs",
    "--validation_batch_size",
    type=int,
    default=384,
    help="batch size for validation. Default: 10.",
)
parser.add_argument(
    "-ilr",
    "--init_learning_rate",
    type=float,
    default=0.0001,
    help="set initial learning rate. Default: 0.0001.",
)
parser.add_argument(
    "--milestones",
    type=list,
    default=[100, 150],
    help="Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]",
)
parser.add_argument(
    "--progress_iter",
    type=int,
    default=100,
    help="frequency of reporting progress and validation. N: after every N iterations. Default: 100.",
)
parser.add_argument(
    "--logimagefreq", type=int, default=1, help="frequency of logging image.",
)
parser.add_argument(
    "--checkpoint_epoch",
    type=int,
    default=5,
    help="checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.",
)
parser.add_argument(
    "-wp", "--workspace", default="tianyu-z", type=str, help="comet-ml workspace"
)
parser.add_argument(
    "-dh", "--data_h", default=128, type=int, help="H of the data shape"
)
parser.add_argument(
    "-dw", "--data_w", default=128, type=int, help="W of the data shape"
)
parser.add_argument(
    "-pn",
    "--projectname",
    default="super-slomo",
    type=str,
    help="comet-ml project name",
)
parser.add_argument(
    "--nocomet", action="store_true", help="not using comet_ml logging."
)
parser.add_argument(
    "--cometid", type=str, default="", help="the comet id to resume exps",
)
parser.add_argument(
    "-rs",
    "--randomseed",
    type=int,
    default=2021,
    help="batch size for validation. Default: 10.",
)
args = parser.parse_args()
