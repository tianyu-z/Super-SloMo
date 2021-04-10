from comet_ml import Experiment, ExistingExperiment
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import model
import dataloader_c as dataloader
import numpy as np
from utils import init_net, upload_images, get_lr
from loss import supervisedLoss, psnr, ssim
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


class Trainer:
    def __init__(self, args=args):
        super().__init__()
        self.args = args
        # random_seed setting
        random_seed = args.randomseed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(random_seed)
        else:
            torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.slomo = model.Slomo(self.args.H, self.args.W, self.device)
        self.slomo.to(self.device)
        if self.args.init_type != "":
            init_net(self.slomo, self.args.init_type)
            print(self.args.init_type + " initializing slomo done")
        if self.args.train_continue:
            if not self.args.nocomet and self.args.cometid != "":
                self.comet_exp = ExistingExperiment(
                    previous_experiment=self.args.cometid
                )
            elif not self.args.nocomet and self.args.cometid == "":
                self.comet_exp = Experiment(
                    workspace=self.args.workspace, project_name=self.args.projectname
                )
            else:
                self.comet_exp = None
            self.ckpt_dict = torch.load(self.args.checkpoint)
            self.slomo.load_state_dict(self.ckpt_dict["model_state_dict"])
            self.args.init_learning_rate = self.ckpt_dict["learningRate"]
            self.optimizer = optim.Adam(
                self.slomo.parameters(), lr=self.args.init_learning_rate
            )
            self.optimizer.load_state_dict(self.ckpt_dict["opt_state_dict"])
            print("Pretrained model loaded!")
        else:
            # start logging info in comet-ml
            if not self.args.nocomet:
                self.comet_exp = Experiment(
                    workspace=self.args.workspace, project_name=self.args.projectname
                )
                # self.comet_exp.log_parameters(flatten_opts(self.args))
            else:
                self.comet_exp = None
            self.ckpt_dict = {
                "trainLoss": {},
                "valLoss": {},
                "valPSNR": {},
                "valSSIM": {},
                "valMSE": {},
                "learningRate": {},
                "epoch": -1,
                "detail": "End to end Super SloMo.",
                "trainBatchSz": self.args.train_batch_size,
                "validationBatchSz": self.args.validation_batch_size,
            }
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.args.milestones, gamma=0.1
        )
        # Channel wise mean calculated on adobe240-fps training dataset
        mean = [0.5, 0.5, 0.5]
        std = [1, 1, 1]
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.transform = transforms.Compose([transforms.ToTensor(), self.normalize])

        trainset = dataloader.SuperSloMo(
            root=self.args.dataset_root + "/train", transform=self.transform, train=True
        )
        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
        )

        validationset = dataloader.SuperSloMo(
            root=self.args.dataset_root + "/validation",
            transform=self.transform,
            # randomCropSize=(128, 128),
            train=False,
        )
        self.validationloader = torch.utils.data.DataLoader(
            validationset,
            batch_size=self.args.validation_batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )
        ### loss
        self.supervisedloss = supervisedLoss()
        self.best = {
            "valLoss": 99999999,
            "valPSNR": -1,
            "valSSIM": -1,
            "valMSE": 99999999,
        }
        self.checkpoint_counter = int(
            (self.ckpt_dict["epoch"] + 1) / self.args.checkpoint_epoch
        )

    def train(self):
        for epoch in range(self.ckpt_dict["epoch"] + 1, self.args.epochs):
            print("Epoch: ", epoch)

            _, _, _, train_loss = self.run_epoch(
                epoch, self.trainloader, logimage=False, isTrain=True,
            )
            with torch.no_grad():
                val_psnr, val_ssim, val_mse, val_loss = self.run_epoch(
                    epoch, self.validationloader, logimage=True, isTrain=False,
                )
            self.ckpt_dict["trainLoss"][str(epoch)] = train_loss
            self.ckpt_dict["valLoss"][str(epoch)] = val_loss
            self.ckpt_dict["valPSNR"][str(epoch)] = val_psnr
            self.ckpt_dict["valSSIM"][str(epoch)] = val_ssim
            self.ckpt_dict["valMSE"][str(epoch)] = val_mse
            self.ckpt_dict["learningRate"][str(epoch)] = get_lr(self.optimizer)
            self.ckpt_dict["epoch"] = epoch

            self.best = self.save_best(self.ckpt_dict, self.best, epoch)
            if (epoch % self.args.checkpoint_epoch) == self.args.checkpoint_epoch - 1:
                self.save()

    def save_best(self, current, best, epoch):
        save_best_done = False
        for metric_name in ["valLoss", "valSSIM", "valPSNR", "valMSE"]:
            if not save_best_done:
                if ("Loss" in metric_name) or ("MSE" in metric_name):
                    if best[metric_name] > current[metric_name][str(epoch)]:
                        best[metric_name] = current[metric_name][str(epoch)]
                        self.save(metric_name)
                        print(
                            "New Best "
                            + metric_name
                            + ": "
                            + str(best[metric_name])
                            + "saved"
                        )
                        save_best_done = True
                else:
                    if best[metric_name] < current[metric_name][str(epoch)]:
                        best[metric_name] = current[metric_name][str(epoch)]
                        self.save(metric_name)
                        print(
                            "New Best "
                            + metric_name
                            + ": "
                            + str(best[metric_name])
                            + "saved"
                        )
                        save_best_done = True
        return best

    @torch.no_grad()
    def save(self, save_metric_name=""):
        self.ckpt_dict["model_state_dict"] = self.slomo.state_dict()
        self.ckpt_dict["opt_state_dict"] = self.optimizer.state_dict()
        file_name = (
            str(self.checkpoint_counter) if save_metric_name == "" else save_metric_name
        )
        torch.save(
            self.ckpt_dict,
            self.args.checkpoint_dir + "/SuperSloMo" + file_name + ".ckpt",
        )
        if save_metric_name == "":
            self.checkpoint_counter += 1

    ### Train and Valid
    def run_epoch(self, epoch, dataloader, logimage=False, isTrain=True):
        # For details see training.
        psnr_value = 0
        MSE_value = 0
        ssim_value = 0
        loss_value = 0
        if not isTrain:
            valid_images = []
        for index, all_data in enumerate(dataloader, 0):
            (
                Ft_p,
                I0,
                IFrame,
                I1,
                g_I0_F_t_0,
                g_I1_F_t_1,
                FlowBackWarp_I0_F_1_0,
                FlowBackWarp_I1_F_0_1,
                F_1_0,
                F_0_1,
            ) = self.slomo(all_data, pred_only=False, isTrain=isTrain)
            if (not isTrain) and logimage:
                if index % self.args.logimagefreq == 0:
                    valid_images.append(
                        255.0
                        * I0.cpu()[0]
                        .resize_(1, 1, self.args.data_h, self.args.data_w)
                        .repeat(1, 3, 1, 1)
                    )
                    valid_images.append(
                        255.0
                        * IFrame.cpu()[0]
                        .resize_(1, 1, self.args.data_h, self.args.data_w)
                        .repeat(1, 3, 1, 1)
                    )
                    valid_images.append(
                        255.0
                        * I1.cpu()[0]
                        .resize_(1, 1, self.args.data_h, self.args.data_w)
                        .repeat(1, 3, 1, 1)
                    )
                    valid_images.append(
                        255.0
                        * Ft_p.cpu()[0]
                        .resize_(1, 1, self.args.data_h, self.args.data_w)
                        .repeat(1, 3, 1, 1)
                    )
            # loss
            loss = self.supervisedloss(
                Ft_p,
                IFrame,
                I0,
                I1,
                g_I0_F_t_0,
                g_I1_F_t_1,
                FlowBackWarp_I0_F_1_0,
                FlowBackWarp_I1_F_0_1,
                F_1_0,
                F_0_1,
            )
            if isTrain:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            loss_value += loss.item()

            # metrics
            MSE_value += nn.MSE_LossFn(Ft_p, IFrame).item()
            psnr_value += psnr(Ft_p, IFrame, outputTensor=False)
            ssim_value += ssim(Ft_p, IFrame, outputTensor=False)

        name_loss = "TrainLoss" if isTrain else "ValLoss"
        itr = int(index + epoch * (len(dataloader)))
        if self.comet_exp is not None:
            self.comet_exp.log_metric(
                "PSNR", psnr_value / len(dataloader), step=itr, epoch=epoch
            )
            self.comet_exp.log_metric(
                "SSIM", ssim_value / len(dataloader), step=itr, epoch=epoch
            )
            self.comet_exp.log_metric(
                "MSE", MSE_value / len(dataloader), step=itr, epoch=epoch
            )
            self.comet_exp.log_metric(
                name_loss, loss_value / len(dataloader), step=itr, epoch=epoch
            )
            if logimage:
                upload_images(
                    valid_images,
                    epoch,
                    exp=self.comet_exp,
                    im_per_row=4,
                    rows_per_log=int(len(valid_images) / 4),
                )
        print(
            " Loss: %0.6f  Iterations: %4d/%4d  ValPSNR: %0.4f  ValSSIM: %0.4f ValMSE: %0.4f"
            % (
                loss_value / len(dataloader),
                index,
                len(dataloader),
                psnr_value / len(dataloader),
                ssim_value / len(dataloader),
                MSE_value / len(dataloader),
            )
        )
        return (
            (psnr_value / len(dataloader)),
            (ssim_value / len(dataloader)),
            (MSE_value / len(dataloader)),
            (loss_value / len(dataloader)),
        )


if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()
