from comet_ml import Experiment, ExistingExperiment
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
from torchvision import transforms
import torch.optim as optim
from torch import nn
import os
import torch
import torch.utils.data as data
import pickle
import argparse
from PIL import Image
import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=100, type=int, help="sum of epochs")
parser.add_argument(
    "-bs", "--batchsize", default=16384, type=int, help="mini-batch size"
)
parser.add_argument(
    "-wp", "--workspace", default="tianyu-z", type=str, help="comet-ml workspace"
)
parser.add_argument(
    "-nw", "--num_workers", default=4, type=int, help="number of CPU you get"
)
parser.add_argument(
    "-se", "--save_every", default=10, type=int, help="save for every x epoches"
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
    "-pn", "--projectname", default="ccml-vae", type=str, help="comet-ml project name",
)
parser.add_argument(
    "-dh", "--data_h", default=128, type=int, help="H of the data shape"
)
parser.add_argument(
    "-dw", "--data_w", default=128, type=int, help="W of the data shape"
)
parser.add_argument(
    "--nocomet", action="store_true", help="not using comet_ml logging."
)
parser.add_argument("-lr", default=0.0001, type=float, help="G learning rate")
parser.add_argument(
    "--milestones",
    type=list,
    default=[
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        800,
        850,
        900,
        950,
        1000,
    ],
    help="Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]",
)
parser.add_argument("-ld", default=32, type=int, help="latent space dimentsion")
parser.add_argument(
    "--train_continue",
    type=bool,
    default=False,
    help="If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.",
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


random_seed = args.randomseed
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = (
            sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        )
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class VAE(torch.nn.Module):
    latent_dim = args.ld

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(256, args.ld)
        self._enc_log_sigma = torch.nn.Linear(256, args.ld)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(
            std_z, requires_grad=False
        )  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


default_data_path = {
    "200MB": "/miniscratch/tyz/datasets/CloudCast/200MB/pkls/",
    "8GB": "/miniscratch/tyz/datasets/CloudCast/8GB/pkls/",
}


def load_cs_small(root=default_data_path):
    path_train = os.path.join(root["200MB"], "train.pkl")
    path_test = os.path.join(root["200MB"], "test.pkl")
    with open(path_train, "rb") as file:
        data_train = pickle.load(file)
        data_train = data_train
    with open(path_test, "rb") as file:
        data_test = pickle.load(file)
    return [data_train, data_test]


class CloudCast(data.Dataset):
    def __init__(
        self, is_train, transform=None, batchsize=16,
    ):
        """
        param num_objects: a list of number of possible objects.
        """
        super(CloudCast, self).__init__()

        self.dataset_all = load_cs_small()
        if is_train:
            self.dataset = self.dataset_all[0]
        else:
            self.dataset = self.dataset_all[1]
        self.length = self.dataset.shape[-1]
        self.is_large = False
        self.is_train = is_train
        self.transform = transform
        self.batchsize = batchsize
        # For generating data
        if self.is_large:
            self.image_size_ = 728
        else:
            self.image_size_ = 128
        self.step_length_ = 0.1

    def __getitem__(self, idx):
        tensor = self.transform(self.dataset[:, :, idx])
        return [tensor, torch.tensor(self.batchsize)]

    def __len__(self):
        return self.length


def text_to_array(text, width=640, height=40):
    """
    Creates a numpy array of shape height x width x 3 with
    text written on it using PIL

    Args:
        text (str): text to write
        width (int, optional): Width of the resulting array. Defaults to 640.
        height (int, optional): Height of the resulting array. Defaults to 40.

    Returns:
        np.ndarray: Centered text
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), (255, 255, 255))
    try:
        font = ImageFont.truetype("UnBatang.ttf", 25)
    except OSError:
        font = ImageFont.load_default()

    d = ImageDraw.Draw(img)
    text_width, text_height = d.textsize(text)
    h = 40 // 2 - 3 * text_height // 2
    w = width // 2 - text_width
    d.text((w, h), text, font=font, fill=(30, 30, 30))
    return np.array(img)


def all_texts_to_array(texts, width=640, height=40):
    """
    Creates an array of texts, each of height and width specified
    by the args, concatenated along their width dimension

    Args:
        texts (list(str)): List of texts to concatenate
        width (int, optional): Individual text's width. Defaults to 640.
        height (int, optional): Individual text's height. Defaults to 40.

    Returns:
        list: len(texts) text arrays with dims height x width x 3
    """
    return [text_to_array(text, width, height) for text in texts]


def all_texts_to_tensors(texts, width=640, height=40):
    """
    Creates a list of tensors with texts from PIL images

    Args:
        texts (list(str)): texts to write
        width (int, optional): width of individual texts. Defaults to 640.
        height (int, optional): height of individual texts. Defaults to 40.

    Returns:
        list(torch.Tensor): len(texts) tensors 3 x height x width
    """
    arrays = all_texts_to_array(texts, width, height)
    arrays = [array.transpose(2, 0, 1) for array in arrays]
    return [torch.tensor(array) for array in arrays]


def upload_images(
    image_outputs, epoch, exp=None, im_per_row=2, rows_per_log=10, legends=[],
):
    """
    Save output image

    Args:
        image_outputs (list(torch.Tensor)): all the images to log
        im_per_row (int, optional): umber of images to be displayed per row.
            Typically, for a given task: 3 because [input prediction, target].
            Defaults to 3.
        rows_per_log (int, optional): Number of rows (=samples) per uploaded image.
            Defaults to 5.
        comet_exp (comet_ml.Experiment, optional): experiment to use.
            Defaults to None.
    """
    nb_per_log = im_per_row * rows_per_log
    n_logs = len(image_outputs) // nb_per_log + 1

    header = None
    if len(legends) == im_per_row and all(isinstance(t, str) for t in legends):
        header_width = max(im.shape[-1] for im in image_outputs)
        headers = all_texts_to_tensors(legends, width=header_width)
        header = torch.cat(headers, dim=-1)

    for logidx in range(n_logs):
        ims = image_outputs[logidx * nb_per_log : (logidx + 1) * nb_per_log]
        if not ims:
            continue
        ims = torch.stack([im.squeeze() for im in ims]).squeeze()
        image_grid = vutils.make_grid(
            ims, nrow=im_per_row, normalize=True, scale_each=True, padding=0
        )

        if header is not None:
            image_grid = torch.cat([header.to(image_grid.device), image_grid], dim=1)

        image_grid = image_grid.permute(1, 2, 0).cpu().numpy()
        exp.log_image(
            Image.fromarray((image_grid * 255).astype(np.uint8)),
            name=f"{str(epoch)}_#{logidx}",
        )


if __name__ == "__main__":

    input_dim = args.data_h * args.data_w
    batch_size = args.batchsize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(input_dim, 256, 256)
    decoder = Decoder(args.ld, 256, input_dim)
    vae = VAE(encoder, decoder)
    if args.train_continue:
        if not args.nocomet:
            comet_exp = ExistingExperiment(previous_experiment=args.cometid)
        else:
            comet_exp = None
        dict1 = torch.load(args.checkpoint)
        vae.load_state_dict(dict1["state_dict"])
        checkpoint_counter = dict1["checkpoint_counter"]
        optimizer = optim.Adam(vae.parameters(), lr=dict1["learningRate"])
    else:
        # start logging info in comet-ml
        if not args.nocomet:
            comet_exp = Experiment(
                workspace=args.workspace, project_name=args.projectname
            )
            # comet_exp.log_parameters(flatten_opts(args))
        else:
            comet_exp = None
        dict1 = {"epoch": -1, "learningRate": args.lr, "trainloss": -1, "validloss": -1}
        optimizer = optim.Adam(vae.parameters(), lr=args.lr)
        checkpoint_counter = 0

    vae.to(device)
    transform = transforms.Compose([transforms.ToTensor()])

    CloudCasttrain = CloudCast(
        is_train=True, transform=transforms.Compose([transforms.ToTensor()])
    )
    CloudCasttrainloader = torch.utils.data.DataLoader(
        CloudCasttrain,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
    )
    CloudCasttest = CloudCast(
        is_train=False, transform=transforms.Compose([transforms.ToTensor()])
    )
    CloudCasttestloader = torch.utils.data.DataLoader(
        CloudCasttest,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
    )

    len_vaild = len(CloudCasttestloader)
    interval_vaild = int(len_vaild / 4)

    criterion = nn.MSELoss()

    # scheduler to decrease learning rate by a factor of 10 at milestones.
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1
    )
    for epoch in range(dict1["epoch"] + 1, args.epochs):

        loss_train = 0
        loss_val = 0
        if epoch > dict1["epoch"] + 1:
            # Increment scheduler count
            scheduler.step()
        # if epoch == 0:  # test if validation works
        #     # validate
        #     valid_image_idxs = []
        #     valid_images = []
        #     with torch.no_grad():
        #         for i, data in enumerate(CloudCasttrainloader, 0):
        #             inputs_, classes_ = data
        #             inputs_, classes_ = (
        #                 Variable(inputs_.resize_(batch_size, input_dim)),
        #                 Variable(classes_),
        #             )
        #             dec_ = vae(inputs_)
        #             ll_ = latent_loss(vae.z_mean, vae.z_sigma)
        #             loss_ = criterion(dec_, inputs_) + ll_
        #             l_ = loss_.item()
        #             loss_val += l_
        #             if i % interval_vaild == 0:
        #                 valid_image_idxs.append(i)
        #                 valid_images.append(
        #                     255.0
        #                     * inputs_[0]
        #                     .resize_(1, 1, args.data_h, args.data_w)
        #                     .repeat(1, 3, 1, 1)
        #                 )
        #                 valid_images.append(
        #                     255.0
        #                     * dec_[0]
        #                     .resize_(1, 1, args.data_h, args.data_w)
        #                     .repeat(1, 3, 1, 1)
        #                 )
        #     print(
        #         "validloss: {:.6f},  epoch : {:02d}".format(
        #             loss_val / len(CloudCasttestloader), epoch
        #         ),
        #         end="\r",
        #         flush=True,
        #     )
        #     dict2 = {
        #         "epoch": epoch,
        #         "learningRate": get_lr(optimizer),
        #         "trainloss": loss_train,
        #         "valloss": loss_val,
        #     }
        #     comet_exp.log_metrics(dict2, step=i, epoch=epoch)
        #     upload_images(
        #         valid_images,
        #         epoch,
        #         exp=comet_exp,
        #         im_per_row=2,
        #         rows_per_log=int(len(valid_images) / 2),
        #     )
        # train
        for i, data in enumerate(CloudCasttrainloader, 0):
            inputs, classes = data
            inputs, classes = inputs.to(device), classes.to(device)
            inputs, classes = (
                Variable(inputs.resize_(batch_size, input_dim)),
                Variable(classes),
            )
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            l = loss.item()
            loss_train += l
        print(
            "trainloss: {:.6f},  epoch : {:02d}".format(
                loss_train / len(CloudCasttrainloader), epoch
            ),
            end="\r",
            flush=True,
        )
        # validate
        valid_image_idxs = []
        valid_images = []
        with torch.no_grad():
            for i, data in enumerate(CloudCasttrainloader, 0):
                inputs_, classes_ = data
                inputs_, classes_ = inputs_.to(device), classes_.to(device)
                inputs_, classes_ = (
                    Variable(inputs_.resize_(batch_size, input_dim)),
                    Variable(classes_),
                )
                dec_ = vae(inputs_)
                ll_ = latent_loss(vae.z_mean, vae.z_sigma)
                loss_ = criterion(dec_, inputs_) + ll_
                l_ = loss_.item()
                loss_val += l_
                if i % interval_vaild == 0:
                    valid_image_idxs.append(i)
                    valid_images.append(
                        255.0
                        * inputs_[0]
                        .resize_(1, 1, args.data_h, args.data_w)
                        .repeat(1, 3, 1, 1)
                    )
                    valid_images.append(
                        255.0
                        * dec_[0]
                        .resize_(1, 1, args.data_h, args.data_w)
                        .repeat(1, 3, 1, 1)
                    )
        print(
            "validloss: {:.6f},  epoch : {:02d}".format(
                loss_val / len(CloudCasttestloader), epoch
            ),
            end="\r",
            flush=True,
        )
        dict2 = {
            "epoch": epoch,
            "learningRate": get_lr(optimizer),
            "trainloss": loss_train / len(CloudCasttestloader),
            "valloss": loss_val / len(CloudCasttestloader),
        }
        comet_exp.log_metric("epoch", dict2["epoch"], epoch=epoch)
        comet_exp.log_metric("learningRate", dict2["learningRate"], epoch=epoch)
        comet_exp.log_metric("trainloss", dict2["trainloss"], epoch=epoch)
        comet_exp.log_metric("valloss", dict2["valloss"], epoch=epoch)
        upload_images(
            valid_images,
            epoch,
            exp=comet_exp,
            im_per_row=2,
            rows_per_log=int(len(valid_images) / 2),
        )

        if (epoch % args.save_every) == 0:
            dict1 = {
                "epoch": epoch,
                "timestamp": datetime.datetime.now(),
                "trainBatchSz": batch_size,
                "validationBatchSz": batch_size,
                "learningRate": get_lr(optimizer),
                "trainloss": loss_train / len(CloudCasttestloader),
                "valloss": loss_val / len(CloudCasttestloader),
                "state_dict": vae.state_dict(),
                "checkpoint_counter": checkpoint_counter,
            }
            torch.save(
                dict1, args.checkpoint_dir + "/vae" + str(checkpoint_counter) + ".ckpt",
            )
            checkpoint_counter += 1
