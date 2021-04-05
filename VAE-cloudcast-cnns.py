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
import torch.utils.data as data
import pickle
import argparse
from PIL import Image
import datetime

# https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=100, type=int, help="sum of epochs")
parser.add_argument(
    "-bs", "--batchsize", default=16384, type=int, help="mini-batch size"
)
parser.add_argument(
    "-wp", "--workspace", default="tianyu-z", type=str, help="comet-ml workspace"
)
parser.add_argument(
    "-it",
    "--init_type",
    default="kaiming",
    type=str,
    help="the name of an initialization method: normal | xavier | kaiming | orthogonal",
)

parser.add_argument(
    "-nw",
    "--num_workers",
    default=0,
    type=int,
    help="number of CPU you get, needs to be 0 when you have a super large batchsize (>1024)",
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
torch.backends.cudnn.benchmark = True


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32, device=None):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, True),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(16, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

        self.device = device

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        if device is None:
            esp = torch.randn(*mu.size())
        else:
            esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        # print(h.shape)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        # print("after fc3", z.shape)
        z = self.decoder(z)
        # z = UnFlatten()(z)
        # print("1: ", z.shape)
        # z = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2).cuda()(z)
        # print("2: ", z.shape)
        # z = nn.LeakyReLU(0.2, True).cuda()(z)
        # z = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2).cuda()(z)
        # print("3: ", z.shape)
        # z = nn.LeakyReLU(0.2, True).cuda()(z)
        # z = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2).cuda()(z)
        # print("4: ", z.shape)
        # z = nn.LeakyReLU(0.2, True).cuda()(z)
        # z = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2).cuda()(z)
        # print("5: ", z.shape)
        # z = nn.LeakyReLU(0.2, True).cuda()(z)
        # z = nn.ConvTranspose2d(16, 1, kernel_size=6, stride=2).cuda()(z)
        # print("5: ", z.shape)
        # z = nn.LeakyReLU(0.2, True).cuda()(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        # print("before fc3", z.shape)
        z = self.decode(z)
        # print(z.shape)
        return z, mu, logvar


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


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


# utils below


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


def init_weights(net, init_type="normal", init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if debug:
                print(classname)
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(
    net,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
    debug=False,
    initialize_weights=True,
):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
        print("Model weights initialized!")
    return net


if __name__ == "__main__":

    input_dim = args.data_h * args.data_w
    batch_size = args.batchsize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae = VAE(image_channels=1, h_dim=1024, z_dim=128)
    vae = vae.to(device)
    vae = init_net(vae, args.init_type)
    if args.train_continue:
        if not args.nocomet:
            comet_exp = ExistingExperiment(previous_experiment=args.cometid)
        else:
            comet_exp = None
        dict1 = torch.load(args.checkpoint)
        vae.load_state_dict(dict1["state_dict"])
        vae = vae.to(device)
        checkpoint_counter = dict1["checkpoint_counter"]
        # print(next(vae.parameters()).is_cuda)
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
        # print(next(vae.parameters()).is_cuda)
        optimizer = optim.Adam(vae.parameters(), lr=args.lr)
        checkpoint_counter = 0

    # transform = transforms.Compose([transforms.ToTensor()])

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
    interval_vaild = int(len_vaild / 2)

    criterion = nn.MSELoss()

    # scheduler to decrease learning rate by a factor of 10 at milestones.
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1
    )
    for epoch in range(dict1["epoch"] + 1, args.epochs):

        train = {"loss": 0, "bce": 0, "kld": 0}
        val = {"loss": 0, "bce": 0, "kld": 0}
        if epoch > dict1["epoch"] + 1:
            # Increment scheduler count
            scheduler.step()
        for i, d in enumerate(CloudCasttrainloader, 0):
            inputs, classes = d
            inputs, classes = (
                inputs.resize_(batch_size, 1, args.data_h, args.data_w).to(device),
                classes.to(device),
            )
            optimizer.zero_grad()
            recon_images, mu, logvar = vae(inputs)
            loss, bce, kld = loss_fn(recon_images, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            train["loss"] += loss.item()
            train["bce"] += bce.item()
            train["kld"] += kld.item()
        to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(
            epoch,
            args.epochs,
            train["loss"] / (batch_size * len(CloudCasttrainloader)),
            train["bce"] / (batch_size * len(CloudCasttrainloader)),
            train["kld"] / (batch_size * len(CloudCasttrainloader)),
        )
        print(to_print)
        comet_exp.log_metric(
            "trainloss",
            train["loss"] / (batch_size * len(CloudCasttrainloader)),
            epoch=epoch,
        )
        comet_exp.log_metric(
            "bce", train["bce"] / (batch_size * len(CloudCasttrainloader)), epoch=epoch
        )
        comet_exp.log_metric(
            "kld", train["kld"] / (batch_size * len(CloudCasttrainloader)), epoch=epoch
        )
        # validate
        valid_image_idxs = []
        valid_images = []
        with torch.no_grad():
            for i, d in enumerate(CloudCasttrainloader, 0):
                inputs_, classes_ = d
                inputs_, classes_ = (
                    inputs_.resize_(batch_size, 1, args.data_h, args.data_w).to(device),
                    classes_.to(device),
                )
                recon_images, mu, logvar = vae(inputs_)
                loss, bce, kld = loss_fn(recon_images, inputs, mu, logvar)
                val["loss"] += loss.item()
                val["bce"] += bce.item()
                val["kld"] += kld.item()
                if i % interval_vaild == 0:
                    valid_image_idxs.append(i)
                    valid_images.append(
                        255.0
                        * inputs_[0]
                        .cpu()
                        .resize_(1, 1, args.data_h, args.data_w)
                        .repeat(1, 3, 1, 1)
                    )
                    valid_images.append(
                        255.0
                        * recon_images[0]
                        .cpu()
                        .resize_(1, 1, args.data_h, args.data_w)
                        .repeat(1, 3, 1, 1)
                    )
        to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(
            epoch,
            args.epochs,
            val["loss"] / (batch_size * len(CloudCasttestloader)),
            val["bce"] / (batch_size * len(CloudCasttestloader)),
            val["kld"] / (batch_size * len(CloudCasttestloader)),
        )
        print(to_print)
        comet_exp.log_metric(
            "valloss",
            val["loss"] / (batch_size * len(CloudCasttestloader)),
            epoch=epoch,
        )
        comet_exp.log_metric(
            "valbce", val["bce"] / (batch_size * len(CloudCasttestloader)), epoch=epoch
        )
        comet_exp.log_metric(
            "valkld", val["kld"] / (batch_size * len(CloudCasttestloader)), epoch=epoch
        )

        upload_images(
            valid_images,
            epoch,
            exp=comet_exp,
            im_per_row=2,
            rows_per_log=int(len(valid_images) / 2),
        )
        comet_exp.log_metric("learningRate", get_lr(optimizer), epoch=epoch)
        if (epoch % args.save_every) == 0:
            dict1 = {
                "epoch": epoch,
                "timestamp": datetime.datetime.now(),
                "trainBatchSz": batch_size,
                "validationBatchSz": batch_size,
                "learningRate": get_lr(optimizer),
                "trainloss": train["loss"] / (batch_size * len(CloudCasttestloader)),
                "valloss": val["loss"] / (batch_size * len(CloudCasttestloader)),
                "state_dict": vae.state_dict(),
                "checkpoint_counter": checkpoint_counter,
            }
            torch.save(
                dict1, args.checkpoint_dir + "/vae" + str(checkpoint_counter) + ".ckpt",
            )
            checkpoint_counter += 1
