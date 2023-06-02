
import argparse
import os
import sys
sys.path.append("../")

from dataset_notebook import PixelSetData, GroupByShapesBatchSampler
import torch
from models.stclassifier import PseLTae
import numpy as np
from torch.utils import data
from torchvision.transforms import transforms
from tqdm import tqdm
from transforms import (
    Identity,
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ShiftAug,
    ToTensor,
)
from utils.train_utils import bool_flag
from evaluation import to_cuda
from dtaidistance import dtw

def get_crop_data(crop_id, loader, model, device):

    data = []
    for batch in tqdm(loader):
        pixels, valid_pixels, positions, extra, gdd = to_cuda(batch, device)

        pe = model.temporal_encoder.position_enc(gdd)
        pe = np.double(pe.detach().cpu().numpy())

        idx = batch['index']
        for i in range(len(batch['label'])):
            if batch['label'][i] == crop_id:
                data.append({'index': idx[i], 
                             'pe': pe[i],
                             'gdd': gdd[i].detach().cpu().numpy(),
                             'extra': batch['extra'][i].detach().cpu().numpy(),
                             'pixels': pixels[i].detach().cpu().numpy(),
                             'valid_pixels': valid_pixels[i].detach().cpu().numpy(),
                             'positions': positions[i].detach().cpu().numpy()})
                
    return data


def load_model(config, path, device) -> torch.nn.Module:
    model = PseLTae(
        input_dim=config.input_dim,
        num_classes=9,
        with_extra=config.with_extra,
        with_gdd_extra=config.with_gdd_extra,
        with_pos_enc=config.with_pos_enc,
        with_gdd_pos=config.with_gdd_pos,
        pos_type=config.pos_type,
    )
    best_model_path = os.path.join(path, "model.pt")
    state_dict = torch.load(best_model_path, map_location=device)["state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    return model

def create_train_loader(datasets, config):

    train_transform = transforms.Compose(
        [
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            ShiftAug(max_shift=60, p=1.0) if config.with_shift_aug else Identity(),
            Normalize(),
            ToTensor(),
        ]
    )
        
    train_dataset = PixelSetData(
        config.data_root,
        datasets,
        config.classes,
        train_transform,
        split='train',
        fold_num=config.fold_num
    )
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
        batch_size=config.batch_size,
        drop_last=True,
    )
    return train_data_loader

def create_test_loader(datasets, config, random_sample_time_steps=False):
    """
    Create data loaders for unsupervised domain adaptation
    """
    # Test dataset
    test_transform = transforms.Compose(
        [
            RandomSampleTimeSteps(config.seq_length) if random_sample_time_steps else Identity(),
            Normalize(),
            ToTensor(),
        ]
    )
    test_dataset = PixelSetData(
        config.data_root,
        datasets,
        config.classes,
        test_transform,
        split='test',
        fold_num=config.fold_num,
    )
    test_loader = data.DataLoader(
        test_dataset,
        num_workers=config.num_workers,
        batch_sampler=GroupByShapesBatchSampler(test_dataset, config.batch_size),
    )

    print(f"evaluation dataset:", datasets)
    print(f"test data: {len(test_dataset)} ({len(test_loader)} batches)")

    return test_loader

def create_config(experiment_name, source, target, pos_type, data_root="../../data/timematch_data/", notebook=True):
    classes = sorted(['corn', 'horsebeans', 'meadow', 'spring_barley', 'unknown',
                      'winter_barley', 'winter_rapeseed', 'winter_triticale', 'winter_wheat'])
    parser = get_parser()
    config = parser.parse_args("") if notebook else parser.parse_args()
    config.classes = classes
    config.fold_num = 0
    config.num_classes = len(classes)
    config.output_dir = "./outputs"
    # change to the appropriate dataset folder path
    config.data_root = data_root
    if notebook:
        config.experiment_name = experiment_name
        config.source = source
        config.target = target
        config.pos_type = pos_type

    config.output_dir = os.path.join(config.output_dir, config.experiment_name)
    os.makedirs(config.output_dir, exist_ok=True)
    return config

def get_parser():
    parser = argparse.ArgumentParser()

    # Setup parameters
    #default="/media/data/timematch_data"
    parser.add_argument(
        "--data_root",
        default="../data/timematch_data",
        type=str,
        help="Path to datasets root directory",
    )
    parser.add_argument(
        "--num_blocks",
        default=100,
        type=int,
        help="Number of geographical blocks in dataset for splitting. Default 100.",
    )

    available_tiles = [
        "denmark/32VNH/2017",
        "france/30TXT/2017",
        "france/31TCJ/2017",
        "austria/33UVP/2017",
    ]

    parser.add_argument(
        "--source", default=["denmark/32VNH/2017"], nargs="+", choices=available_tiles
    )
    parser.add_argument(
        "--target", default=["france/30TXT/2017"], nargs="+", choices=available_tiles
    )
    parser.add_argument(
        "--num_folds", default=1, type=int, help="Number of train/test folds"
    )
    parser.add_argument(
        "--val_ratio",
        default=0.1,
        type=float,
        help="Ratio of training data to use for validation. Default 10%.",
    )
    parser.add_argument(
        "--test_ratio",
        default=0.2,
        type=float,
        help="Ratio of training data to use for testing. Default 20%.",
    )
    parser.add_argument(
        "--sample_pixels_val",
        type=bool_flag,
        default=True,
        help="speed up validation at the cost of randomness",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Path to the folder where the results should be stored",
    )
    parser.add_argument(
        "-e", "--experiment_name", default=None, help="Name of the experiment"
    )
    parser.add_argument(
        "--num_workers", default=8, type=int, help="Number of data loading workers"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Name of device to use for tensor computations",
    )
    parser.add_argument(
        "--log_step",
        default=10,
        type=int,
        help="Interval in batches between display of training metrics",
    )
    parser.add_argument("--eval", action="store_true", help="run only evaluation")
    parser.add_argument(
        "--overall", action="store_true", help="print overall results, if exists"
    )
    parser.add_argument("--combine_spring_and_winter", default=False, type=bool_flag)

    # Training configuration
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs per fold"
    )
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--steps_per_epoch", default=500, type=int, help="Batches per epoch")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument(
        "--weight_decay", default=1e-4, type=float, help="Weight decay rate"
    )
    parser.add_argument(
        "--focal_loss_gamma", default=1.0, type=float, help="gamma value for focal loss"
    )
    parser.add_argument(
        "--num_pixels",
        default=64,
        type=int,
        help="Number of pixels to sample from the input sample",
    )
    parser.add_argument(
        "--seq_length",
        default=30,
        type=int,
        help="Number of time steps to sample from the input sample",
    )
    parser.add_argument(
        "--model",
        default="pseltae",
        choices=["psetae", "pseltae", "psetcnn", "psegru"],
    )
    parser.add_argument(
        "--input_dim", default=10, type=int, help="Number of channels of input sample"
    )
    parser.add_argument(
        "--with_extra",
        default=False,
        type=bool_flag,
        help="whether to input extra geometric features to the PSE",
    )
    parser.add_argument(
        "--with_gdd_extra",
        default=False,
        type=bool_flag,
        help="whether to input extra gdd weather data",
    )
    parser.add_argument("--with_pos_enc", default=True, type=bool_flag)
    parser.add_argument("--with_gdd_pos", default=False, action="store_true")
    parser.add_argument("--with_shift_aug", default=False, action="store_true")
    parser.add_argument("--eval_target", default=True, type=bool_flag)
    parser.add_argument("--max_temporal_shift", default=60, type=int)
    parser.add_argument("--class_balance", default=False, action="store_true")
    parser.add_argument("--tile_balance", default=False, action="store_true")
    parser.add_argument("--tensorboard_log_dir", default="runs")
    parser.add_argument(
        "--train_on_target",
        default=5,
        action="store_true",
        help="supervised training on target for upper bound comparison",
    )
    parser.add_argument(
        "--pos_type",
        default="default",
        choices=['default', 'fourier', 'rnn'],
    )

    parser.add_argument(
        "--crop_type",
        default=None,
        type=str,
        help="The crop type to be plotted",
    )


    return parser


def plot_warpingpaths(s1, s2, paths, path=None, filename=None, shownumbers=False, showlegend=False,
                      figure=None, matshow_kwargs=None, s1_title="", s2_title=""):
    """Plot the warping paths matrix.

    :param s1: Series 1
    :param s2: Series 2
    :param paths: Warping paths matrix
    :param path: Path to draw (typically this is the best path)
    :param filename: Filename for the image (optional)
    :param shownumbers: Show distances also as numbers
    :param showlegend: Show colormap legend
    :param figure: Matplotlib Figure object
    :return: Figure, Axes
    """
    try:
        from matplotlib import pyplot as plt
        from matplotlib import gridspec
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        return
    ratio = max(len(s1), len(s2))
    min_y = min(np.min(s1), np.min(s2))
    max_y = max(np.max(s1), np.max(s2))

    if figure is None:
        fig = plt.figure(figsize=(10, 10), frameon=True)
    else:
        fig = figure
    if showlegend:
        grows = 3
        gcols = 3
        height_ratios = [1, 6, 1]
        width_ratios = [1, 6, 1]
    else:
        grows = 2
        gcols = 2
        height_ratios = [1, 6]
        width_ratios = [1, 6]
    gs = gridspec.GridSpec(grows, gcols, wspace=1, hspace=1,
                           left=0, right=10.0, bottom=0, top=1.0,
                           height_ratios=height_ratios,
                           width_ratios=width_ratios)
    max_s2_x = np.max(s2)
    max_s2_y = len(s2)
    max_s1_x = np.max(s1)
    min_s1_x = np.min(s1)
    max_s1_y = len(s1)

    if path is None:
        p = dtw.best_path(paths)
    elif path == -1:
        p = None
    else:
        p = path

    def format_fn2_x(tick_val, tick_pos):
        return max_s2_x - tick_val

    def format_fn2_y(tick_val, tick_pos):
        return int(max_s2_y - tick_val)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()
    if p is not None:
        ax0.text(0, 0, "Dist = {:.4f}".format(paths[p[-1][0] + 1, p[-1][1] + 1]))
    ax0.xaxis.set_major_locator(plt.NullLocator())
    ax0.yaxis.set_major_locator(plt.NullLocator())

    ax1 = fig.add_subplot(gs[0, 1])
    # ax1.set_ylim([min_y, max_y])
    ax1.set_axis_off()
    ax1.xaxis.tick_top()
    # ax1.set_aspect(0.454)
    #ax1.plot(range(len(s2)), s2, ".-")
    ax1.imshow(s2)
    # ax1.set_xlim([-0.5, len(s2) - 0.5])
    # ax1.xaxis.set_major_locator(plt.NullLocator())
    # ax1.yaxis.set_major_locator(plt.NullLocator())

    ax2 = fig.add_subplot(gs[1, 0])
    # ax2.set_xlim([-max_y, -min_y])
    ax2.set_axis_off()
    # ax2.set_aspect(0.8)
    # ax2.xaxis.set_major_formatter(FuncFormatter(format_fn2_x))
    # ax2.yaxis.set_major_formatter(FuncFormatter(format_fn2_y))
    # ax2.xaxis.set_major_locator(plt.NullLocator())
    # ax2.yaxis.set_major_locator(plt.NullLocator())
    # ax2.plot(-s1, range(max_s1_y, 0, -1), ".-")
    ax2.imshow(s1)
    # ax2.set_ylim([0.5, len(s1) + 0.5])

    ax3 = fig.add_subplot(gs[1, 1])
    # ax3.set_aspect(1)
    kwargs = {} if matshow_kwargs is None else matshow_kwargs
    img = ax3.matshow(paths[1:, 1:], **kwargs)
    # ax3.grid(which='major', color='w', linestyle='-', linewidth=0)
    # ax3.set_axis_off()
    if p is not None:
        py, px = zip(*p)
        ax3.plot(px, py, ".-", color="red")
    # ax3.xaxis.set_major_locator(plt.NullLocator())
    # ax3.yaxis.set_major_locator(plt.NullLocator())
    if shownumbers:
        for r in range(1, paths.shape[0]):
            for c in range(1, paths.shape[1]):
                ax3.text(c - 1, r - 1, "{:.2f}".format(paths[r, c]))

    gs.tight_layout(fig, pad=1.0, h_pad=1.0, w_pad=1.0)
    # fig.subplots_adjust(hspace=0, wspace=0)

    if showlegend:
        # ax4 = fig.add_subplot(gs[0:, 2])
        ax4 = fig.add_axes([0.9, 0.25, 0.015, 0.5])
        fig.colorbar(img, cax=ax4)

    # Align the subplots:
    ax1pos = ax1.get_position().bounds
    ax2pos = ax2.get_position().bounds
    ax3pos = ax3.get_position().bounds
    ax2.set_position((ax2pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax2pos[2], ax3pos[3])) # adjust the time series on the left vertically
    if len(s1) < len(s2):
        ax3.set_position((ax3pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax3pos[2], ax3pos[3])) # move the time series on the left and the distance matrix upwards
        if showlegend:
            ax4pos = ax4.get_position().bounds
            ax4.set_position((ax4pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax4pos[2], ax3pos[3])) # move the legend upwards
    if len(s1) > len(s2):
        ax3.set_position((ax1pos[0], ax3pos[1], ax3pos[2], ax3pos[3])) # move the time series at the top and the distance matrix to the left
        ax1.set_position((ax1pos[0], ax1pos[1], ax3pos[2], ax1pos[3])) # adjust the time series at the top horizontally
        if showlegend:
            ax4pos = ax4.get_position().bounds
            ax4.set_position((ax1pos[0] + ax3pos[2] + (ax1pos[0] - (ax2pos[0] + ax2pos[2])), ax4pos[1], ax4pos[2], ax4pos[3])) # move the legend to the left to equalize the horizontal spaces between the subplots
    if len(s1) == len(s2):
        ax1.set_position((ax3pos[0], ax1pos[1], ax3pos[2], ax1pos[3])) # adjust the time series at the top horizontally
        
    ax = fig.axes

    ax1.set_title(s1_title)
    ax2.set_title(s2_title)
    if filename:
        if type(filename) != str:
            filename = str(filename)
        plt.savefig(filename)
        plt.close()
        fig, ax = None, None
    return fig, ax