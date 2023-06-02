#! usr/bin/env python

import os
import sys
sys.path.append("../")


from transforms import (
    Identity,
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ShiftAug,
    ToTensor,
)

from dtw.dtw_pe import DTWPositionalEncoding
from torchvision import transforms

from dataset import PixelSetData, GroupByShapesBatchSampler
import torch
from torch.utils import data

from models.stclassifier import PseLTae
from train import get_parser
from utils.train_utils import to_cuda
from evaluation import evaluation
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from dtaidistance import dtw_barycenter

import random

random.seed(42)

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


def create_test_loader(datasets, config, random_sample_time_steps=False):
    """
    Create data loaders for unsupervised domain adaptation
    """
    # Test dataset
    test_transform = transforms.Compose(
        [
            RandomSampleTimeSteps(
                config.seq_length) if random_sample_time_steps else Identity(),
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
        batch_sampler=GroupByShapesBatchSampler(
            test_dataset, config.batch_size),
    )

    print(f"evaluation dataset:", datasets)
    print(f"test data: {len(test_dataset)} ({len(test_loader)} batches)")

    return test_loader


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
        fold_num=config.fold_num,
        kept_path=config.occ_idx,
        occluded_class=config.occluded_class
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


def create_config():
    classes = sorted(['corn', 'horsebeans', 'meadow', 'spring_barley', 'unknown',
                      'winter_barley', 'winter_rapeseed', 'winter_triticale', 'winter_wheat'])
    parser = get_parser()
    config = parser.parse_args()
    config.classes = classes
    config.fold_num = 0
    config.num_classes = len(classes)
    config.output_dir = "./outputs"
    # change to the appropriate dataset folder path
    config.data_root = '../data/timematch_data/'

    config.output_dir = os.path.join(config.output_dir, config.experiment_name)
    return config


def get_mean_pe(model, loader, config, device):
    pe_source = []
    model.eval()

    for batch in tqdm(loader):
        _, _, _, _, gdd = to_cuda(batch, device)
        # Get PE values
        pe = model.temporal_encoder.position_enc(gdd)
        pe = np.double(pe.detach().cpu().numpy())

        pe_source.extend(pe)

    mean_pe = dtw_barycenter.dba(pe_source, None, use_c=True)
    return mean_pe


def get_mean_pe_per_crop(model, loader, config, device):
    pe_source_per_crop = [[] for _ in range(len(config.classes))]
    model.eval()

    for batch in tqdm(loader):
        labels_true = batch['label'].cpu().numpy()
        pixels, valid_pixels, positions, extra, gdd = to_cuda(batch, device)
        # Get PE values
        pe = model.temporal_encoder.position_enc(gdd)
        pe = np.double(pe.detach().cpu().numpy())
        # Get predictions
        logits = model.forward(pixels, valid_pixels, positions, extra, gdd)
        predictions = logits.argmax(dim=1)
        predictions = predictions.cpu().numpy()

        for crop_type in range(len(config.classes)):

            correct_idx = np.intersect1d(np.argwhere(labels_true == crop_type),
                                         np.argwhere(predictions - labels_true == 0))
            pe_source_per_crop[crop_type].extend(pe[correct_idx])

    mean_pes = []
    for pes in pe_source_per_crop:
        mean_pe = dtw_barycenter.dba(pes, None, use_c=True)
        mean_pes.append(mean_pe)

    return mean_pes


def save_results(config, results, filename=""):

    report = results['classification_dict']
    report.update({"accuracy": {"precision": None, "recall": None,
                                "f1-score": report["accuracy"], "support": report['macro avg']['support']}})

    df = pd.DataFrame(report).transpose()

    df.to_csv(f"{filename}.csv")


def main():
    config = create_config()

    print("Loading data...")
    source_test_loader = create_train_loader(config.source, config)
    target_test_loader = create_test_loader(
        config.target, config, random_sample_time_steps=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    output_dir = os.path.join("./outputs", config.experiment_name)
    fold_dir = os.path.join(output_dir, "fold_0")
    model = load_model(config, fold_dir, device)
    model.eval()

    print("Evaluating on target without DTW...")

    os.makedirs(config.output_dir, exist_ok=True)

    target_name = ','.join([str(tile).replace("/", "_")
                           for tile in config.target])
    no_dtw_results_file = f"./{config.output_dir}/no_dtw_result_{target_name}"
    if not os.path.exists(no_dtw_results_file):
        test_metrics = evaluation(
            model, target_test_loader, device, config.classes, mode='test')
        print(
            f"Test result for {config.experiment_name} on target {config.target}:\
                 accuracy={test_metrics['accuracy']:.4f}, f1={test_metrics['macro_f1']:.4f}"
        )

        print(test_metrics['classification_report'])

        save_results(config, test_metrics, no_dtw_results_file)


    if not os.path.exists(f"{config.output_dir}/mean_pe.pkl"):
        print("Computing mean PE...")
        pe_source = get_mean_pe(
            model, source_test_loader, config, device)
        with open(f"{config.output_dir}/mean_pe.pkl", "wb") as f:
            pickle.dump(pe_source, f)
    else:
        print("Loading mean PE...")
        with open(f"{config.output_dir}/mean_pe.pkl", "rb") as f:
            pe_source = pickle.load(f)

    original_pos = model.temporal_encoder.position_enc
    print(f"Replacing {type(original_pos)} with DTWPositionalEncoding")
    dtw_enc = DTWPositionalEncoding(device)

    dtw_enc.set_source_pes(pe_source)
    dtw_enc.original_pos_enc = original_pos

    model.temporal_encoder.position_enc = dtw_enc
    model.temporal_encoder.return_att = False

    # test with DTW
    print("Evaluating on target...")
    test_metrics_dtw = evaluation(
        model, target_test_loader, device, config.classes, mode='test')

    print(
        f"Test result for {config.experiment_name} on target {config.target}:\
              accuracy={test_metrics_dtw['accuracy']:.4f}, f1={test_metrics_dtw['macro_f1']:.4f}"
    )

    print(test_metrics_dtw['classification_report'])
    result_file = f"{config.output_dir}/result_mean_dtw_{target_name}"
    save_results(config, test_metrics_dtw, result_file)


if __name__ == "__main__":
    main()
