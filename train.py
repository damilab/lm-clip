from utils import AvgMeter, get_lr, evaluate_mlc, evaluate_slc
from functools import partial
import torch
import torch.nn as nn
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter
from asl_loss import AsymmetricLossOptimized, BalancedAsymmetricLossOptimized
from model_openai_clip import OpenAICLIPModel
import numpy as np
import dataset_loaders.voc_mlt as voc_mlt
import dataset_loaders.coco_mlt as coco_mlt
import dataset_loaders.cifar_100_lt as cifar_100_lt
import os
import clip
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from ray.tune.search.optuna import OptunaSearch
import ray
from ray.air.config import RunConfig


def train_epoch(
    model,
    train_loader,
    encoded_labels,
    optimizer,
    epoch,
):
    loss_meter = AvgMeter()

    gt_labels = []
    predict_p = []
    sf = nn.Softmax(dim=1)

    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items()}

        # Calculate train loss
        loss_mean, preds = model(batch, encoded_labels=encoded_labels, mode="train")

        label_one_hot_int = batch["label_one_hot"].to(torch.int64)

        optimizer.zero_grad(set_to_none=True)
        loss_mean.backward()
        optimizer.step()

        count = batch["image"].size(0)
        loss_meter.update(loss_mean.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

        gt_labels.extend(label_one_hot_int.cpu().numpy().tolist())
        predict_p.extend(sf(preds).cpu().detach().numpy())

    train_metrics = {}
    train_metrics["loss"] = loss_meter.avg

    # If dataset starts with cifar_100_lt, evaluate single-label classification
    if CFG.dataset.startswith("cifar_100_lt"):
        top1, top1error = evaluate_slc(predict_p, gt_labels)

        train_metrics["top1"] = top1
        train_metrics["top1error"] = top1error

    else:
        try:
            mAP, APs, mAP_head, mAP_middle, mAP_tail, AUROC, AUROCs = evaluate_mlc(
                predict_p,
                gt_labels,
                train_loader.dataset.head_classes,
                train_loader.dataset.middle_classes,
                train_loader.dataset.tail_classes,
            )

            train_metrics["mAP"] = mAP
            train_metrics["mAP_head"] = mAP_head
            train_metrics["mAP_middle"] = mAP_middle
            train_metrics["mAP_tail"] = mAP_tail
            train_metrics["AUROC"] = AUROC
        except:
            print("ValueError: Input contains NaN.")
            print(
                "train epoch[{}/{}] loss: {:.3f}".format(
                    epoch + 1, CFG.epochs, loss_meter.avg
                )
            )

            train_metrics["mAP"] = np.nan
            train_metrics["mAP_head"] = np.nan
            train_metrics["mAP_middle"] = np.nan
            train_metrics["mAP_tail"] = np.nan
            train_metrics["AUROC"] = np.nan

    return train_metrics


def valid_epoch(
    model,
    valid_loader,
    encoded_labels,
    epoch,
):
    loss_meter = AvgMeter()

    gt_labels = []
    predict_p = []
    sf = nn.Softmax(dim=1)

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items()}

        # Calculate valid loss
        loss_mean, preds = model(batch, encoded_labels=encoded_labels, mode="valid")

        label_one_hot_int = batch["label_one_hot"].to(torch.int64)

        count = batch["image"].size(0)
        loss_meter.update(loss_mean.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

        gt_labels.extend(label_one_hot_int.cpu().numpy().tolist())
        predict_p.extend(sf(preds).cpu().detach().numpy())

    valid_metrics = {}
    valid_metrics["loss"] = loss_meter.avg

    if CFG.dataset.startswith("cifar_100_lt"):
        top1, top1error = evaluate_slc(predict_p, gt_labels)

        valid_metrics["top1"] = top1
        valid_metrics["top1error"] = top1error
    else:
        mAP, APs, mAP_head, mAP_middle, mAP_tail, AUROC, AUROCs = evaluate_mlc(
            predict_p,
            gt_labels,
            valid_loader.dataset.head_classes,
            valid_loader.dataset.middle_classes,
            valid_loader.dataset.tail_classes,
        )

        valid_metrics["mAP"] = mAP
        valid_metrics["mAP_head"] = mAP_head
        valid_metrics["mAP_middle"] = mAP_middle
        valid_metrics["mAP_tail"] = mAP_tail
        valid_metrics["AUROC"] = AUROC
        # except:
        #     print("ValueError: Input contains NaN.")
        #     print(
        #         "valid epoch[{}/{}] loss: {:.3f}".format(
        #             epoch + 1, CFG.epochs, loss_meter.avg
        #         )
        #     )

        #     valid_metrics["mAP"] = np.nan
        #     valid_metrics["mAP_head"] = np.nan
        #     valid_metrics["mAP_middle"] = np.nan
        #     valid_metrics["mAP_tail"] = np.nan
        #     valid_metrics["AUROC"] = np.nan

    return valid_metrics


def save_checkpoint(
    epoch,
    model_state_dict,
    optimizer,
    train_loss_mean,
    valid_loss,
    best_mAP,
    best_mAP_tail,
    path,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss_mean,
            "valid_loss": valid_loss,
            "best_mAP": best_mAP,
            "best_mAP_tail": best_mAP_tail,
        },
        path,
    )
    print(f"Saved {path}!")


def start_training(config, CFG, run_name):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # config is ray tune hyperparameter search space
    # Apply the hyperparameters from config to CFG
    for key, value in config.items():
        setattr(CFG, key, value)

    print(CFG)

    model, image_preprocessor = clip.load(CFG.model_name, device=CFG.device, jit=False)
    model = model.float()

    if CFG.dataset == "voc_mlt":
        train_loader = voc_mlt.build_loaders(
            root="dataset_loaders/voc_mlt",
            mode="train",
            image_size=CFG.size,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            class_caption=CFG.class_caption,
            use_dataset_train_captions=CFG.use_dataset_train_captions,
            use_sample_weights=CFG.use_sample_weights,
            sample_weights_power=CFG.sample_weights_power,
            class_weights_power=CFG.class_weights_power,
        )
        valid_loader = voc_mlt.build_loaders(
            root="dataset_loaders/voc_mlt",
            mode="valid",
            image_size=CFG.size,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            class_caption=CFG.class_caption,
            use_dataset_train_captions=CFG.use_dataset_train_captions,
            use_sample_weights=False,
        )
    elif CFG.dataset == "coco_mlt":
        train_loader = coco_mlt.build_loaders(
            root="dataset_loaders/coco_mlt",
            mode="train",
            image_size=CFG.size,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            class_caption=CFG.class_caption,
            use_dataset_train_captions=CFG.use_dataset_train_captions,
            caption_max_length=CFG.max_length,
            use_sample_weights=CFG.use_sample_weights,
            sample_weights_power=CFG.sample_weights_power,
            class_weights_power=CFG.class_weights_power,
        )
        valid_loader = coco_mlt.build_loaders(
            root="dataset_loaders/coco_mlt",
            mode="valid",
            image_size=CFG.size,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            class_caption=CFG.class_caption,
            use_dataset_train_captions=CFG.use_dataset_train_captions,
            caption_max_length=CFG.max_length,
            use_sample_weights=False,
        )
    elif CFG.dataset == "cifar_100_lt_r100":
        train_loader = cifar_100_lt.build_loaders(
            mode="train",
            image_size=CFG.size,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            class_caption=CFG.class_caption,
            use_dataset_train_captions=CFG.use_dataset_train_captions,
            imbalance_factor="r-100",
            use_sample_weights=CFG.use_sample_weights,
            sample_weights_power=CFG.sample_weights_power,
            class_weights_power=CFG.class_weights_power,
        )
        valid_loader = cifar_100_lt.build_loaders(
            mode="valid",
            image_size=CFG.size,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            class_caption=CFG.class_caption,
            use_dataset_train_captions=CFG.use_dataset_train_captions,
            imbalance_factor="r-100",
            use_sample_weights=False,
        )
    else:
        raise ValueError(
            "Only voc_mlt, coco_mlt and cifar_100_lt are supported as datasets"
        )

    num_labels_train = train_loader.dataset.num_classes
    num_labels_valid = valid_loader.dataset.num_classes

    # if CFG.loss_function includes "bal"
    if "bal" in CFG.loss_function:
        asl_function_train = BalancedAsymmetricLossOptimized(
            gamma_neg=CFG.asl_gamma_neg,
            gamma_pos=CFG.asl_gamma_pos,
            clip=CFG.asl_clip,
            eps=CFG.asl_eps,
            pos_weight=(
                train_loader.dataset.class_weights.to(CFG.device)
                if CFG.use_weighted_loss
                else None
            ),
            num_labels=num_labels_train,
            label_smoothing=CFG.label_smoothing,
            return_mean=True,
        ).to(CFG.device)
        asl_function_valid = BalancedAsymmetricLossOptimized(
            gamma_neg=CFG.asl_gamma_neg,
            gamma_pos=CFG.asl_gamma_pos,
            clip=CFG.asl_clip,
            eps=CFG.asl_eps,
            pos_weight=(
                valid_loader.dataset.class_weights.to(CFG.device)
                if CFG.use_weighted_loss
                else None
            ),
            num_labels=num_labels_valid,
            label_smoothing=CFG.label_smoothing,
            return_mean=True,
        ).to(CFG.device)
    elif "asl" in CFG.loss_function:
        asl_function_train = AsymmetricLossOptimized(
            gamma_neg=CFG.asl_gamma_neg,
            gamma_pos=CFG.asl_gamma_pos,
            clip=CFG.asl_clip,
            eps=CFG.asl_eps,
        ).to(CFG.device)
        asl_function_valid = AsymmetricLossOptimized(
            gamma_neg=CFG.asl_gamma_neg,
            gamma_pos=CFG.asl_gamma_pos,
            clip=CFG.asl_clip,
            eps=CFG.asl_eps,
        ).to(CFG.device)

    model = OpenAICLIPModel(
        config=CFG,
        clip_model=model,
        train_class_weights=(
            train_loader.dataset.class_weights.to(CFG.device)
            if CFG.use_weighted_loss
            else None
        ),
        valid_class_weights=(
            valid_loader.dataset.class_weights.to(CFG.device)
            if CFG.use_weighted_loss
            else None
        ),
        asl_function_train=asl_function_train,
        asl_function_valid=asl_function_valid,
    ).to(CFG.device)

    if CFG.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CFG.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=CFG.weight_decay,
        )
    elif CFG.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CFG.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=CFG.weight_decay,
        )
    elif CFG.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=CFG.lr,
            momentum=CFG.momentum,
            weight_decay=CFG.weight_decay,
        )
    else:
        raise ValueError("Only AdamW, Adam and SGD are supported as optimizers")

    start_epoch = 0
    best_mAP = 0.0
    best_mAP_tail = 0.0

    # Load from checkpoint to continue training
    print(f"Run Name: {run_name}")

    #  Check if folder exists
    if os.path.exists(
        os.path.join("runs", run_name, "best_valid_average_precision.pt")
    ):
        print("Loading from checkpoint...")
        checkpoint = torch.load(
            os.path.join("runs", run_name, "best_valid_average_precision.pt")
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_mAP = checkpoint["best_mAP"] if "best_mAP" in checkpoint else 0.0
        best_mAP_tail = (
            checkpoint["best_mAP_tail"] if "best_mAP_tail" in checkpoint else 0.0
        )
    else:
        print("No checkpoint found. Training from scratch...")

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=0
    )
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="max", factor=0.9, patience=5
    # )

    # Calculate tokens for the labels
    encoded_labels_train = torch.cat(
        [
            clip.tokenize(label, context_length=CFG.max_length)
            for label in train_loader.dataset.label_strings
        ]
    ).to(CFG.device)
    encoded_labels_valid = torch.cat(
        [
            clip.tokenize(label, context_length=CFG.max_length)
            for label in valid_loader.dataset.label_strings
        ]
    ).to(CFG.device)

    # Set up tensorboard
    if not CFG.ray:
        writer = SummaryWriter(log_dir=f"runs/{run_name}")

    cfg_dict = {
        key: str(value)
        for key, value in CFG.__dict__.items()
        if not key.startswith("__") and not callable(key)
    }
    config_json = json.dumps(cfg_dict, indent=4, sort_keys=True, ensure_ascii=False)

    if not CFG.ray:
        writer.add_text("config", config_json, 0)

    for epoch in range(start_epoch, CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            encoded_labels=encoded_labels_train,
            optimizer=optimizer,
            epoch=epoch,
        )
        # Print metrics with 4 decimal points
        for key, value in train_metrics.items():
            print(f"train/{key}: {value:.4f}")

        model.eval()
        with torch.no_grad():
            valid_metrics = valid_epoch(
                model=model,
                valid_loader=valid_loader,
                encoded_labels=encoded_labels_valid,
                epoch=epoch,
            )
            # Print metrics with 4 decimal points
            for key, value in valid_metrics.items():
                print(f"val/{key}: {value:.4f}")

        # Unpack metrics in one dictionary with prefixes
        metrics = {"train": train_metrics, "val": valid_metrics}

        if not CFG.ray:
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in valid_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)

        model_state_dict = model.state_dict()

        if CFG.ray:
            ray.train.report(metrics)

        # Save checkpoints
        if metrics["val"]["mAP"] > best_mAP:
            best_mAP = metrics["val"]["mAP"]
            if CFG.save_best_mAP_checkpoint:
                save_checkpoint(
                    epoch,
                    model_state_dict,
                    optimizer,
                    metrics["train"]["loss"],
                    metrics["val"]["loss"],
                    best_mAP,
                    best_mAP_tail,
                    writer.log_dir + "/best_valid_mAP.pt",
                )

        if metrics["val"]["mAP_tail"] and metrics["val"]["mAP_tail"] > best_mAP_tail:
            best_mAP_tail = metrics["val"]["mAP_tail"]
            if CFG.save_best_tail_mAP_checkpoint:
                save_checkpoint(
                    epoch,
                    model_state_dict,
                    optimizer,
                    metrics["train"]["loss"],
                    metrics["val"]["loss"],
                    best_mAP,
                    best_mAP_tail,
                    writer.log_dir + "/best_valid_mAP_tail.pt",
                )

        if CFG.save_newest_checkpoint:
            save_checkpoint(
                epoch,
                model_state_dict,
                optimizer,
                metrics["train"]["loss"],
                metrics["val"]["loss"],
                best_mAP,
                best_mAP_tail,
                writer.log_dir + "/newest_train_checkpoint.pt",
            )

        # lr_scheduler.step(metrics["val"]["mAP_tail"])
        lr_scheduler.step()


if __name__ == "__main__":
    import argparse
    import importlib.util

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--ray", type=bool, default=False)
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Please provide a config .py file.")

    run_name = args.config.split("/")[-1].replace(".py", "")

    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    CFG = config_module.CFG
    CFG.ray = args.ray

    if args.ray:
        # Init ray
        ray.init(runtime_env={"working_dir": os.path.abspath(".")})

        # Configure ray tune hyperparameter search space
        config = {
            "batch_size": tune.choice([8, 16, 32, 64]),
            "asl_gamma_neg": tune.loguniform(2.0, 10.0),
            # "asl_gamma_pos": tune.loguniform(0),
            # "asl_clip": tune.loguniform(0.01, 0.1),
            "asl_mul": tune.loguniform(0.1, 10.0),
            # "label_smoothing": tune.loguniform(0.005, 0.1),
            "sample_weights_power": tune.loguniform(1.0, 2.0),
            "class_weights_power": tune.loguniform(1.0, 2.0),
            "CFG": CFG,
        }

        gpus_per_trial = 1
        cpus_per_trial = 4
        num_samples = 500
        max_num_epochs = CFG.epochs
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=5,
            metric="val/mAP_tail",
            mode="max",
            reduction_factor=2,
        )

        optuna_search = OptunaSearch(metric="val/mAP_tail", mode="max")
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(start_training, CFG=CFG, run_name=run_name),
                resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
            ),
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
                num_samples=num_samples,
                scheduler=scheduler,
            ),
            run_config=ray.train.RunConfig(
                storage_path=os.path.abspath("./ray_results"), name=run_name
            ),
            param_space=config,
        )
        result = tuner.fit()

        best_result = result.get_best_result()
        print(f"Best trial config: {best_result.config}")
        print(f"Best trial final metrics: {best_result.metrics}")
    else:
        start_training(CFG=CFG, run_name=run_name, config=dict())
