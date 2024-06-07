from utils import (
    AvgMeter,
    get_lr,
    evaluate_mlc,
)
import torch
import torch.nn as nn
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter
from asl_loss import AsymmetricLossOptimized
from model_openai_clip import OpenAICLIPModel
import numpy as np
import datasets.voc_mlt as voc_mlt
import datasets.coco_mlt as coco_mlt
import os
import clip

def train_epoch(
    model,
    train_loader,
    encoded_labels,
    optimizer,
    epoch,
):
    # Calculate text embeddings for the labels
    with torch.no_grad():
        label_embeddings = model.model.encode_text(encoded_labels)
        label_embeddings = label_embeddings / label_embeddings.norm(
            dim=-1, keepdim=True
        )

    loss_meter = AvgMeter()

    gt_labels = []
    predict_p = []
    sf = nn.Softmax(dim=1)

    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items()}

        # Calculate train loss
        loss_mean, preds = model(batch, label_embeddings=label_embeddings, mode="train")

        label_one_hot_int = batch["label_one_hot"].to(torch.int64)

        optimizer.zero_grad(set_to_none=True)
        loss_mean.backward()
        optimizer.step()

        count = batch["image"].size(0)
        loss_meter.update(loss_mean.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

        gt_labels.extend(label_one_hot_int.cpu().numpy().tolist())
        predict_p.extend(sf(preds).cpu().detach().numpy())

    try:
        mAP, APs, mAP_head, mAP_middle, mAP_tail, AUROC, AUROCs = evaluate_mlc(
            predict_p,
            gt_labels,
            train_loader.dataset.head_classes,
            train_loader.dataset.middle_classes,
            train_loader.dataset.tail_classes,
        )
        print("train mAP: {}".format(mAP))
        print("train mAP head: {}".format(mAP_head))
        print("train mAP middle: {}".format(mAP_middle))
        print("train mAP tail: {}".format(mAP_tail))

        print("train AUROC: {}".format(AUROC))
    except:
        print("ValueError: Input contains NaN.")
        print(
            "train epoch[{}/{}] loss: {:.3f}".format(
                epoch + 1, CFG.epochs, loss_meter.avg
            )
        )
        mAP = np.nan
        mAP_head = np.nan
        mAP_middle = np.nan
        mAP_tail = np.nan
        AUROC = np.nan
        APs = None
        AUROCs = None
    return (
        loss_meter.avg,
        mAP,
        APs,
        mAP_head,
        mAP_middle,
        mAP_tail,
        AUROC,
        AUROCs,
    )


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

    # Calculate text embeddings for the labels
    with torch.no_grad():
        label_embeddings = model.model.encode_text(encoded_labels)
        label_embeddings = label_embeddings / label_embeddings.norm(
            dim=-1, keepdim=True
        )

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items()}

        # Calculate valid loss
        loss_mean, preds = model(batch, label_embeddings=label_embeddings, mode="valid")

        label_one_hot_int = batch["label_one_hot"].to(torch.int64)

        count = batch["image"].size(0)
        loss_meter.update(loss_mean.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

        gt_labels.extend(label_one_hot_int.cpu().numpy().tolist())
        predict_p.extend(sf(preds).cpu().detach().numpy())

    mAP, APs, mAP_head, mAP_middle, mAP_tail, AUROC, AUROCs = evaluate_mlc(
        predict_p,
        gt_labels,
        valid_loader.dataset.head_classes,
        valid_loader.dataset.middle_classes,
        valid_loader.dataset.tail_classes,
    )

    print("valid mAP: {}".format(mAP))
    print("valid mAP head: {}".format(mAP_head))
    print("valid mAP middle: {}".format(mAP_middle))
    print("valid mAP tail: {}".format(mAP_tail))

    print("valid AUROC: {}".format(AUROC))

    return loss_meter.avg, mAP, APs, mAP_head, mAP_middle, mAP_tail, AUROC, AUROCs

def save_checkpoint(epoch, model_state_dict, optimizer, train_loss_mean, valid_loss, best_mAP, best_mAP_tail, path):
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

def train(CFG, run_name):
    model, image_preprocessor = clip.load(CFG.model_name, device=CFG.device, jit=False)
    model = model.float()

    if CFG.dataset == "voc_mlt":
        train_loader = voc_mlt.build_loaders(
            root="datasets/voc_mlt",
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
            root="datasets/voc_mlt",
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
            root="datasets/coco_mlt",
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
            root="datasets/coco_mlt",
            mode="valid",
            image_size=CFG.size,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            class_caption=CFG.class_caption,
            use_dataset_train_captions=CFG.use_dataset_train_captions,
            caption_max_length=CFG.max_length,
            use_sample_weights=False,
        )
    else:
        raise ValueError(
            "Only voc_mlt and coco_mlt are supported as datasets"
        )

    num_labels_train = train_loader.dataset.num_classes
    num_labels_valid = valid_loader.dataset.num_classes

    asl_function_train = AsymmetricLossOptimized(
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
    asl_function_valid = AsymmetricLossOptimized(
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

    model = OpenAICLIPModel(
        config=CFG,
        clip_model=model,
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
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    cfg_dict = {
        key: str(value)
        for key, value in CFG.__dict__.items()
        if not key.startswith("__") and not callable(key)
    }
    config_json = json.dumps(cfg_dict, indent=4, sort_keys=True, ensure_ascii=False)

    writer.add_text("config", config_json, 0)

    for epoch in range(start_epoch, CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        (
            train_loss_mean,
            train_mAP,
            train_APs,
            train_mAP_head,
            train_mAP_middle,
            train_mAP_tail,
            train_AUROC,
            train_AUROCs,
        ) = train_epoch(
            model=model,
            train_loader=train_loader,
            encoded_labels=encoded_labels_train,
            optimizer=optimizer,
            epoch=epoch,
        )
        
        model.eval()
        with torch.no_grad():
            (
                valid_loss,
                valid_mAP,
                valid_APs,
                valid_mAP_head,
                valid_mAP_middle,
                valid_mAP_tail,
                valid_AUROC,
                valid_AUROCs,
            ) = valid_epoch(
                model=model,
                valid_loader=valid_loader,
                encoded_labels=encoded_labels_valid,
                epoch=epoch,
            )

        writer.add_scalar("train/loss", train_loss_mean, epoch)
        writer.add_scalar("train/mAP", train_mAP, epoch)
        writer.add_scalar("train/mAP_head", train_mAP_head, epoch)
        writer.add_scalar("train/mAP_middle", train_mAP_middle, epoch)
        writer.add_scalar("train/mAP_tail", train_mAP_tail, epoch)
        writer.add_scalar("train/auroc", train_AUROC, epoch)

        writer.add_scalar("val/loss", valid_loss, epoch)
        writer.add_scalar("val/mAP", valid_mAP, epoch)
        writer.add_scalar("val/mAP_head", valid_mAP_head, epoch)
        writer.add_scalar("val/mAP_middle", valid_mAP_middle, epoch)
        writer.add_scalar("val/mAP_tail", valid_mAP_tail, epoch)
        writer.add_scalar("val/auroc", valid_AUROC, epoch)

        model_state_dict = model.state_dict()

        # Save checkpoints
        if valid_mAP > best_mAP:
            best_mAP = valid_mAP
            if CFG.save_best_mAP_checkpoint:
                save_checkpoint(epoch, model_state_dict, optimizer, train_loss_mean, valid_loss, best_mAP, best_mAP_tail, writer.log_dir + "/best_valid_mAP.pt")

        if valid_mAP_tail > best_mAP_tail:
            best_mAP_tail = valid_mAP_tail
            if CFG.save_best_tail_mAP_checkpoint:
                save_checkpoint(epoch, model_state_dict, optimizer, train_loss_mean, valid_loss, best_mAP, best_mAP_tail, writer.log_dir + "/best_valid_mAP_tail.pt")

        if CFG.save_newest_checkpoint:
            save_checkpoint(epoch, model_state_dict, optimizer, train_loss_mean, valid_loss, best_mAP, best_mAP_tail, writer.log_dir + "/newest_train_checkpoint.pt")

        lr_scheduler.step()


if __name__ == "__main__":
    import argparse
    import importlib.util

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Please provide a config .py file.")

    run_name = args.config.split("/")[-1].replace(".py", "")

    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    CFG = config_module.CFG

    train(CFG, run_name)
