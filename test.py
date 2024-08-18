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
from asl_loss import BalancedAsymmetricLossOptimized
from model_openai_clip import OpenAICLIPModel
import numpy as np
import dataset_loaders.voc_mlt as voc_mlt
import dataset_loaders.coco_mlt as coco_mlt
import os
import clip

DATA_SPLIT = "valid"


def test(CFG, run_name, checkpoint_name, zeroshot):
    # Deactivate loss functions for testing
    CFG.loss_function = []

    # Load the dataset
    if CFG.dataset == "voc_mlt":
        data_loader = voc_mlt.build_loaders(
            root="dataset_loaders/voc_mlt",
            mode=DATA_SPLIT,
            image_size=CFG.size,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            class_caption=CFG.class_caption,
            use_dataset_train_captions=CFG.use_dataset_train_captions,
            use_sample_weights=False,
            use_data_augmentation=False,
        )
    elif CFG.dataset == "coco_mlt":
        data_loader = coco_mlt.build_loaders(
            root="dataset_loaders/coco_mlt",
            mode=DATA_SPLIT,
            image_size=CFG.size,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            class_caption=CFG.class_caption,
            use_dataset_train_captions=CFG.use_dataset_train_captions,
            use_sample_weights=False,
            use_data_augmentation=False,
        )
    else:
        raise ValueError("Only voc_mlt and coco_mlt are supported as datasets")

    num_labels = data_loader.dataset.num_classes

    # Load the model from the checkpoint
    clip_model, image_preprocessor = clip.load(
        CFG.model_name, device=CFG.device, jit=False
    )
    clip_model = clip_model.float()
    model = OpenAICLIPModel(
        config=CFG,
        clip_model=clip_model,
        train_class_weights=None,
        valid_class_weights=None,
    ).to(CFG.device)

    # Load from checkpoint
    if not zeroshot:
        checkpoint = torch.load(f"runs/{run_name}/{checkpoint_name}.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        print(f"Loaded run {run_name} checkpoint {checkpoint_name} at epoch {epoch}")
    else:
        epoch = "NONE"
        print(f"Loaded config {run_name} for CLIP zero-shot evaluation")

    # Calculate tokens for the labels
    encoded_labels = torch.cat(
        [
            clip.tokenize(label, context_length=CFG.max_length)
            for label in data_loader.dataset.label_strings
        ]
    ).to(CFG.device)

    # Calculate text embeddings for the labels
    with torch.no_grad():
        label_embeddings = model.model.encode_text(encoded_labels)
        label_embeddings = label_embeddings / label_embeddings.norm(
            dim=-1, keepdim=True
        )

    ids = []
    image_features = []
    caption_features = []
    classes_one_hot = []
    predictions = []

    tqdm_object = tqdm(data_loader, total=len(data_loader))
    activation_function = nn.Softmax(dim=1)
    # activation_function = nn.Sigmoid()

    class_predictions = {}
    for i in range(num_labels):
        class_predictions[i] = {
            "image_features": [],
            "caption_features": [],
            "classes_one_hot": [],
            "predictions": [],
        }

    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items()}

        with torch.no_grad():
            image_embeddings = model.model.encode_image(batch["image"])
            caption_embeddings = model.model.encode_text(batch["caption"])

            image_embeddings = image_embeddings / image_embeddings.norm(
                dim=-1, keepdim=True
            )
            caption_embeddings = caption_embeddings / caption_embeddings.norm(
                dim=-1, keepdim=True
            )

        # Add samples that have the classes we want to sample
        for i in range(batch["label_one_hot"].shape[0]):
            ids.append(batch["idx"][i])
            image_features.append(image_embeddings[i].unsqueeze(0))
            caption_features.append(caption_embeddings[i].unsqueeze(0))
            classes_one_hot.append(batch["label_one_hot"][i].unsqueeze(0))

            with torch.no_grad():
                logit_scale = model.model.logit_scale.exp()
                dot_similarity = (
                    logit_scale * image_embeddings[i].unsqueeze(0) @ label_embeddings.T
                )
            preds = activation_function(dot_similarity)

            predictions.append(preds)

    image_features = torch.cat(image_features, dim=0)
    caption_features = torch.cat(caption_features, dim=0)
    classes_one_hot = torch.cat(classes_one_hot, dim=0)
    predictions = torch.cat(predictions, dim=0)

    # Calculate the average precision
    try:
        classes_one_hot = classes_one_hot.cpu().numpy()
        predictions = predictions.cpu().numpy()
    except AttributeError:
        pass

    mAP, APs, mAP_head, mAP_middle, mAP_tail, AUROC, AUROCs = evaluate_mlc(
        predictions,
        classes_one_hot,
        data_loader.dataset.head_classes,
        data_loader.dataset.middle_classes,
        data_loader.dataset.tail_classes,
    )

    # Calculate the average precision on only multi-label samples
    predictions_multi = []
    classes_one_hot_multi = []
    image_features_multi = []
    for i in range(len(classes_one_hot)):
        if classes_one_hot[i].sum() > 1:
            predictions_multi.append(predictions[i])
            classes_one_hot_multi.append(classes_one_hot[i])
            image_features_multi.append(image_features[i])

    (
        mAP_multi,
        APs_multi,
        mAP_head_multi,
        mAP_middle_multi,
        mAP_tail_multi,
        AUROC_multi,
        AUROCs_multi,
    ) = evaluate_mlc(
        predictions_multi,
        classes_one_hot_multi,
        data_loader.dataset.head_classes,
        data_loader.dataset.middle_classes,
        data_loader.dataset.tail_classes,
    )

    # Calculate the average precision on only single-label samples
    predictions_single = []
    classes_one_hot_single = []
    image_features_single = []
    for i in range(len(classes_one_hot)):
        if classes_one_hot[i].sum() == 1:
            predictions_single.append(predictions[i])
            classes_one_hot_single.append(classes_one_hot[i])
            image_features_single.append(image_features[i])

    (
        mAP_single,
        APs_single,
        mAP_head_single,
        mAP_middle_single,
        mAP_tail_single,
        AUROC_single,
        AUROCs_single,
    ) = evaluate_mlc(
        predictions_single,
        classes_one_hot_single,
        data_loader.dataset.head_classes,
        data_loader.dataset.middle_classes,
        data_loader.dataset.tail_classes,
        calculate_auroc=False,
    )

    image_features_cpu = image_features.cpu()

    # Calculate centroids of all classes
    class_centroids = torch.zeros(num_labels, image_features_cpu.size(-1))
    positive_class_distances = []
    negative_class_distances = []

    positive_head_class_distances = []
    negative_head_class_distances = []

    positive_middle_class_distances = []
    negative_middle_class_distances = []

    positive_tail_class_distances = []
    negative_tail_class_distances = []

    # Calculate class centroids
    for i in range(num_labels):
        class_samples = []
        for j in range(len(classes_one_hot)):
            if classes_one_hot[j, i] == 1:
                class_samples.append(image_features_cpu[j])

        class_samples = torch.stack(class_samples)
        class_centroids[i] = class_samples.mean(dim=0)

    for i in range(num_labels):
        for j in range(len(classes_one_hot)):
            distance = torch.nn.functional.pairwise_distance(
                image_features_cpu[j], class_centroids[i]
            )

            if classes_one_hot[j, i] == 1:
                positive_class_distances.append(distance)

                if i in data_loader.dataset.head_classes:
                    positive_head_class_distances.append(distance)
                elif i in data_loader.dataset.middle_classes:
                    positive_middle_class_distances.append(distance)
                elif i in data_loader.dataset.tail_classes:
                    positive_tail_class_distances.append(distance)
                else:
                    raise ValueError("Class is not in any class group")
            else:
                negative_class_distances.append(distance)

                if i in data_loader.dataset.head_classes:
                    negative_head_class_distances.append(distance)
                elif i in data_loader.dataset.middle_classes:
                    negative_middle_class_distances.append(distance)
                elif i in data_loader.dataset.tail_classes:
                    negative_tail_class_distances.append(distance)
                else:
                    raise ValueError("Class is not in any class group")

    # Calculate mean of distances
    mean_positive_class_distances = torch.stack(positive_class_distances).mean()
    mean_negative_class_distances = torch.stack(negative_class_distances).mean()
    positive_distance_over_negative_distance = (
        mean_positive_class_distances / mean_negative_class_distances
    )

    mean_positive_head_class_distances = torch.stack(
        positive_head_class_distances
    ).mean()
    mean_negative_head_class_distances = torch.stack(
        negative_head_class_distances
    ).mean()
    positive_head_distance_over_negative_head_distance = (
        mean_positive_head_class_distances / mean_negative_head_class_distances
    )

    mean_positive_middle_class_distances = torch.stack(
        positive_middle_class_distances
    ).mean()
    mean_negative_middle_class_distances = torch.stack(
        negative_middle_class_distances
    ).mean()
    positive_middle_distance_over_negative_middle_distance = (
        mean_positive_middle_class_distances / mean_negative_middle_class_distances
    )

    mean_positive_tail_class_distances = torch.stack(
        positive_tail_class_distances
    ).mean()
    mean_negative_tail_class_distances = torch.stack(
        negative_tail_class_distances
    ).mean()
    positive_tail_distance_over_negative_tail_distance = (
        mean_positive_tail_class_distances / mean_negative_tail_class_distances
    )

    # Calculate multi-label class distances
    positive_class_distances_multi = []
    negative_class_distances_multi = []

    positive_head_class_distances_multi = []
    negative_head_class_distances_multi = []

    positive_middle_class_distances_multi = []
    negative_middle_class_distances_multi = []

    positive_tail_class_distances_multi = []
    negative_tail_class_distances_multi = []

    for i in range(num_labels):
        for j in range(len(classes_one_hot)):
            # Skip samples that are not multi-label
            if classes_one_hot[j].sum() < 2:
                continue

            distance = torch.nn.functional.pairwise_distance(
                image_features_cpu[j], class_centroids[i]
            )

            if classes_one_hot[j, i] == 1:
                positive_class_distances_multi.append(distance)

                if i in data_loader.dataset.head_classes:
                    positive_head_class_distances_multi.append(distance)
                elif i in data_loader.dataset.middle_classes:
                    positive_middle_class_distances_multi.append(distance)
                elif i in data_loader.dataset.tail_classes:
                    positive_tail_class_distances_multi.append(distance)
                else:
                    raise ValueError("Class is not in any class group")
            else:
                negative_class_distances_multi.append(distance)

                if i in data_loader.dataset.head_classes:
                    negative_head_class_distances_multi.append(distance)
                elif i in data_loader.dataset.middle_classes:
                    negative_middle_class_distances_multi.append(distance)
                elif i in data_loader.dataset.tail_classes:
                    negative_tail_class_distances_multi.append(distance)
                else:
                    raise ValueError("Class is not in any class group")

    # Calculate mean of distances
    mean_positive_class_distances_multi = torch.stack(
        positive_class_distances_multi
    ).mean()
    mean_negative_class_distances_multi = torch.stack(
        negative_class_distances_multi
    ).mean()
    positive_distance_over_negative_distance_multi = (
        mean_positive_class_distances_multi / mean_negative_class_distances_multi
    )

    mean_positive_head_class_distances_multi = torch.stack(
        positive_head_class_distances_multi
    ).mean()
    mean_negative_head_class_distances_multi = torch.stack(
        negative_head_class_distances_multi
    ).mean()
    positive_head_distance_over_negative_head_distance_multi = (
        mean_positive_head_class_distances_multi
        / mean_negative_head_class_distances_multi
    )

    mean_positive_middle_class_distances_multi = torch.stack(
        positive_middle_class_distances_multi
    ).mean()
    mean_negative_middle_class_distances_multi = torch.stack(
        negative_middle_class_distances_multi
    ).mean()
    positive_middle_distance_over_negative_middle_distance_multi = (
        mean_positive_middle_class_distances_multi
        / mean_negative_middle_class_distances_multi
    )

    mean_positive_tail_class_distances_multi = torch.stack(
        positive_tail_class_distances_multi
    ).mean()
    mean_negative_tail_class_distances_multi = torch.stack(
        negative_tail_class_distances_multi
    ).mean()
    positive_tail_distance_over_negative_tail_distance_multi = (
        mean_positive_tail_class_distances_multi
        / mean_negative_tail_class_distances_multi
    )

    # Print with precision
    print(run_name, checkpoint_name, "epoch", epoch, CFG.dataset, DATA_SPLIT)
    print(
        "PD: {:.4f} ND: {:.4f} PD/ND: {:.4f}".format(
            mean_positive_class_distances.item(),
            mean_negative_class_distances.item(),
            positive_distance_over_negative_distance.item(),
        )
    )
    print(
        "PD head: {:.4f} ND head: {:.4f} PD/ND head: {:.4f}".format(
            mean_positive_head_class_distances.item(),
            mean_negative_head_class_distances.item(),
            positive_head_distance_over_negative_head_distance.item(),
        )
    )
    print(
        "PD middle: {:.4f} ND middle: {:.4f} PD/ND middle: {:.4f}".format(
            mean_positive_middle_class_distances.item(),
            mean_negative_middle_class_distances.item(),
            positive_middle_distance_over_negative_middle_distance.item(),
        )
    )
    print(
        "PD tail: {:.4f} ND tail: {:.4f} PD/ND tail: {:.4f}".format(
            mean_positive_tail_class_distances.item(),
            mean_negative_tail_class_distances.item(),
            positive_tail_distance_over_negative_tail_distance.item(),
        )
    )
    print()
    print(
        "PD multi: {:.4f} ND multi: {:.4f} PD/ND multi: {:.4f}".format(
            mean_positive_class_distances_multi.item(),
            mean_negative_class_distances_multi.item(),
            positive_distance_over_negative_distance_multi.item(),
        )
    )
    print(
        "PD multi head: {:.4f} ND multi head: {:.4f} PD/ND multi head: {:.4f}".format(
            mean_positive_head_class_distances_multi.item(),
            mean_negative_head_class_distances_multi.item(),
            positive_head_distance_over_negative_head_distance_multi.item(),
        )
    )
    print(
        "PD multi middle: {:.4f} ND multi middle: {:.4f} PD/ND multi middle: {:.4f}".format(
            mean_positive_middle_class_distances_multi.item(),
            mean_negative_middle_class_distances_multi.item(),
            positive_middle_distance_over_negative_middle_distance_multi.item(),
        )
    )
    print(
        "PD multi tail: {:.4f} ND multi tail: {:.4f} PD/ND multi tail: {:.4f}".format(
            mean_positive_tail_class_distances_multi.item(),
            mean_negative_tail_class_distances_multi.item(),
            positive_tail_distance_over_negative_tail_distance_multi.item(),
        )
    )
    print()

    print("mAP: {:.4f}".format(mAP))
    print("mAP head: {:.4f}".format(mAP_head))
    print("mAP middle: {:.4f}".format(mAP_middle))
    print("mAP tail: {:.4f}".format(mAP_tail))
    print()
    print("mAP single: {:.4f}".format(mAP_single))
    print("mAP head single: {:.4f}".format(mAP_head_single))
    print("mAP middle single: {:.4f}".format(mAP_middle_single))
    print("mAP tail single: {:.4f}".format(mAP_tail_single))
    print()
    print("mAP multi: {:.4f}".format(mAP_multi))
    print("mAP head multi: {:.4f}".format(mAP_head_multi))
    print("mAP middle multi: {:.4f}".format(mAP_middle_multi))
    print("mAP tail multi: {:.4f}".format(mAP_tail_multi))

    from sklearn.manifold import TSNE
    import time
    import torch

    # Generate features of all images and their captions
    all_classes = np.arange(num_labels)

    # cat dog [7, 11]
    # cow horse sheep [9, 12, 16]
    classes_to_sample = [11, 12, 14]
    # classes_to_sample = all_classes
    classes_not_to_sample = np.setdiff1d(all_classes, classes_to_sample)
    STRICT_CLASS_FILTER = True

    features = image_features

    start = time.time()
    perplexity = min(len(features) - 1, 30)
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    projections = tsne.fit_transform(features.cpu())
    end = time.time()
    print(f"generating projections with T-SNE took: {(end - start):.2f} seconds")

    COLOR_TO_SHOW = "class"  # "class" or "AP"
    CFG.prediction_threshold = 0.25

    try:
        classes_one_hot = classes_one_hot.cpu().numpy()
        predictions = predictions.cpu().numpy()
    except AttributeError:
        pass

    predictions_one_hot = (predictions > CFG.prediction_threshold).astype(int)

    # Calculate colors for the classes based on HSV
    class_hues = []
    index = 0
    for i in range(len(data_loader.dataset.label_strings)):
        class_hues.append(index / len(classes_to_sample) * 330)
        if i in classes_to_sample:
            index += 1

    mAP, _ = eval_map(predictions, classes_one_hot)
    APs = []
    for i in range(len(classes_one_hot)):
        preds = predictions[i]
        labels = classes_one_hot[i]

        AP, _ = eval_map(preds, labels)
        APs.append(AP)

    # Add classes and (head, middle, tail) to the image_labels
    projection_labels = []
    for i in range(len(classes_one_hot)):
        classes_text = []
        for j in range(len(classes_one_hot[i])):
            if classes_one_hot[i][j] == 1:
                class_split_text = ""
                if j in data_loader.dataset.head_classes:
                    class_split_text = "HEAD"
                elif j in data_loader.dataset.middle_classes:
                    class_split_text = "MIDDLE"
                elif j in data_loader.dataset.tail_classes:
                    class_split_text = "TAIL"
                classes_text.append(
                    data_loader.dataset.classes[j]["name"]
                    + "("
                    + class_split_text
                    + ")"
                )
        projection_labels.append(",".join(classes_text))

    projection_prediction_texts = []
    for i in range(len(predictions_one_hot)):
        classes_text = []
        for j in range(len(predictions_one_hot[i])):
            if predictions_one_hot[i][j] == 1:
                classes_text.append(data_loader.dataset.classes[j]["name"])
        projection_prediction_texts.append(",".join(classes_text))

    projections_final = projections

    import pandas as pd

    projections_df = pd.DataFrame(projections_final, columns=["x", "y"])
    projections_df["labels"] = projection_labels
    projections_df["predictions"] = projection_prediction_texts

    # Write dataframe to csv
    projections_df.to_csv(
        "../runs/"
        + run_name
        + "/"
        + run_name
        + "_on"
        + CFG.dataset
        + "_"
        + DATA_SPLIT
        + ".csv",
        index=False,
    )


if __name__ == "__main__":
    import argparse
    import importlib.util

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--checkpoint", type=str, default="best_valid_mAP")
    parser.add_argument("--zeroshot", type=bool, default=False)
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Please provide a config .py file.")

    run_name = args.config.split("/")[-1].replace(".py", "")
    checkpoint_name = args.checkpoint.replace(".pt", "")

    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    CFG = config_module.CFG

    test(CFG, run_name, checkpoint_name, args.zeroshot)
