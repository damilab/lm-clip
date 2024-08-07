import torch
import torchvision.transforms as transforms
import os
import pickle
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
import numpy as np
import clip
from datasets import load_dataset


class CIFAR100LT(torch.utils.data.Dataset):
    def __init__(
        self,
        mode,
        image_size,
        class_caption,
        use_dataset_train_captions,
        imbalance_factor,
        caption_max_length=77,
        class_weights_power=1,
        sample_weights_power=1,
        use_data_augmentation=False,
    ):
        self.classes = [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "cra",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
        ]
        self.num_classes = len(self.classes)

        self.use_data_augmentation = use_data_augmentation
        self.imbalance_factor = imbalance_factor

        dataset_mode = "train" if mode == "train" else "test"
        self.dataset = load_dataset(
            "tomas-gajarsky/cifar100-lt", self.imbalance_factor, split=dataset_mode
        )
        self.num_samples = len(self.dataset)

        self.captions = []
        for sample in self.dataset:
            caption = class_caption + self.classes[sample["fine_label"]]
            self.captions.append(caption)

        self.mode = mode
        self.image_size = image_size
        self.transforms = self.get_transforms(self.mode)
        self.class_caption = class_caption

        print("Tokenizing captions...")
        self.encoded_captions = []
        for caption in self.captions:
            encoded_caption = None
            split = caption.split(" ")
            max_words = len(split)
            while encoded_caption is None:
                try:
                    encoded_caption = clip.tokenize(
                        " ".join(split[: max_words + 1]),
                        context_length=caption_max_length,
                    )
                except:
                    max_words = max_words - 1
                    print(
                        f"Caption {caption} is too long, trying with {max_words} words."
                    )
            self.encoded_captions.append(encoded_caption)

        # Turn into tensor
        self.encoded_captions = torch.cat(self.encoded_captions)

        self.label_strings = list(
            map(lambda x: self.class_caption + x, self.classes)
        )  # a photo of a _
        self.num_classes = len(self.classes)

        # One-hot encoding of labels
        self.labels_one_hot = np.zeros((self.num_samples, self.num_classes))
        for i in range(self.num_samples):
            self.labels_one_hot[i, self.dataset[i]["fine_label"]] = 1

        # Calculate class weights loss weighting and weighted sampling based on the distribution of classes in the dataset
        self.class_weights = torch.zeros(self.num_classes, dtype=float)
        self.sample_weights = torch.zeros(self.num_samples, dtype=float)

        number_of_occurences = np.zeros(self.num_classes)
        for i in range(self.num_samples):
            number_of_occurences[self.dataset[i]["fine_label"]] += 1

        for i in range(self.num_classes):
            self.class_weights[i] = self.num_samples / number_of_occurences[i]
            print(
                f"Class {self.classes[i]} has {number_of_occurences[i]} occurences, weight is {self.class_weights[i]}."
            )

        # Calculate sample weights based on class weights, more weight for samples from underrepresented classes
        for i in range(self.num_samples):
            self.sample_weights[i] = self.class_weights[self.dataset[i]["fine_label"]]

        # Apply weighting exponents
        self.class_weights = self.class_weights**class_weights_power
        self.sample_weights = self.sample_weights**sample_weights_power

        # Normalize weights
        self.class_weights = self.class_weights / torch.max(self.class_weights)
        self.sample_weights = self.sample_weights / torch.max(self.sample_weights)

        number_of_unique_images = self.num_samples
        print(
            f"Loaded {number_of_unique_images} caption-image pairs, {self.num_classes} classes, imbalance factor is {self.imbalance_factor}."
        )

    def __getitem__(self, idx):
        item = {}

        image = self.dataset[idx]["img"]
        image = self.transforms(image)

        item["idx"] = idx
        item["image"] = image
        item["label_one_hot"] = torch.tensor(self.labels_one_hot[idx]).float()
        item["caption"] = self.encoded_captions[idx]

        return item

    def __len__(self):
        return self.num_samples

    # TODO: Add more augmentations
    def get_transforms(self, mode="train"):
        if mode == "train" and self.use_data_augmentation:
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            )

    def calculate_stats(self):
        loader = build_loaders(self.tokenizer, self.mode, self.image_size, 1, 4)

        psum = torch.tensor([0.0])
        psum_sq = torch.tensor([0.0])

        # loop through images
        for inputs in tqdm(loader):
            image = inputs["image"]

            psum += image.sum(axis=[0, 2, 3])
            psum_sq += (image**2).sum(axis=[0, 2, 3])

        # pixel count
        count = len(self.pairs) * self.image_size * self.image_size

        # mean and STD
        total_mean = psum / count
        total_var = (psum_sq / count) - (total_mean**2)
        total_std = torch.sqrt(total_var)

        # output
        print("Training data stats:")
        print("- mean: {:.4f}".format(total_mean.item()))
        print("- std:  {:.4f}".format(total_std.item()))


def build_loaders(
    mode,
    image_size,
    batch_size,
    num_workers,
    class_caption,
    use_dataset_train_captions,
    imbalance_factor,
    caption_max_length=77,
    use_sample_weights=False,
    use_data_augmentation=True,
    force_no_shuffle=False,
    class_weights_power=1,
    sample_weights_power=1,
    tokenizer=None,
):
    dataset = CIFAR100LT(
        mode=mode,
        image_size=image_size,
        class_caption=class_caption,
        use_dataset_train_captions=use_dataset_train_captions,
        imbalance_factor=imbalance_factor,
        caption_max_length=caption_max_length,
        use_data_augmentation=use_data_augmentation,
        class_weights_power=class_weights_power,
        sample_weights_power=sample_weights_power,
    )

    sampler = None
    if use_sample_weights:
        sampler = WeightedRandomSampler(
            dataset.sample_weights, len(dataset.sample_weights), replacement=True
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(
            True
            if mode == "train"
            and use_sample_weights == False
            and force_no_shuffle == False
            else False
        ),
        sampler=sampler,
    )
    return dataloader
