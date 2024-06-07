import torch
import torchvision.transforms as transforms
import os
import pickle
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
import numpy as np
import clip


class COCOMLTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        mode,
        image_size,
        class_caption,
        use_dataset_train_captions,
        caption_max_length=77,
        class_weights_power=1,
        sample_weights_power=1,
        use_data_augmentation=False,
    ):
        self.root = root
        self.use_data_augmentation = use_data_augmentation

        # Load class splits
        with open(
            os.path.join(self.root, "longtail2017/class_split.pkl"), "rb"
        ) as file:
            self.class_split = pickle.load(file)
            self.head_classes = list(self.class_split["head"])
            self.middle_classes = list(self.class_split["middle"])
            self.tail_classes = list(self.class_split["tail"])
            print(self.class_split)

        # Load class splits
        # with open(os.path.join(self.root, "longtail2017class_split.pkl"), "rb") as file:
        #     self.class_split = pickle.load(file).
        #     print(self.class_split)

        self.classes = []
        with open(os.path.join(self.root, "longtail2017/coco_labels.txt"), "r") as file:
            # Read lines as string array
            classes_raw = file.readlines()
            classes_raw = list(map(lambda x: str.removesuffix(x, "\n"), classes_raw))
            # person 0
            # Create id:name dictionary
            for class_raw in classes_raw:
                class_raw = class_raw.strip()
                class_name, class_id = class_raw.rsplit(" ", 1)
                self.classes.append({"id": int(class_id), "name": class_name})

        self.image_paths = []
        self.labels = []
        self.labels_one_hot = []

        if mode == "train":
            with open(
                os.path.join(self.root, "longtail2017/coco_lt_captions.txt"),
                "r",
                encoding="utf-8",
            ) as file:
                lines = file.readlines()
                # VOCdevkit/VOC2012/JPEGImages/2008_000023.jpg a bottle of beer and a candle on a table with pizza and a television.
                # Create id:caption dictionary
                self.captions = []
            for line in lines:
                caption = line.strip()
                caption = caption.split(" ", 1)[1]
                self.captions.append(caption)

            with open(
                os.path.join(self.root, "longtail2017/coco_lt_train.txt"), "r"
            ) as file:
                # Read lines as string array
                train_images = file.readlines()
                train_images = list(
                    map(lambda x: str.removesuffix(x, "\n"), train_images)
                )
                # train2017/000000554625.jpg 0 62 64 66

                for train_image in train_images:
                    image_path, labels = train_image.split(" ", 1)
                    labels = labels.strip()
                    labels = list(map(int, labels.split(" ")))
                    label_one_hot = np.zeros(len(self.classes))
                    for label in labels:
                        label_one_hot[label] = 1
                    self.image_paths.append(image_path)
                    self.labels.append(labels)
                    self.labels_one_hot.append(label_one_hot)
        elif mode == "valid":
            with open(
                os.path.join(self.root, "longtail2017/coco_lt_test.txt"), "r"
            ) as file:
                # Read lines as string array
                val_images = file.readlines()
                val_images = list(map(lambda x: str.removesuffix(x, "\n"), val_images))
                # val2017/000000397133.jpg 0 39 41 43 44 45 50 51 60 69 71

                for val_image in val_images:
                    split = val_image.split(" ", 1)
                    if len(split) == 1:
                        image_path = split[0]
                        labels = []
                    else:
                        image_path, labels = split
                        labels = labels.strip()
                        labels = set(map(int, labels.split(" ")))

                    label_one_hot = np.zeros(len(self.classes))
                    for label in labels:
                        label_one_hot[label] = 1
                    self.image_paths.append(image_path)
                    self.labels.append(labels)
                    self.labels_one_hot.append(label_one_hot)
        else:
            raise ValueError("Invalid mode.")

        if not use_dataset_train_captions or mode == "valid":
            self.captions = []
            for labels in self.labels:
                caption = class_caption + ", ".join(
                    list(map(lambda x: self.classes[x]["name"], labels))
                )
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
            map(lambda x: self.class_caption + x["name"], self.classes)
        )  # a photo of a _
        self.num_classes = len(self.classes)

        # Calculate class weights for nn.CrossEntropyLoss based on the distribution of classes in the dataset
        self.class_weights = torch.zeros(self.num_classes, dtype=float)
        self.sample_weights = torch.zeros(len(self.image_paths), dtype=float)

        index = 0
        for this_class in self.classes:
            number_of_occurences = 0
            for i in range(len(self.image_paths)):
                if this_class["id"] in self.labels[i]:
                    number_of_occurences += 1

            self.class_weights[index] = len(self.image_paths) / number_of_occurences
            print(
                f"Class {this_class['name']} has {number_of_occurences} occurences, weight is {self.class_weights[index]}."
            )
            index += 1

        # Calculate sample weights based on class weights, more weight for samples from underrepresented classes
        for i in range(len(self.image_paths)):
            for category_id in self.labels[i]:
                # Get index of category_id in self.classes
                index = next(
                    (
                        index
                        for (index, d) in enumerate(self.classes)
                        if d["id"] == category_id
                    ),
                    None,
                )
                self.sample_weights[i] += self.class_weights[index]

        # Apply weighting exponent
        self.class_weights = self.class_weights**sample_weights_power
        self.sample_weights = self.sample_weights**sample_weights_power

        self.class_weights = self.class_weights / torch.max(self.class_weights)
        self.sample_weights = self.sample_weights / torch.max(self.sample_weights)

        number_of_unique_images = len(self.image_paths)
        print(
            f"Loaded {number_of_unique_images} caption-image pairs, {len(self.classes)} classes, head/middle/tail class ratio is {len(self.class_split['head'])}/{len(self.class_split['middle'])}/{len(self.class_split['tail'])}."
        )

    def __getitem__(self, idx):
        item = {}

        image_path = os.path.join(self.root, "images", self.image_paths[idx])
        image_path = os.path.normpath(image_path)
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = self.transforms(image)

        item["idx"] = idx
        item["image"] = image
        item["label_one_hot"] = torch.tensor(self.labels_one_hot[idx]).float()
        item["caption"] = self.encoded_captions[idx]

        for category_id in self.labels[idx]:
            # Get index of category_id in self.classes
            index = next(
                (
                    index
                    for (index, d) in enumerate(self.classes)
                    if d["id"] == category_id
                ),
                None,
            )
            item["label_one_hot"][index] = 1

        return item

    def __len__(self):
        return len(self.image_paths)

    # TODO: Add more augmentations
    def get_transforms(self, mode="train"):
        if mode == "train" and self.use_data_augmentation:
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
    root,
    mode,
    image_size,
    batch_size,
    num_workers,
    class_caption,
    use_dataset_train_captions,
    caption_max_length=77,
    use_sample_weights=False,
    use_data_augmentation=True,
    force_no_shuffle=False,
    class_weights_power=1,
    sample_weights_power=1,
    tokenizer=None,
):
    dataset = COCOMLTDataset(
        root=root,
        mode=mode,
        image_size=image_size,
        class_caption=class_caption,
        use_dataset_train_captions=use_dataset_train_captions,
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
