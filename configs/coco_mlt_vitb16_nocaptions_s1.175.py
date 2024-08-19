import torch


class CFG:
    # Dataset options: "coco_mlt", "voc_mlt"
    dataset = "coco_mlt"
    # Is used for the label embeddings to predict classes. Also used as image captions if use_dataset_train_captions is False
    class_caption = "a photo of a "
    # Determines if the dataset natural language captions should be used during training
    use_dataset_train_captions = False

    # Controls which checkpoints are saved during training each epoch
    save_newest_checkpoint = False
    save_best_mAP_checkpoint = False
    save_best_tail_mAP_checkpoint = False

    # Controls image input size to the image encoder
    size = 224
    # Use device "cuda" by default, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Determines which pre-trained model to use for the image encoder: "RN50" or "ViT-B/16"
    model_name = "ViT-B/16"
    # Sets the max token length for the text encoder, CLIP uses 77
    max_length = 77
    # Batch size for training and validation
    batch_size = 8
    # Number of workers for the dataloaders
    num_workers = 4
    # Number of epochs to train for, important for learning rate scheduling
    epochs = 50

    # Determines which loss functions are used during training: A combination of "clip", "siglip", "bal"
    loss_function = [
        "clip",
        "bal",
    ]
    # Sets the λ factor for balancing ASL and CLIP loss functions
    asl_mul = 5

    # ASL loss function parameters
    asl_gamma_neg = 9.8
    asl_gamma_pos = 0.0
    asl_clip = 0.05
    asl_eps = 1e-8

    # Configures oversampling of tail classes, s parameter in the paper, sample_weights_power and class_weights_power are set to the same value for simplicity
    use_sample_weights = True
    sample_weights_power = 1.175

    # Configures class weights for the loss function
    use_weighted_loss = True
    class_weights_power = 1.175

    # Label smoothing for ASL loss
    label_smoothing = 0.1

    # Determines which optimizer to use: "AdamW", "Adam", or "SGD"
    optimizer = "AdamW"
    # Base learning rate for the optimizer
    lr = 1e-6
    # Momentum for the SGD optimizer
    momentum = 0.9
    # Weight decay for the optimizer
    weight_decay = 0.01
