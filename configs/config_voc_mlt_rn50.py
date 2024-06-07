import torch


class CFG:
    debug = False
    save_newest_checkpoint = True
    save_best_mAP_checkpoint = True
    save_best_tail_mAP_checkpoint = True

    dataset = "voc_mlt"

    optimizer = "AdamW"
    momentum = 0.9
    batch_size = 32
    num_workers = 4

    # COCO-LT 1e-6
    # VOC-LT 1e-6
    lr = 1e-6

    weight_decay = 0.01
    factor = 0.8
    epochs = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # model_name = 'vit_large_patch14_clip_224.openai'
    # image_embedding = 1024
    # text_encoder_model = 'bert-base-uncased'
    # text_embedding = 768
    # text_tokenizer = 'bert-base-uncased'
    # max_length = 250

    model_name = "RN50"  # RN50 or ViT-B/16
    max_length = 77

    class_caption = "a photo of a "
    use_dataset_train_captions = True

    loss_function = [
        "contrastive_original",
        "asl",
    ]  # ['contrastive', 'contrastive_original', 'siglip', 'bce', 'asl', 'contrastive_only_image_similarity']

    asl_mul = 6.0

    asl_gamma_neg = 4.0
    asl_gamma_pos = 0.0
    asl_clip = 0.05
    asl_eps = 1e-8

    siglip_logit_bias = 0

    # asl_gamma_neg = 4.0
    # asl_gamma_pos = 0.0
    # asl_clip = 0.05
    # asl_eps = 1e-8

    use_sample_weights = True
    sample_weights_power = 1.25

    use_weighted_loss = True
    class_weights_power = 1.25

    pretrained = True  # for both image encoder and text encoder
    trainable = True  # for both image encoder and text encoder
    temperature = 1.0

    prediction_threshold = 0.5
    label_smoothing = 0.1  # 0.1

    # image size
    size = 224
