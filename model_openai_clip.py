import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from asl_loss import AsymmetricLossOptimized

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_right, recv_op_left]
    )
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (
            NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),
        )


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(
            left_rank, right_rank, tensor_to_left, tensor_to_right, group=group
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + NeighbourExchangeBidir.apply(
            ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs
        )


def neighbour_exchange_bidir_with_grad(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    return NeighbourExchangeBidir.apply(
        left_rank, right_rank, group, tensor_to_left, tensor_to_right
    )


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    assert (
        has_distributed
    ), "torch.distributed did not import correctly, please use a PyTorch version with support."
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0)
                )
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class ClipLossMultiLabel(nn.Module):

    def __init__(
        self,
        CFG,
        train_class_weights,
        valid_class_weights,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.train_class_weights = train_class_weights
        self.valid_class_weights = valid_class_weights

        self.asl_function = AsymmetricLossOptimized(
            gamma_neg=CFG.asl_gamma_neg,
            gamma_pos=CFG.asl_gamma_pos,
            clip=CFG.asl_clip,
            eps=CFG.asl_eps,
            num_labels=CFG.batch_size,
            label_smoothing=CFG.label_smoothing,
            return_mean=False,
        ).to(CFG.device)

        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, actual_labels) -> torch.Tensor:
        # Calculate the number of matching labels between samples
        # actual_labels should be a binary tensor of shape (batch_size, num_classes)
        matching_labels = actual_labels @ actual_labels.T

        # Calculate the total number of labels for each sample pair
        total_labels = (
            actual_labels.sum(dim=1, keepdim=True)
            + actual_labels.sum(dim=1)
            - matching_labels
        )

        # To avoid division by zero, set any zero totals to 1 (will result in zero matching percentage)
        total_labels = torch.where(
            total_labels == 0, torch.ones_like(total_labels), total_labels
        )

        # Calculate the percentage of matching labels between samples
        matching_percentage = matching_labels / total_labels

        return matching_percentage

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(
        self,
        image_features,
        text_features,
        labels_one_hot,
        logit_scale,
        mode,
        output_dict=False,
    ):
        device = image_features.device
        image_text_sim = logit_scale * image_features @ text_features.T
        text_image_sim = logit_scale * text_features @ image_features.T

        # Normalize the logits
        image_text_sim = F.log_softmax(image_text_sim, dim=1)
        text_image_sim = F.log_softmax(text_image_sim, dim=1)

        targets = self.get_ground_truth(labels_one_hot)

        # Compute the cross-entropy loss with the targets
        loss_image_text = F.cross_entropy(image_text_sim, targets, reduction="none")
        loss_text_image = F.cross_entropy(text_image_sim, targets, reduction="none")

        # loss_image_text = self.asl_function(image_text_sim, targets)
        # loss_text_image = self.asl_function(text_image_sim, targets)

        total_loss = (loss_image_text + loss_text_image) / 2

        # # Apply class weights
        # if mode == "train" and self.train_class_weights is not None:
        #     total_loss = total_loss * self.train_class_weights[labels]
        # elif mode == "valid" and self.valid_class_weights is not None:
        #     total_loss = total_loss * self.valid_class_weights[labels]

        total_loss = total_loss.mean()

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class SigLipLoss(nn.Module):
    """Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
        self,
        cache_labels=False,
        rank=0,
        world_size=1,
        bidir=True,
        use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(
        self, device, dtype, num_logits, negative_only=False
    ) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(
        self,
        image_features,
        text_features,
        logit_scale,
        logit_bias=None,
        negative_only=False,
    ):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(
        self, image_features, text_features, logit_scale, logit_bias, output_dict=False
    ):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss


# Loss function wrappers
class NoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        image_embeddings,
        text_embeddings,
        label_embeddings,
        dot_similarity,
        label_one_hot,
        temperature,
        mode,
    ):
        return 0.0


class OriginalCLIPLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = ClipLoss()

    def forward(
        self,
        image_embeddings,
        text_embeddings,
        label_embeddings,
        dot_similarity,
        label_one_hot,
        temperature,
        mode,
    ):
        return self.loss(image_embeddings, text_embeddings, temperature)


class CLIPLossMultiLabelWrapper(nn.Module):
    def __init__(self, CFG, train_class_weights, valid_class_weights):
        super().__init__()
        self.loss = ClipLossMultiLabel(CFG, train_class_weights, valid_class_weights)

    def forward(
        self,
        image_embeddings,
        text_embeddings,
        label_embeddings,
        dot_similarity,
        label_one_hot,
        temperature,
        mode,
    ):
        return self.loss(
            image_embeddings, text_embeddings, label_one_hot, temperature, mode
        )


class SigLipLossWrapper(nn.Module):
    def __init__(self, logit_bias):
        super().__init__()
        self.loss = SigLipLoss()
        self.logit_bias = logit_bias

    def forward(
        self,
        image_embeddings,
        text_embeddings,
        label_embeddings,
        dot_similarity,
        label_one_hot,
        temperature,
        mode,
    ):
        return self.loss(
            image_embeddings, text_embeddings, temperature, logit_bias=self.logit_bias
        )


class ASLLossWrapper(nn.Module):
    def __init__(self, asl_function_train, asl_function_valid, asl_mul):
        super().__init__()

        self.asl_function_train = asl_function_train
        self.asl_function_valid = asl_function_valid
        self.asl_mul = asl_mul

    def forward(
        self,
        image_embeddings,
        text_embeddings,
        label_embeddings,
        dot_similarity,
        label_one_hot,
        temperature,
        mode,
    ):
        loss = (
            self.asl_function_train(dot_similarity, label_one_hot)
            if mode == "train"
            else self.asl_function_valid(dot_similarity, label_one_hot)
        )
        return loss * self.asl_mul


class OpenAICLIPModel(nn.Module):
    def __init__(
        self,
        config,
        clip_model,
        train_class_weights,
        valid_class_weights,
        asl_function_train=None,
        asl_function_valid=None,
    ):
        super().__init__()
        self.model = clip_model

        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        self.loss_functions = []
        for loss_function in config.loss_function:
            if loss_function == "clip":
                self.loss_functions.append(OriginalCLIPLossWrapper())
            elif loss_function == "clip_multilabel":
                self.loss_functions.append(
                    CLIPLossMultiLabelWrapper(
                        CFG=config,
                        train_class_weights=train_class_weights,
                        valid_class_weights=valid_class_weights,
                    )
                )
            elif loss_function == "siglip":
                self.loss_functions.append(
                    SigLipLossWrapper(logit_bias=config.siglip_logit_bias)
                )
            elif loss_function == "asl":
                if asl_function_train is None or asl_function_valid is None:
                    raise ValueError(
                        "ASL loss function requires asl_function_train and asl_function_valid to be provided"
                    )

                self.loss_functions.append(
                    ASLLossWrapper(
                        asl_function_train, asl_function_valid, asl_mul=config.asl_mul
                    )
                )
            else:
                raise ValueError(f"Unknown loss function: {loss_function}")

    def forward(self, batch, encoded_labels, mode):
        image_embeddings = self.model.encode_image(batch["image"])
        text_embeddings = self.model.encode_text(batch["caption"])
        label_embeddings = self.model.encode_text(encoded_labels)

        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        label_embeddings = label_embeddings / label_embeddings.norm(
            dim=-1, keepdim=True
        )

        label_one_hot = batch["label_one_hot"]

        logit_scale = self.model.logit_scale.exp()
        dot_similarity = logit_scale * image_embeddings @ label_embeddings.T

        loss = 0
        for loss_function in self.loss_functions:
            loss = loss + loss_function(
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                label_embeddings=label_embeddings,
                dot_similarity=dot_similarity,
                label_one_hot=label_one_hot,
                temperature=logit_scale,
                mode=mode,
            )

        return loss, dot_similarity
