import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from asl_loss import BalancedAsymmetricLossOptimized

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
            elif loss_function == "asl" or loss_function == "bal":
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
        # Encode images, captions, and labels into the same embedding space
        image_embeddings = self.model.encode_image(batch["image"])
        text_embeddings = self.model.encode_text(batch["caption"])
        label_embeddings = self.model.encode_text(encoded_labels)

        # Normalize embeddings for cosine similarity
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        label_embeddings = label_embeddings / label_embeddings.norm(
            dim=-1, keepdim=True
        )

        label_one_hot = batch["label_one_hot"]

        # Calculate dot similarity between image and label embeddings
        logit_scale = self.model.logit_scale.exp()
        dot_similarity = logit_scale * image_embeddings @ label_embeddings.T

        # Apply loss functions
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
