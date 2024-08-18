import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
        pos_weight=None,
        num_labels=None,
        label_smoothing=0.0,
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.num_labels = num_labels

        if self.label_smoothing > 0:
            assert (
                self.num_labels is not None
            ), "num_labels must be specified for label smoothing"
            assert (
                self.label_smoothing < 1.0
            ), "label smoothing parameter must be less than 1"
            assert (
                self.label_smoothing > 0.0
            ), "label smoothing parameter must be bigger than 0"

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Label smoothing
        # y_ls = (1 - α) * y_hot + α / K
        if self.label_smoothing > 0 and self.num_labels is not None:
            self.targets = (
                1 - self.label_smoothing
            ) * self.targets + self.label_smoothing / self.num_labels

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = self.targets * torch.log(xs_pos.clamp(min=self.eps))

        # Weighting positive samples
        if self.pos_weight is not None:
            los_pos *= self.pos_weight

        los_neg = self.anti_targets * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * self.targets
            pt1 = xs_neg * self.anti_targets  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = (
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets
            )
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()


class BalancedAsymmetricLossGradientAnalysis(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
        pos_weight=None,
        num_labels=None,
        label_smoothing=0.0,
    ):
        super(BalancedAsymmetricLossGradientAnalysis, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.num_labels = num_labels

        if self.label_smoothing > 0:
            assert (
                self.num_labels is not None
            ), "num_labels must be specified for label smoothing"
            assert (
                self.label_smoothing < 1.0
            ), "label smoothing parameter must be less than 1"
            assert (
                self.label_smoothing > 0.0
            ), "label smoothing parameter must be bigger than 0"

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Label smoothing
        # y_ls = (1 - α) * y_hot + α / K
        if self.label_smoothing > 0 and self.num_labels is not None:
            self.targets = (
                1 - self.label_smoothing
            ) * self.targets + self.label_smoothing / self.num_labels

        # Normalize x to be between 0 and 1, so sigmoid does not return all 1 for values larger than 1
        x_norm = x / x.max()

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x_norm)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = self.targets * torch.log(xs_pos.clamp(min=self.eps))

        # Weighting positive samples
        if self.pos_weight is not None:
            los_pos *= self.pos_weight

        los_neg = self.anti_targets * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * self.targets
            pt1 = xs_neg * self.anti_targets  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma_pos = self.gamma_pos * self.targets
            one_sided_gamma_neg = self.gamma_neg * self.anti_targets
            one_sided_w_pos = torch.pow(1 - pt0, one_sided_gamma_pos)
            one_sided_w_nes = torch.pow(1 - pt1, one_sided_gamma_neg)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)

            los_pos *= one_sided_w_pos
            los_neg *= one_sided_w_nes

        # [batch_size, num_classes], [batch_size, num_classes]
        return -los_pos, -los_neg


class AsymmetricLossOptimized(nn.Module):
    """Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations"""

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
    ):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = (
            self.asymmetric_w
        ) = self.loss = None

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        # return -self.loss.sum()
        # # Changing it to mean
        return -self.loss.mean()


class BalancedAsymmetricLossOptimized(nn.Module):
    """Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations"""

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
        pos_weight=None,
        num_labels=None,
        label_smoothing=0.0,
        return_mean=True,
    ):
        super(BalancedAsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.num_labels = num_labels
        self.return_mean = return_mean

        if self.label_smoothing > 0:
            assert (
                self.num_labels is not None
            ), "num_labels must be specified for label smoothing"
            assert (
                self.label_smoothing < 1.0
            ), "label smoothing parameter must be less than 1"
            assert (
                self.label_smoothing > 0.0
            ), "label smoothing parameter must be bigger than 0"

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = (
            self.asymmetric_w
        ) = self.loss = None

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Label smoothing
        # y_ls = (1 - α) * y_hot + α / K
        if self.label_smoothing > 0 and self.num_labels is not None:
            self.targets = (
                1 - self.label_smoothing
            ) * self.targets + self.label_smoothing / self.num_labels

        # Normalize x to be between 0 and 1, so sigmoid does not return all 1 for values larger than 1
        x_norm = x / x.max()

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x_norm)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))

        # Weighting positive samples, self.pos_weight are class weights between 0 and 1
        if self.pos_weight is not None:
            self.loss *= self.pos_weight + 1

        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        if self.return_mean:
            return -self.loss.mean()
        else:
            return -self.loss


class ASLSingleLabel(nn.Module):
    """
    This loss is intended for single-label classification problems
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction="mean"):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        """
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1, target.long().unsqueeze(1), 1
        )

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(
                self.eps / num_classes
            )

        # loss calculation
        loss = -self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss
