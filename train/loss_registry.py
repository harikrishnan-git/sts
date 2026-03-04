from losses.prototypical_loss import prototypical_loss_acc
from losses.matchingnet_loss import matchingnet_loss_acc
from losses.triplet_loss import triplet_loss_acc
from losses.supcon_loss import supcon_loss_acc

LOSS_REGISTRY = {
    "proto": prototypical_loss_acc,
    "matching": matchingnet_loss_acc,
    "triplet": triplet_loss_acc,
    "supcon": supcon_loss_acc,
}
