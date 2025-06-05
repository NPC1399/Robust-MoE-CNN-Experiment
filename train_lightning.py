import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from args import get_args_parser
from utils.general_utils import prepare_data, initialize_weights, split_data_and_move_to_device
import models
from utils.schedules import get_optimizer


class LightningClassifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        # prepare data to know num_classes and normalization
        train_loader, _, test_loader, normalize, num_classes, _ = prepare_data(args)
        self.train_loader = train_loader
        self.val_loader = test_loader
        self.model = models.__dict__[args.arch](num_classes=num_classes,
                                                n_expert=args.n_expert,
                                                ratio=args.ratio)
        initialize_weights(self.model)
        if args.normalize:
            self.model.normalize = normalize

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = split_data_and_move_to_device(batch, self.device)
        output = self(images)
        loss = F.cross_entropy(output, target)
        acc = (output.argmax(dim=1) == target).float().mean()
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = split_data_and_move_to_device(batch, self.device)
        output = self(images)
        loss = F.cross_entropy(output, target)
        acc = (output.argmax(dim=1) == target).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model, self.hparams)
        return optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def main():
    parser = get_args_parser()
    parser.set_defaults(
        arch="resnet50_imagenet_moe",
        dataset="ImageNet",
        batch_size=256,
        normalize=True,
    )
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus per node')
    parser.add_argument('--num-nodes', type=int, default=1, help='number of nodes for training')
    args = parser.parse_args()

    model = LightningClassifier(args)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus if torch.cuda.is_available() else None,
        num_nodes=args.num_nodes,
        strategy='ddp' if args.gpus > 1 or args.num_nodes > 1 else None,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
