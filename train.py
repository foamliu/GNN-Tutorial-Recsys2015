import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader

from config import device, grad_clip, print_freq, batch_size
from data_gen import YooChooseBinaryDataset
from models import Net
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_acc = 0
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model = ImgClsModel()
        model = Net()
        model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = nn.BCELoss()

    # Custom dataloaders
    dataset = YooChooseBinaryDataset(root='../')
    dataset = dataset.shuffle()
    train_dataset = dataset[:800000]
    val_dataset = dataset[800000:900000]
    test_dataset = dataset[900000:]
    len(train_dataset), len(val_dataset), len(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # Decay learning rate if there is no improvement for 10 consecutive epochs
        # if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
        #     adjust_learning_rate(optimizer, 0.1)

        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)

        writer.add_scalar('model/train_loss', train_loss, epoch)

        # One epoch's validation
        train_acc = evaluate(train_loader, model)
        val_acc = evaluate(val_loader, model)
        test_acc = evaluate(test_loader, model)
        print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.
              format(epoch, train_loss, train_acc, val_acc, test_acc))

        writer.add_scalar('model/train_acc', train_acc, epoch)
        writer.add_scalar('model/val_acc', val_acc, epoch)
        writer.add_scalar('model/test_acc', test_acc, epoch)

        # Check if there was an improvement
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_acc, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, data in enumerate(train_loader):
        # Move to GPU, if available
        data = data.to(device)
        label = data.y.to(device)

        # Forward prop.
        out = model(data)

        # Calculate loss
        loss = criterion(out, label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(epoch, i,
                                                                     len(train_loader),
                                                                     loss=losses,
                                                                     )
            logger.info(status)

    return losses.avg


def evaluate(loader, model):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    predictions = []
    labels = []

    # Batches
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    return roc_auc_score(labels, predictions)


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
