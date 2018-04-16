import argparse
import torch
from models import MACNetwork
from datasets import CLEVRDataset, collate_fn, transform
from utils import to_var
from configs import settings
from tqdm import tqdm
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torch.autograd import Variable
from torch import nn
from torch.optim import Adam, lr_scheduler
import os


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()


def test(epoch=0, test=True):
    reset_meters()
    # Test
    loader = test_loader if test else val_loader
    print('Testing...')
    model.eval()
    for i, data in enumerate(loader):
        images, questions, lengths, answers = [to_var(x, volatile=True) for x in data]
        answers = answers.squeeze()

        out, cvs, rvs = model(images, questions, lengths)
        loss = criterion(out, answers)

        meter_accuracy.add(out.data, answers.data)
        confusion_meter.add(out.data, answers.data)
        meter_loss.add(loss.data[0])

    loss = meter_loss.value()[0]
    acc = meter_accuracy.value()[0]

    test_loss_logger.log(epoch, loss)
    test_accuracy_logger.log(epoch, acc)
    confusion_logger.log(confusion_meter.value())

    print("Epoch{} Test acc:{:4}, loss:{:4}".format(epoch, acc, loss))


def train(epoch):
    reset_meters()
    model.train()
    args.test_mode = False

    # Train
    print("Epoch {}".format(epoch))

    with tqdm(total=steps) as pbar:
        for i, data in enumerate(train_loader):
            images, questions, lengths, answers = [to_var(x) for x in data]

            answers = answers.squeeze()

            out, _, _ = model(images, questions, lengths)
            loss = criterion(out, answers)

            optimizer.zero_grad()

            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

            optimizer.step()

            meter_accuracy.add(out.data, answers.data)
            meter_loss.add(loss.data[0])
            pbar.set_postfix(loss=meter_loss.value()[0], acc=meter_accuracy.value()[0],
                             lr=optimizer.param_groups[0]['lr'])
            pbar.update()

        loss = meter_loss.value()[0]
        acc = meter_accuracy.value()[0]

        if epoch == 0:
            setting_logger.log(str(args))

        train_loss_logger.log(epoch, loss)
        train_error_logger.log(epoch, acc)

        print("\nEpoch{} Train acc:{:4}, loss:{:4}".format(epoch, acc, loss))
        torch.save(model.state_dict(), weight_folder + "/model_{}.pth".format(epoch))
        scheduler.step(loss)  # adjust learning rate if no improvement in train set loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CapsNet')

    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-num_epochs', type=int, default=30)
    parser.add_argument('-d', type=int, default=512, help='Hidden dim')
    parser.add_argument('-p', type=int, default=12, help='Num of MAC Cell')
    parser.add_argument('--dataset', type=str, default='CLEVR', metavar='N',
                        help='name of dataset: CLEVR or CLEVR_Human')
    parser.add_argument('--exp_name', type=str, default='Try MAC network',
                        metavar='N', help='Experiment name for storing result')
    parser.add_argument('-gpu', type=int, default=0, help="which gpu to use")
    parser.add_argument('--patience', type=int, default=5, help='Patience for scheduler')

    args = parser.parse_args()

    setting = settings[args.dataset]
    args.vocab_size = setting['vocab_size']
    args.embed_size = setting['embed_size']
    args.num_classes = setting['num_classes']
    args.env_name = '{} {}'.format(args.dataset, args.exp_name)

    print('Training on {}'.format(args.dataset))

    use_cuda = torch.cuda.is_available()

    train_dataset = CLEVRDataset('data/CLEVR_v1.0/', download=False, mode='train', transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=4,
                                               collate_fn=collate_fn,
                                               shuffle=True)

    val_dataset = CLEVRDataset('data/CLEVR_v1.0/', download=False, mode='val', transform=transform)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=4,
                                             collate_fn=collate_fn,
                                             shuffle=False)

    test_dataset = CLEVRDataset('data/CLEVR_v1.0/', download=False, mode='test', transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=4,
                                              collate_fn=collate_fn,
                                              shuffle=False)

    steps = len(train_dataset) // args.batch_size
    weight_folder = 'weights/{}'.format(args.env_name.replace(' ', '_'))
    if not os.path.isdir(weight_folder):
        os.mkdir(weight_folder)

    model = MACNetwork(args)
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience)

    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(setting['num_classes'], normalized=True)

    setting_logger = VisdomLogger('text', opts={'title': 'Settings'}, env=args.env_name)
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'}, env=args.env_name)
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'}, env=args.env_name)
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'}, env=args.env_name)
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'}, env=args.env_name)
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(setting['num_classes'])),
                                                     'rownames': list(range(setting['num_classes']))},
                                    env=args.env_name)

    with torch.cuda.device(args.gpu):
        if use_cuda:
            print("activating cuda")
            model.cuda()

        for epoch in range(args.num_epochs):
            train(epoch)
            test(epoch, False)

            if optimizer.param_groups[0]['lr'] <= 1e-7:
                exit()
