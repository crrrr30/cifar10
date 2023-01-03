import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import os, glob, tqdm, argparse
import numpy as np

from data_loader import get_train_dataloader, get_test_dataloader
from model import Model
from efficientnet import EfficientNetB0
from solver import Solver
from optimizers import Ranger


def parse_args():

    parser = argparse.ArgumentParser(description = 'Train CIFAR-10')
    
    parser.add_argument('--num-epochs', help = 'Number of epochs to train the model for', default = 500, type = int)
    parser.add_argument('--batch-size', help = 'Batch size', default = 128, type = int)
    parser.add_argument('--log', help = 'Directory to save logs to', default = 'logs', type = str)
    parser.add_argument('--checkpoint-interval', help = 'Frequency of saving checkpoints', default = 5, type = str)
    parser.add_argument('--gpu', help = 'Specify which GPU to use (separate with commas). -1 means to use CPU', default = None, type = str)
    parser.add_argument('--resume', help = 'Directory of the checkpoint to resume from or the path to the checkpoint', default = None, type = str)
    
    return parser.parse_args()


if __name__ == '__main__':

    start_epoch = 0

    args = parse_args()
    
    if args.gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    print('==> Prepping data...')
    train_dataloader = get_train_dataloader(args.batch_size)
    test_dataloader = get_test_dataloader(args.batch_size)

    print('==> Building CNN...')
    model = Model()
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model.to(device)

    optimizer = Ranger(model.parameters())

    if args.resume:

        if os.path.isdir(args.resume):
            checkpoint = torch.load(
                sorted(glob.glob(os.path.join(args.resume, 'checkpoint*.pkl'))).pop(),
            map_location = lambda _, __: _)
            print(f'Resuming from epoch {checkpoint["epoch"] + 1}...')
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        elif os.path.isdir(args.resume):
            checkpoint = torch.load(
                sorted(glob.glob(os.path.join(args.log, args.resume, 'checkpoint*.pkl'))).pop(),
            map_location = lambda _, __: _)
            print(f'Resuming from epoch {checkpoint["epoch"] + 1}...')
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        else:
            raise FileNotFoundError(f'Checkpoint not found at {args.resume}!')  

    print(f'Number of total params: {sum([np.prod(p.shape) for p in model.parameters()])}')

    if start_epoch >= args.num_epochs: 
        print('The model has already been trained for the number of epochs required')

    if not os.path.exists(args.log):
        os.makedirs(args.log)
    index = 0 if len(os.listdir(args.log)) == 0 else int(sorted(os.listdir(args.log)).pop()[:4]) + 1
    args.log = os.path.join(args.log, '%.4d-train' % index)
    os.makedirs(args.log)
    print(f'==> Saving logs to {args.log}')

    solver = Solver(
        model = model, optimizer = optimizer, criterion = nn.CrossEntropyLoss(),
        start_epoch = start_epoch, num_epochs = args.num_epochs, device = device, 
        log_dir = args.log,  checkpoint_interval = args.checkpoint_interval,
    )
    
    print(f'==> Training for {args.num_epochs} epochs...')
    solver.train(train_dataloader, test_dataloader)