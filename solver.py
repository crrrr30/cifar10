import io
import os
import tqdm
import logging
import numpy as np

import torch


class OneLineHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write('\u001b[1000D' + msg)
            self.flush()
        except Exception:
            self.handleError(record)


class Solver:

    def __init__(self, model, optimizer, criterion, start_epoch, num_epochs, device, log_dir, checkpoint_interval):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs

        self.device = device

        self.log_dir = log_dir
        self.checkpoint_interval = checkpoint_interval

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        self.logger.addHandler(logging.StreamHandler())
        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s")) 
        self.logger.addHandler(file_handler)


    def train(self, train_dataloader, test_dataloader):
        
        for i in range(self.start_epoch, self.num_epochs):

            self.logger.info(f'Epoch {i}:')

            self.model.train()
            bar = tqdm.tqdm(train_dataloader)
            training_loss = []; correct = 0; total = 0

            for x, y in bar:
                
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()

                training_loss.append(loss.item())
                _, predicted = y_hat.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                bar.set_description('Loss: %.3f | Acc: %.3f%%' % (
                    np.mean(training_loss),
                    100. * correct / total
                ))


            if i % self.checkpoint_interval == 0:

                checkpoint = {
                    'epoch': i,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(self.log_dir, f'checkpoint{i:04d}.pkl'))

                self.model.eval()
                test_loss = []; correct = 0; total = 0
                
                with torch.no_grad():
                    for x, y in test_dataloader:
                    
                        x, y = x.to(self.device), y.to(self.device)
                        y_hat = self.model(x)
                        loss = self.criterion(y_hat, y)

                        test_loss.append(loss.item())
                        _, predicted = y_hat.max(1)
                        total += y.size(0)
                        correct += predicted.eq(y).sum().item()

                self.logger.info(f'Testing loss: {np.mean(test_loss):.3f} | Testing acc: {100 * correct / total:.3f}%')

        
        self.logger.info(f'Training finished...')
        

    def validate(self, test_dataloader):
        
        self.model.eval()
        test_loss = []; correct = 0; total = 0
        
        with torch.no_grad():     
            for x, y in test_dataloader:
            
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)

                test_loss.append(loss.item())
                _, predicted = y_hat.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

        self.logger.info(f'Testing loss: {np.mean(test_loss):.3f} | Testing acc: {100 * correct / total:.3f}%')
