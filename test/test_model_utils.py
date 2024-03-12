import torch
import torchvision
from time import time
import numpy as np
import copy

#-------------------------------------------#
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#-------------------------------------------#
def train_one_batch(data, model, criterion, optimizer, device='cpu'):
    index, inputs, labels, time = data
    inputs = inputs.to(device)  # inputs = inputs.cuda()
    labels = labels.to(device)  # labels = labels.cuda()

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

    _, predicted = torch.max(outputs.data, 1)

    iter_size = labels.size(0)
    iter_correct = (predicted == labels).sum().item()
    iter_loss = loss.mean().item()
    return iter_size, iter_correct, iter_loss, loss, time.mean().item()
#-------------------------------------------#
def validate_model(model, test_loader, criterion, device):
    model.eval()
    total, correct, test_loss = 0, 0, 0
    with torch.no_grad():
        for data in test_loader:
            _, inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += criterion(outputs, labels).sum().item()
    
    return test_loss / total, correct / total
#-------------------------------------------#
default_transforms = torchvision.transforms.Compose(
                        [torchvision.transforms.Resize([224,224]), torchvision.transforms.ToTensor()])
#-------------------------------------------#
def _train_model(model, dataset, train_loader, test_loader, epochs=100, device='cpu',
                 criterion=torch.nn.CrossEntropyLoss(), optimizer=None,
                 model_name='alexnet', output_dir=None, criteria='random'):

    if output_dir is not None:
        log_f = open(output_dir+'/stdout.txt', 'w')

    #model = torch.nn.DataParallel(model)
    model = model.to(device)

    # train
    iter_count = len(train_loader)
    loss_per_epoch, acc_per_epoch, time_per_epoch, val_loss_per_epoch, val_acc_per_epoch, loading_per_epoch = [], [], [], [], [], []

    # For early stopping
    '''best_loss = 10 ** 9  # Initialize with a very large value
    patience_limit = 5      # Decide how many epochs to ignore performance degradation
    patience_check = 0      # Record how many epochs the performance has deteriorated so far'''

    for epoch in range(epochs):
        start_epoch = time()
        batch_count, partition = 0, 0
        total, correct, train_loss = 0, 0, 0
        epo_total, epo_correct, epo_loss = 0, 0, 0
        dataload_times = []

        for data in train_loader:
            model.train()

            iter_size, iter_correct, iter_loss, losses, iter_time = train_one_batch(data, model, criterion, optimizer, device)

            idx, sample, target, _ = data
            try:
                if criteria == 'random':
                    dataset.cache_batch(idx, sample, target, torch.ones(len(idx)))
                elif criteria == 'loss_sample':
                    dataset.cache_batch(idx, sample, target, losses)
            except AttributeError:
                pass

            dataload_times.append(iter_time)
            total, correct, train_loss = (total+iter_size), (correct+iter_correct), (train_loss+(iter_loss*iter_size))

            batch_count += 1
            partition += 1

            if (batch_count * 10) % round(iter_count, -1) == 0:
                current = time()
                if log_f:
                    log_f.write("\tBatch: {}/{} - Loss: {:.4f}, Acc: {:.4f}, current_time elapsed: {:.6f}   \n"\
                            .format(batch_count, iter_count, train_loss / total, correct / total, current - start_epoch))
                else:
                    print("\tBatch: {}/{} - Loss: {:.4f}, Acc: {:.4f}, current_time elapsed: {:.6f}   "\
                            .format(batch_count, iter_count, train_loss / total, correct / total, current - start_epoch))#, end='\r')
                epo_total, epo_correct, epo_loss = (epo_total+total), (epo_correct+correct), (epo_loss+train_loss)
                total, correct, train_loss, partition = 0, 0, 0, 0

        end_epoch = time()

        loss_per_epoch.append(epo_loss / epo_total)
        acc_per_epoch.append(epo_correct / epo_total)
        time_per_epoch.append(end_epoch - start_epoch)
        loading_per_epoch.append(np.mean(dataload_times))

        start_test = time()
        val_loss, val_acc = validate_model(model, test_loader, criterion, device)
        end_test = time()

        val_loss_per_epoch.append(val_loss)
        val_acc_per_epoch.append(val_acc)

        if log_f:
            log_f.write("Epoch: {}/{} - Loss: {:.4f}, Acc: {:.4f}, time elapsed: {:.6f}, val. Loss: {:.4f}, val. Acc: {:.4f}, time elapsed: {:.4f}, loading time: {:.6f}\n"\
                    .format(epoch+1, epochs, loss_per_epoch[-1], acc_per_epoch[-1], time_per_epoch[-1], val_loss, val_acc, end_test - start_test, np.mean(dataload_times)))
            log_f.flush()
        else:
            print("Epoch: {}/{} - Loss: {:.4f}, Acc: {:.4f}, time elapsed: {:.6f}, val. Loss: {:.4f}, val. Acc: {:.4f}, time elapsed: {:.4f}, loading time: {:.6f}"\
                    .format(epoch+1, epochs, loss_per_epoch[-1], acc_per_epoch[-1], time_per_epoch[-1], val_loss, val_acc, end_test - start_test, np.mean(dataload_times)))

        # Decide early stopping or not
        '''if val_loss > best_loss:
            patience_check += 1

            # early stopping
            if patience_check >= patience_limit:
                break
                """if log_f:
                    log_f.write("Early stopping warning: patience_check == {:d} exceeds patience_limit".format(patience_check))
                else:
                    print("Early stopping warning: patience_check == {:d} exceeds patience_limit".format(patience_check))"""
        else:
            best_loss = val_loss
            patience_check = 0
            torch.save(model.state_dict(), output_dir+"/model-train-"+str(model_name)+"-{}.ckpt".format(epoch))'''
        torch.save(model.state_dict(), output_dir+"/model-train-"+str(model_name)+"-{}.ckpt".format(epoch))

    log_f.close()

    return loss_per_epoch, acc_per_epoch, time_per_epoch, val_loss_per_epoch, val_acc_per_epoch, loading_per_epoch
