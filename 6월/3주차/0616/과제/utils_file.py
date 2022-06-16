import torchvision.transforms as transforms
import torch
import os

import configer
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# 1. aug 2. train loop 3.val loop 4. save model 5. eval
# data augmentation 함수
def data_augmentation():
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ])
    }
    return data_transform


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    history = []
    train_loss = 0
    valid_loss = 0
    train_acc = 0
    valid_acc = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(configer.device)
                labels = labels.to(configer.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc
            else:
                valid_loss = epoch_loss
                valid_acc = epoch_acc

        history.append([train_loss, valid_loss, train_acc, valid_acc])
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])


def loss_acc_visualize(history, optim):
    plt.figure(figsize=(20, 10))

    plt.suptitle(str(optim))

    plt.subplot(121)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['valid_loss'], label='valid_loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(122)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['valid_acc'], label='valid_acc')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.show()
    # plt.savefig(str(path) + 'loss_acc.png')


def visual_predict(model, data):
    c = np.random.randint(0, len(data))
    img, label = data[c]

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        # out = model(img.view(1, 3, 224, 224).cuda())
        out = model(img.view(1, 3, 224, 224).cpu())
        out = torch.exp(out)
        print(out)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.title(data.labels[label])
    plt.subplot(122)
    plt.barh(data.labels, out.cpu().numpy()[0])
    plt.show()


def class_accuracies(model, data):
    accuracy_dict = {}
    with torch.no_grad():
        model.eval()
        for c in data.valid_dict.keys():
            correct_count = 0
            total_count = len(data.valid_dict[str(c)])
            gt = data.labels.index(str(c))
            for path in data.valid_dict[str(c)]:
                # print(path)
                im = Image.open(path).convert('RGB')
                # im.show()
                im = transforms.ToTensor()(im)
                im = transforms.Resize((224, 224))(im)
                # out = model(im.view(1, 3, 224, 224).cuda())
                out = model(im.view(1, 3, 224, 224).cpu())
                # print(out)
                out = torch.exp(out)
                pred = list(out.cpu().numpy()[0])
                # print(pred)
                pred = pred.index(max(pred))
                # print(pred,gt)

                if gt == pred:
                    correct_count += 1
            print(f"Accuracy for class {str(c)} : ",
                  correct_count / total_count)
            accuracy_dict[str(c)] = correct_count / total_count

    plt.figure(figsize=(10, 5))
    plt.title('class_accuracies')
    plt.barh(list(accuracy_dict.keys()), list(accuracy_dict.values()))
    plt.show()

    return accuracy_dict
