import numpy as np
from sklearn.metrics import f1_score
import torch.nn as nn
from torchvision.models import resnet18

import matplotlib.pyplot as plt
import PIL
import torchvision.transforms as transforms
import pathlib
import torchvision.utils
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.datapipes.iter import batch
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import PIL.Image
import torchvision.transforms.functional as TF
from ignite.engine import Events
from ignite.engine import create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Recall, Precision
from ignite.metrics import Loss
from torch import mode, optim, save

# se-resnet
# import se_resnet

device = 'cpu'
SEED_NUMBER = 7777
LABEL_MAP = {
    0: "Nuclepolasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Peroxisomes",
    9: "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Action filaments",
    13: "Focal adhesion ends",
    14: "Microtubules",
    15: "Microtubules ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membranc",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings"
}


class AdjustedGamma(object):
    def __call__(self, img):
        return TF.adjust_gamma(img, 0.8, gain=1)


image_transform = transforms.Compose([
    AdjustedGamma(),
    transforms.ToTensor()
])


class MulitiBandMultiLabelDataset(Dataset):
    BAND_NAMES = ["_red.png", "_green.png", "_blue.png", "_yellow.png"]

    def __init__(self, images_df, base_path, image_transform=image_transform, augmentation=None):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)

        self.images_df = images_df.copy()
        self.image_transform = image_transform
        self.augmentation = augmentation
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / x)
        self.mlb = MultiLabelBinarizer(classes=list(LABEL_MAP.keys()))

    def __getitem__(self, index):
        x = self._load_multiband_image(index)
        y = self._load_multiband_target(index)

        if self.augmentation is not None:
            x = self.augmentation(x)

        x = self.image_transform(x)

        return x, y

    def __len__(self):
        return len(self.images_df)

    def _load_multiband_image(self, index):
        row = self.images_df.iloc[index]
        image_bands = []
        for band_name in self.BAND_NAMES:
            p = str(row.Id.absolute()) + band_name
            pil_channel = PIL.Image.open(p)
            image_bands.append(pil_channel)

        # band3image = pil_channel.convert("RGB")
        band4image = PIL.Image.merge("RGBA", bands=image_bands)
        return band4image

    def _load_multiband_target(self, index):
        return list(map(int, self.images_df.iloc[index].Target.split(' ')))

    def collate_func(self, batch):
        images = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        labels_one_hot = self.mlb.fit_transform(labels)

        return torch.stack(images), torch.FloatTensor(labels_one_hot)


df = pd.read_csv("./data/train.csv")
df_train, df_val = train_test_split(df, test_size=.2, random_state=SEED_NUMBER)
train_data = MulitiBandMultiLabelDataset(
    df_train, base_path="./data/train", image_transform=image_transform)
test_data = MulitiBandMultiLabelDataset(
    df_val, base_path="./data/train", image_transform=image_transform)
train_load = DataLoader(
    train_data, collate_fn=train_data.collate_func, batch_size=16, num_workers=6)
test_load = DataLoader(
    test_data, collate_fn=test_data.collate_func, batch_size=16, num_workers=6)


def get_model(n_classes, image_channels=4):
    # resnet 18
    model = resnet18(pretrained=True)
    for p in model.parameters():
        p.requires_grad = True
    inft = model.fc.in_features
    """self.fc = nn.Linear(512 * block.expansion, num_classes)"""
    model.fc = nn.Linear(in_features=inft, out_features=n_classes)
    model.conv1 = nn.Conv2d(
        image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model
    # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
    # padding=3, bias=False)


def train(trainer, train_loader, test_loader, checkpoint_path="bestmodel_{}_{}.torch", epochs=1):
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 16
        if iter % 10 == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss : {:.2f}".format(
                engine.state.epoch, iter, len(
                    train_loader), engine.state.output
            ))

    def log_training_results(engine):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        avg_null = metrics["loss"]
        print("Training results -> Epoch : {} Avg loss : {:.2f}".format(
            engine.state.epoch, avg_null
        ))
        save(model, checkpoint_path.format(engine.state.epoch, avg_null))

    trainer.run(train_loader, max_epochs=epochs)


"""평가 코드"""


def evaluate(model, test_loader, threshold=0.2):
    all_preds = []
    true = []
    model.eval()

    for b in test_loader:
        image, target = b

        if torch.cuda.is_available():
            image, target = image.cuda(), target.cuda()
        else:
            image, target = image.to(device), target.to(device)

        pred = model(image)
        all_preds.append(pred.simoid().cpu().data.numpy())
        true.append(target.cpu().data.numpy())
    P = np.concatenate(all_preds)
    R = np.concatenate(true)

    f1 = f1_score(P > threshold, R, average='macro')

    return f1


"""Prepare model"""
model = get_model(28, 4)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

evaluator = create_supervised_evaluator(
    model,
    device=device,
    metrics={'loss': Loss(criterion)}
)

optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                       model.parameters()), lr=0.00025)

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

if __name__ == '__main__':
    # train
    # model = train(trainer, train_load, test_load, epochs=1)
    train(trainer, train_load, test_load, epochs=1)

    # eval
    res = evaluate(model, test_load, threshold=0.2)
    print("Eval F1 >> ", res)

# # for i in range(1):
# #     sample, _ = train_data[0]

# # plt.figure(figsize=(10, 10))
# # plt.imshow(transforms.ToPILImage()(sample))
# # plt.show()


# def get_model(n_classes, image_channels=4):
#     model = resnet18(pretrained=False)
#     for p in model.parameters():
#         p.requires_grad = True
#     inft = model.fc.in_features
#     model.fc = nn.Linear(in_features=inft, out_features=n_classes)
#     model.avgpool = nn.AdaptiveAvgPool2d(1)
#     model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3,
#                             bias=False)

#     return model


# def get_se_resnet_model(n_classes, image_channels=4):
#     model = se_resnet.se_resnet50()
#     for p in model.parameters():
#         p.requires_grad = True
#     inft = model.fc.in_features
#     model.fc = nn.Linear(in_features=inft, out_features=n_classes)
#     model.avgpool = nn.AdaptiveAvgPool2d(1)
#     model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3,
#                             bias=False)

#     return model


# def train(trainer, train_loader, test_loader, checkpoint_path='bestmodel_{}_{}.torch', epochs=1):
#     @trainer.on(Events.ITERATION_COMPLETED)
#     def log_training_loss(engine):
#         iter = (engine.state.iteration - 1) % len(train_loader) + 1
# #         ctx.channel_send('loss', engine.state.output)
#         if iter % 10 == 0:
#             print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
#                   "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))

#     @trainer.on(Events.EPOCH_COMPLETED)
#     def log_training_results(engine):
#         evaluator.run(test_loader)
#         metrics = evaluator.state.metrics
#         avg_nll = metrics['loss']
#         print("Training Results - Epoch: {}  Avg loss: {:.2f}"
#               .format(engine.state.epoch, avg_nll))
#         save(model, checkpoint_path.format(engine.state.epoch, avg_nll))
#     trainer.run(train_loader, max_epochs=epochs)

#     return model


# # """평가"""
# # # Eval
# def evaluate(model, test_loader, threshold=0.2):
#     all_preds = []
#     true = []
#     model.eval()
#     for b in test_loader:
#         X, y = b
#         if torch.cuda.is_available():
#             X, y = X.cuda(), y.cuda()

#         X, y = X.to(device), y.to(device)
#         pred = model(X)
#         all_preds.append(pred.sigmoid().cpu().data.numpy())
#         true.append(y.cpu().data.numpy())

#     P = np.concatenate(all_preds)
#     R = np.concatenate(true)

#     f1 = f1_score(P > threshold, R, average='macro')
#     print(f1)
#     return f1


# # # Prepare model
# se_resnet_model = get_se_resnet_model(28, 4)
# se_resnet_model = se_resnet_model.to(device)
# model = get_model(28, 4)
# model = model.to(device)
# criterion = nn.BCEWithLogitsLoss()
# if torch.cuda.is_available():
#     criterion = criterion.cuda()

# criterion = criterion.to(device)

# evaluator = create_supervised_evaluator(se_resnet_model,
#                                         device=device,
#                                         metrics={'loss': Loss(criterion)
#                                                  })
# optimizer = optim.Adam(filter(lambda p: p.requires_grad,
#                        se_resnet_model.parameters()), lr=0.0005)
# trainer = create_supervised_trainer(
#     se_resnet_model, optimizer, criterion, device=device)

# if __name__ == '__main__':

#     model = train(trainer, train_load, test_load, epochs=1)
# #     # res = evaluate(model, test_load, threshold=0.2)
