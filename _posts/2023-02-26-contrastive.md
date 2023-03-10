# Contrastive Loss on MNIST Dataset

We are going to take different way of learning. Here I'm presenting question
answer based mathod in which I'll show code first and ask questions around it. This is
similar to real word learning where you do something and keep asking questions
to yourself untill you understand it fully.

Here's the table of contents:

1. TOC
{:toc}

## What is Contrastive Loss?

Contrastive loss is used in classification task. It says that in embedding
space same class vectors should be near and different class vectors should be
appart. It is reagraded as alternative to the cross entropy loss.

The contrastive loss function computes the distance between pairs of examples in the embedding space, and then applies a penalty if the distance is greater than a certain threshold. Specifically, for a given pair of examples, the contrastive loss function calculates the Euclidean distance between their embedding vectors. If the distance is below a certain threshold, a small loss is incurred; if it is above the threshold, a larger loss is incurred. The idea is to encourage the model to learn embeddings that make similar examples closer together and dissimilar examples further apart.

![](/images/contrastive_mnist_example.png "Contrastive MNIST Example")

## Why MNIST Data?

MNIST is a large collection of handwritten digits images. It has 10 class
labeling i.e. zero to nine. Images are black and white with 28 * 28 fixed
size. Simplisity of this dataset makes its ideal for learners to do image
classification.

![](/images/mnist.webp "MNIST Example")

## Basic Setup

We won't be using many pre built functions. As we are learners we'll
build our own model and dataset class. We'll be using pytorch. Also since
pytorch doesn't have contrastive loss function we'll build it by ourself. lets
start with basic imports

```python
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
import torchvision
from tqdm.notebook import tqdm
```

We'll talk about annoy in later part. We are using torchvision to perform
images tasks like transforms and resize. plt is used for plotting and tqdm for
showing progress bar.

## Creating Dataset

We'll extend torch class Dataset to create Mnist dataset class. There are 3
important functions ot the class.

1. `__init__` method: It is intializer of the class. We can pass pass toggles
and data to the class and change beavior on it. We'll use this method to do
all the post processing on the data.
1. `__len__` method: This return the len of the dataset.
1. `__getitem__` method: This return `ith` item from dataset. This method can
be used to do runtime data creation. But here we'll keep data prepared to save
time on training.

Lets jump on the code for Dataset. Just read it once and create questions. we'll
get answers one by one


### Code for creating dataset object for train and validation

We have a dataset class MNISTDataset and we'll use it to create train_dataset
and val_dataset. Given that our input data is randomly arranged we'll split
dataframe into two with 1000 into validation set and rest in train. We are
using image transform to convert data into `PIL Image -> Torch Tensors -> Normalized
data`.

```
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
val_count = 1000
default_transform = transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
dataset = MNISTDataset(data.iloc[:-val_count], default_transform)
val_dataset = MNISTDataset(data.iloc[-val_count:], default_transform)
```

### Code for MNIST dataset class

The code defines a PyTorch dataset class called MNISTDataset that is used for
loading and processing MNIST data in the desaired format. The MNISTDataset class takes
a Pandas DataFrame containing MNIST data as input, along with an optional data
transformation function and a boolean flag is_test indicating whether the data
is being used for testing or training.

The MNISTDataset class randomly selects pairs of images from the input data
for training, with each pair consisting of a "positive" example (an image with
the same label as the current image) and a "negative" example (an image with a
different label than the current image). For each row, the dataset class
returns the two images, a label indicating whether they are positive or
negative examples (doing it randomly), and the label of the original image.
Second image is choosen randomly and wether the second image is of same class
or not is also random having 50-50 chances. This is ensure that the dataset
return 50% postive pairs and 50% negative pairs.

If is_test is True, the dataset class will not generate positive or negative
examples, and will instead return a single image with a label of -1.

The __len__ method returns the total number of samples in the dataset, and the
__getitem__ method returns a single sample from the dataset as a tuple of the
two images, the label indicating whether they are positive or negative
examples, and the label of the original image. The data is also transformed if
a transformation function is provided.

This implementation is used for Siamese Neural Networks that are trained with
Contrastive Loss to learn a metric space where similar examples are closer to
each other and dissimilar examples are further apart.;

```
class MNISTDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, transform=None, is_test=False):
        super(MNISTDataset, self).__init__()
        dataset = []
        labels_positive = {}
        labels_negative = {}
        if is_test == False:
            for i in list(data_df.label.unique()):
                labels_positive[i] = data_df[data_df.label == i].to_numpy()
            for i in list(data_df.label.unique()):
                labels_negative[i] = data_df[data_df.label != i].to_numpy()

        for i, row in tqdm(data_df.iterrows(), total=len(data_df)):
            data = row.to_numpy()
            if is_test:
                label = -1
                first = data.reshape(28, 28)
                second = -1
                dis = -1
            else:
                label = data[0]
                first = data[1:].reshape(28, 28)
                if np.random.randint(0, 2) == 0:
                    second = labels_positive[label][
                        np.random.randint(0, len(labels_positive[label]))
                    ]
                else:
                    second = labels_negative[label][
                        np.random.randint(0, len(labels_negative[label]))
                    ]
                dis = 1.0 if second[0] == label else 0.0
                second = second[1:].reshape(28, 28)

            if transform is not None:
                first = transform(first.astype(np.float32))
                if second is not -1:
                    second = transform(second.astype(np.float32))

            dataset.append((first, second, dis, label))
        self.dataset = dataset
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]

```

## DataLoaders

The code defines two PyTorch DataLoader objects: trainLoader and valLoader.
DataLoader is a PyTorch class that provides an efficient way to load data in
batches during the training or evaluation of a model.

The batch_size argument sets the number of samples in each batch of data. The
shuffle argument indicates whether to shuffle the data at the beginning of
each epoch. The pin_memory argument is set to True, which enables faster data
transfer to the GPU by allocating memory in pinned memory. The num_workers
argument specifies the number of subprocesses to use for loading the data.

Here is the code
```
trainLoader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    prefetch_factor=100
)
valLoader = DataLoader(val_dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    prefetch_factor=100
)
```

## Model

```
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Dropout(0.3)
        ) # d * 32 * 12 * 12
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Dropout(0.3)
        ) # d * 64 * 4 * 4
        self.linear1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x
```

## Our Loss Function

```
class ContrastiveLoss(nn.Module):
    def __init__(self, m=2):
        super(ContrastiveLoss, self).__init__()
        self.m = m
        self.similarity = nn.CosineSimilarity(dim=-1, eps=1e-7)

    def forward(self, first, second, distance):
        score = self.similarity(first, second)
        return nn.MSELoss()(score, distance)
```

## Training Model

```
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = ContrastiveLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)
```

```
lrs = []
losses = []
for epoch in range(50):
    epoch_loss = 0
    batches=0
    print('epoch -', epoch)
    lrs.append(optimizer.param_groups[0]['lr'])
    print('learning rate', lrs[-1])
    for first, second, dis, label in tqdm(trainLoader):
        batches+=1
        optimizer.zero_grad()
        first_out = net(first.to(device))
        second_out = net(second.to(device))
        dis = dis.to(torch.float32).to(device)
        loss = loss_function(first_out, second_out, dis)
        epoch_loss+=loss
        loss.backward()
        optimizer.step()
    losses.append(epoch_loss.cpu().detach().numpy()/batches)
    scheduler.step()
    print('epoch_loss', losses[-1])
```

## Infrencing
## Why consine similarity?
## why annoy?
## Outputs
## Conclusion

[^1]: This is the footnote.

