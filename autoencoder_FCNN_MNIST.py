#  autoencoder_FCNN_MNIST.py
#     Author: T M Feroz Ali
#     Autoencoder using fully connected neural network (FCNN)
#     N/w architecture: 3 linear layer, all RelU non-linearity except last Sigmoid, Latent space dimn: 9 
#     Loss: BCE
#     Dataset: MNIST dataset
#     Analyze Latent space using PCA

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import random
from torchvision.utils import save_image
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.decomposition import PCA
import matplotlib.cm as cm

no_classes = 10
colors = cm.rainbow(np.linspace(0, 1, no_classes))

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
ip_dimn = 28*28
batch_size = 256

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = './Autoencoder_results_LatentPCA/9dimnLatentSpace/BCE_loss'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

train_dataset = torchvision.datasets.MNIST(root='../data/',
                                     train=True, 
                                     transform=transforms.ToTensor(),
                                     download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                                     train=False, 
                                     transform=transforms.ToTensor(),
                                     download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)

no_batches_train = len(train_loader)
no_batches_tst = len(test_loader)
print(f"No_batches train: {no_batches_train}")
print(f"No_batches test: {no_batches_tst}")

# Build a fully connected layer and forward pass
class AutoEncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.linear1 = nn.Linear(in_features=ip_dimn, out_features=14*14)
        self.linear2 = nn.Linear(in_features=14*14, out_features=7*7)
        self.linear3 = nn.Linear(in_features=7*7, out_features=3*3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        # Decoder
        self.linear4 = nn.Linear(in_features=3*3, out_features=7*7)
        self.linear5 = nn.Linear(in_features=7*7, out_features=14*14)
        self.linear6 = nn.Linear(in_features=14*14, out_features=ip_dimn)
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        e = self.relu3(self.linear3(x))
        x = self.relu4(self.linear4(e))
        x = self.relu5(self.linear5(x))
        x = self.sigmoid(self.linear6(x))
        return e, x

# Build model.
model = AutoEncoderNet().to(device)

# Build optimizer.
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Build loss.
# criterion = nn.MSELoss()
criterion = nn.BCELoss()

no_epochs = 150
first_pass = True
epoch_all = []
loss_test_all = []
loss_train_all = []

curr_lr = learning_rate
for epoch in range(no_epochs):

  # Training
  batch_idx = 0
  total_loss_train = 0

  for batch_idx, (images, labels) in enumerate(train_loader):

    images = images.reshape(-1, 28*28)
    images = images.to(device)
    # labels = labels.to(device)

    # Forward pass.
    latent_encoding, x_reconst = model(images)

    # breakpoint()

    # Compute loss.
    loss = criterion(x_reconst, images)
    # loss = F.binary_cross_entropy(x_reconst, images, size_average=False)

    if epoch == 0 and first_pass == True:
      print(f'Initial {epoch} loss: ', loss.item())
      first_pass = False

    # Compute gradients.
    optimizer.zero_grad()
    loss.backward()

    # 1-step gradient descent.
    optimizer.step()

    # calculating train loss
    total_loss_train += loss.item()

    if epoch == 0 and (batch_idx+1) % 10 == 0:
      print(f"Train Batch:{batch_idx}/{no_batches_train}, loss: {loss}, total_loss: {total_loss_train}")

    # Accumulate data for PCA
    if batch_idx == 0:
       X_train = latent_encoding.detach().cpu().numpy()
       X_labels = labels.detach().cpu().numpy()
    else:
       X_train = np.concatenate((X_train, latent_encoding.detach().cpu().numpy()), axis=0)
       X_labels = np.concatenate((X_labels, labels.detach().cpu().numpy()), axis=0)

  # Decay learning rate
  if (epoch+1) % 50 == 0:
      curr_lr /= 10
      update_lr(optimizer, curr_lr)

  print(f'Train Epoch:{epoch}, Average Train loss:{total_loss_train/no_batches_train}' )

  if (epoch+1) % 10 == 0:
    x_concat = torch.cat([images.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
    save_image(x_concat, os.path.join(sample_dir, 'train_reconst-{}.png'.format(epoch+1)))


  # Testing after each epoch
  model.eval()
  with torch.no_grad():

    total_loss_test = 0

    for images, labels in test_loader:

      images = images.reshape(-1, 28*28)
      images = images.to(device)
      labels = labels.to(device)

      # Forward pass.
      _, x_reconst = model(images)

      # Compute test loss.
      loss = criterion(x_reconst, images)
      # loss = F.binary_cross_entropy(x_reconst, images, size_average=False)
      
      total_loss_test += loss.item()

    print(f'Test Epoch:{epoch}, Average Test loss: {total_loss_test/no_batches_tst}')

    if (epoch+1) % 10 == 0:
      x_concat = torch.cat([images.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
      save_image(x_concat, os.path.join(sample_dir, 'test_reconst-{}.png'.format(epoch+1)))


  # PLotting train and test curves
  # breakpoint()
  epoch_all.append(epoch)
  loss_test_all.append(total_loss_test/no_batches_tst)
  loss_train_all.append(total_loss_train/no_batches_train)

  plt.clf()
  plt.plot(epoch_all, loss_train_all, marker = 'o', mec = 'g', label='Average Train loss')
  plt.plot(epoch_all, loss_test_all, marker = 'o', mec = 'r', label='Average Test loss')
  plt.legend()
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()
  plt.savefig(os.path.join(sample_dir, 'Loss.png'))

  # Plotting latent space using PCA
  if (epoch) % 5 == 0:

    X_trainT = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    pca = PCA(n_components = 2)
    X_trainPCA = pca.fit_transform(X_trainT)
    # X_test = pca.transform(X_test)
    # breakpoint()

    plt.clf()
    no_points_plt = 10000
    X = X_trainPCA[0:no_points_plt,0]
    Y = X_trainPCA[0:no_points_plt,1]
    plt.scatter(X, Y, color = colors[X_labels[0:no_points_plt]])
    plt.title('PCA latent space')
    plt.show()
    plt.savefig(os.path.join(sample_dir, f'PCA_latentSpace-{epoch}.png'))

  model.train()
