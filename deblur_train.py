import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import time
import argparse
from tqdm import tqdm
from models.SRCNN import SRCNN
from data.Dataset import DeblurDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type = int, default = 40, help = "No. of epochs to train")
args = vars(parser.parse_args())

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

image_dir = './outputs/saved_images'
os.makedirs(image_dir, exist_ok = True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
batch_size = 2

# Get the data
gauss_blur = os.listdir('./input/gaussian_blurred')
gauss_blur.sort()
sharp = os.listdir('./input/sharp')
sharp.sort()

x_blur = []
for i in range(len(gauss_blur)):
    x_blur.append(gauss_blur[i])

y_sharp = []
for i in range(len(sharp)):
    y_sharp.append(sharp[i])

(x_train, x_val, y_train, y_val) = train_test_split(x_blur, y_sharp, test_size=0.1)
print(f"Training data: {len(x_train)}")
print(f"Validation data: {len(x_val)}")

transform = transforms.Compose([
                                transforms.ToPILImage(), 
                                transforms.Resize((224, 224)), 
                                transforms.ToTensor()
                                ])

train_data = DeblurDataset(x_train, y_train, transform)
val_data = DeblurDataset(x_val, y_val, transform)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Initialize Model Parameters
model = SRCNN().to(device)
print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        verbose=True
    )

# Training
def train(model, dataloader, epoch):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total = int(len(train_data)/dataloader.batch_size)):
        blur_image = data[0]
        sharp_image = data[1]
        blur_image  = blur_image.to(device)
        sharp_image  = sharp_image.to(device)
        optimizer.zero_grad()
        outputs = model(blur_image)
        loss = criterion(outputs, sharp_image)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss/len(dataloader.dataset)
    print(f"Training Loss: {train_loss:.5f}")
    return train_loss 

# Validation
def valid(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total = int(len(val_data)/dataloader.batch_size)):
            blur_image = data[0]
            sharp_image = data[1]
            blur_image  = blur_image.to(device)
            sharp_image  = sharp_image.to(device)
            outputs = model(blur_image)
            loss = criterion(outputs, sharp_image)
            running_loss += loss.item()

            if epoch == 0 and i == int((len(val_data)/dataloader.batch_size)-1):
                save_decoded_image(sharp_image.cpu().data, name=f"./outputs/saved_images/sharp{epoch}.jpg")
                save_decoded_image(blur_image.cpu().data, name=f"./outputs/saved_images/blur{epoch}.jpg")
 
            if i == int((len(val_data)/dataloader.batch_size)-1):
                save_decoded_image(outputs.cpu().data, name=f"./outputs/saved_images/val_deblurred{epoch}.jpg")
 
        val_loss = running_loss/len(dataloader.dataset)
        print(f"Val Loss: {val_loss:.5f}")
        
        return val_loss

# Execution
train_loss = []
val_loss = []
start = time.time()
for epoch in range(args['epochs']):
    print(f"Epoch {epoch+1} of {args['epochs']}")
    train_epoch_loss = train(model, trainloader, epoch)
    val_epoch_loss = valid(model, valloader, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
end = time.time()
print(f"Took {((end-start)/60):.3f} to train")

# Plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./outputs/loss.png')
plt.show()
 
# save the model to disk
print('Saving model...')
torch.save(model.state_dict(), './outputs/model.pth')