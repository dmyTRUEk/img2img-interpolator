# img2img interpolator

import os
import sys

from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 100
#TEST_INPUTS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
TEST_INPUTS = [
    -1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1,
    0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1,
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
]


def main():
    img0_path = sys.argv[1]
    img1_path = sys.argv[2]

    now = datetime.now()

    # Define the transform to resize images to IMAGE_SIZExIMAGE_SIZE pixels
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Create instances of the neural network, dataset, and dataloader
    model = ImageGenerator()
    dataset = ImageDataset([img0_path, img1_path], transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for i, inputs in enumerate(dataloader):
            label = torch.tensor([0.0 if i==0 else 1.0])

            # Forward pass
            outputs = model(label.float())

            # Compute the loss
            loss = criterion(outputs, inputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    res_dir = "outputs/" + now.strftime("%Y-%m-%d_%H-%M-%S")
    maybe_create_dir(res_dir)

    # Test the trained model
    for value in TEST_INPUTS:
        test_input = torch.tensor([value])
        generated_image = model(test_input.float())
        output_image = transforms.ToPILImage()(generated_image[0].cpu())
        output_image.save(f"{res_dir}/image_{value:.2f}.png")



# Define a simple neural network
class ImageGenerator(nn.Module):
    def __init__(self):
        super(ImageGenerator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 10), # input layer
            nn.ReLU(),
            nn.Linear(10, 100),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            # nn.Linear(100, 1000),
            # nn.Sigmoid(),
            # nn.Dropout(0.5),
            nn.Linear(100, IMAGE_SIZE * IMAGE_SIZE * 3), # output layer
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)



# Custom dataset class for loading and processing images
class ImageDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # TODO: is it ok to LOAD images here?
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img



def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)





if __name__ == "__main__":
    main()

