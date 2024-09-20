import os
import torch
from fft_framework import simpDiff
from model import adaptive_penis_net
import torch.optim as optim
import matplotlib.pyplot as plt
from einops import rearrange
from torchvision import utils, transforms, datasets
from torch.utils.data import DataLoader




def train(num_epochs, run_name):
# Load CIFAR
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(31)
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Download and load the training set
    trainset = datasets.CIFAR10(
        root='./data',  # directory where data will be saved
        train=True,  # download the training set
        download=True,  # download the data if it's not already available
        transform=transform  # apply the transformations
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,  # number of images to load at each iteration
        shuffle=True,  # shuffle the data at every epoch
        num_workers=2  # number of subprocesses to use for data loading
    )

# Download and load the test set
    testset = datasets.CIFAR10(
        root='./data',  # directory where data will be saved
        train=False,  # download the test set
        download=True,  # download the data if it's not already available
        transform=transform  # apply the transformations
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,  # number of images to load at each iteration
        shuffle=False,  # do not shuffle the test data
        num_workers=2  # number of subprocesses to use for data loading
    )

# Define the classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = adaptive_penis_net(img_size=(31, 31), num_classes=11, hidden_dim=512, slice_points=[1, 3, 8, 16]).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    diff_obj = simpDiff(image_size=(31, 31), slice_points=[1, 3, 8, 16], device="cpu")

    logdir = f"~/logs/{run_name}"
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)

    for epoch in range(num_epochs):
        running_loss = 0.0

        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"{logdir}/model_{epoch}.pt")


        for idx, (images, labels) in enumerate(trainloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # CFG masking
            mask = torch.rand(len(labels)) < 0.1
            labels[mask] = 10

            inputs = diff_obj.to_fft(images, torch.tensor([16]))
            inputs, labels = inputs.to("cpu"), labels.to("cpu")
            b, t, s = inputs.shape
            x = inputs[:, :-1, :]
            y = inputs[:, 1:, :]
            outputs = model(x, labels)
            # loss_weights = torch.linspace(1, 2, 15).view(1, 15, 1).to('cpu')
            # loss = ((outputs[:, 1:, :] - y) * loss_weights).square().mean()
            loss = (outputs - y).square().mean()
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the running loss
            running_loss += loss.item()

            if idx % 100 == 0:
                print(f"Epoch:{epoch}, Batch:{idx}, Loss:{running_loss / (idx + 1)}")



        model.eval()

        sample_class = torch.tensor([7])
        out, fourier = diff_obj.filtered_sample(
            model, sample_class, omega=2, device="cpu"
        )
        plt.imshow(out.real.cpu().detach().squeeze(0).permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(f"{logdir}/sample_{epoch}.png")

        fourier_imgs = torch.fft.ifftshift(torch.cumsum(fourier, axis=1), dim=(-2))
        fourier_imgs = torch.fft.irfft2(fourier_imgs, s=(31, 31))
        grid = utils.make_grid(fourier.squeeze(0).abs()[:30], nrow=57).clamp(0, 1)
        plt.imshow(grid.cpu().detach().permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(f"{logdir}/sample_grid_{epoch}.png")

        test_y = torch.complex(*torch.chunk(y[0], 2, axis=-1))
        test_y = rearrange(test_y, "s (c h w) -> s c h w", c=3, h=31, w=16)
        test_pred = torch.complex(*torch.chunk(outputs[0], 2, axis=-1))
        test_pred = rearrange(test_pred, "s (c h w) -> s c h w", c=3, h=31, w=16)

        # Missing fundamental, fund[0] adds it back
        fund = torch.complex(*torch.chunk(x[0], 2, axis=-1))
        fund = rearrange(fund, "s (c h w) -> s c h w", c=3, h=31, w=16)
        test_cumsum = torch.cumsum(test_pred, axis=0) + fund[0]
        test_cumsum_img = torch.fft.ifftshift(test_cumsum, dim=(-2))
        test_cumsum_img = torch.fft.irfft2(test_cumsum_img, s=(31, 31))

        real_cumsum = torch.cumsum(test_y, axis=0) + fund[0]
        real_cumsum_img = torch.fft.ifftshift(real_cumsum, dim=(-2))
        real_cumsum_img = torch.fft.irfft2(real_cumsum_img, s=(31, 31))

        test_diff = test_y - test_pred

        test_y, test_pred, test_diff = test_y[:30], test_pred[:30], test_diff[:30]
        grid = utils.make_grid(
            torch.cat([test_y.abs(), test_pred.abs(), test_diff.abs()], axis=0),
            nrow=30,
        ).clamp(0, 1)
        plt.imshow(grid.cpu().detach().permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(f"{logdir}/targets_{epoch}.png")

        grid = utils.make_grid(
            torch.cat([test_cumsum_img[:30], real_cumsum_img[:30]], axis=0), nrow=30
        ).clamp(0, 1)
        plt.imshow(grid.cpu().detach().permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(f"{logdir}/reconstruction_{epoch}.png")

        model.train()

        epoch_loss = running_loss / len(trainloader)
        print(running_loss / (idx + 1))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


if __name__ == '__main__':
    train(100, 'test')
