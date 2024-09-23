import os
import torch
from fft_framework import simpDiff
import torch.optim as optim
import matplotlib.pyplot as plt
from einops import rearrange
from torchvision import utils, transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


def get_dataloaders(rank, world_size, epoch):
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

    sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
    sampler.set_epoch(epoch)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,  # number of images to load at each iteration
        shuffle=False,  # shuffle the data at every epoch
        num_workers=2,  # number of subprocesses to use for data loading
        sampler=sampler
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
        num_workers=2,  # number of subprocesses to use for data loading
    )

    # Define the classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def train(model, num_epochs, run_name, rank, world_size):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    diff_obj = simpDiff(image_size=(31, 31), slice_points=model.slice_points, device=f"cuda:{rank}")

    logdir = f"/home/ubuntu/logs/{run_name}"
    if not os.path.exists(logdir) and rank == 0:
        os.makedirs(logdir, exist_ok=True)

    for epoch in range(num_epochs):
        trainloader, testloader, classes = get_dataloaders(rank, world_size, epoch)
        running_loss = 0.0

        if epoch % 20 == 0 and rank == 0:
            torch.save(model.state_dict(), f"{logdir}/model_{epoch}.pt")


        for idx, (images, labels) in enumerate(trainloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # CFG masking
            mask = torch.rand(len(labels)) < 0.1
            labels[mask] = 10

            inputs = diff_obj.to_fft(images, torch.tensor([16]))
            inputs, labels = inputs.cuda(), labels.cuda()
            b, t, s = inputs.shape
            x = inputs[:, :-1, :]
            y = inputs[:, 1:, :]
            outputs = model(x, labels)
            loss = (outputs - y).square().mean()
            # Backward pass and optimization
            loss.backward()
            dist.barrier()
            
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad = p.grad / world_size

            dist.barrier()
            optimizer.step()

            # Update the running loss
            running_loss += loss.item()

            if idx % 100 == 0 and rank == 0:
                print(f"Epoch:{epoch}, Batch:{idx}, Loss:{running_loss / (idx + 1)}")


        if rank == 0:
            model.eval()

            sample_class = torch.arange(10)
            out, fourier = diff_obj.filtered_sample(model, sample_class, omega=2, device="cuda:0")
            grid = utils.make_grid(out, nrow=10).clamp(0, 1)

            plt.imshow(grid.real.cpu().detach().squeeze(0).permute(1, 2, 0))
            plt.axis("off")
            plt.savefig(f"{logdir}/sample_{epoch}.png")

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
