from fft_framework import simpDiff
from model import adaptive_penis_net
import torch.otpim as otpim
import matplotlib.pyplot as plt
from einops import rearrange
from torchvision import utils


model = adaptive_penis_net(
    img_size=(31, 31), num_classes=11, hidden_dim=512, slice_points=[1, 3, 8, 16]
).to("cuda")


optimizer = optim.Adam(model.parameters(), lr=1e-4)
diff_obj = simpDiff(image_size=(31, 31), slice_points=[1, 3, 8, 16], device="cuda")


num_epochs = 1000
for epoch in range(num_epochs)[22:]:
    running_loss = 0.0

    if epoch % 20 == 0:
        torch.save(model.state_dict(), f"~/model_weights/weights_{epoch}.pt")

    for idx, (images, labels) in enumerate(trainloader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # CFG masking
        mask = torch.rand(len(labels)) < 0.1
        labels[mask] = 10

        inputs = diff_obj.to_fft(images, torch.tensor([16]))
        inputs, labels = inputs.to("cuda"), labels.to("cuda")
        b, t, s = inputs.shape
        x = inputs[:, :-1, :]
        y = inputs[:, 1:, :]
        outputs = model(x, labels)
        # loss_weights = torch.linspace(1, 2, 15).view(1, 15, 1).to('cuda')
        # loss = ((outputs[:, 1:, :] - y) * loss_weights).square().mean()
        loss = (outputs - y).square().mean()
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

        if idx % 500 == 0:
            model.eval()

            sample_class = torch.tensor([7])
            out, fourier = diff_obj.filtered_sample(
                model, sample_class, omega=2, device="cuda"
            )
            print(classes[sample_class])
            plt.imshow(out.real.cpu().detach().squeeze(0).permute(1, 2, 0))
            plt.axis("off")
            plt.show()

            fourier_imgs = torch.fft.ifftshift(torch.cumsum(fourier, axis=1), dim=(-2))
            fourier_imgs = torch.fft.irfft2(fourier_imgs, s=(31, 31))
            grid = utils.make_grid(fourier.squeeze(0).abs()[:30], nrow=57).clamp(0, 1)
            plt.imshow(grid.cpu().detach().permute(1, 2, 0))
            plt.axis("off")
            plt.show()
            # grid = utils.make_grid(fourier_imgs.squeeze(0), nrow=57).clamp(0, 1)
            # plt.imshow(grid.cpu().detach().permute(1, 2, 0))
            # plt.axis('off')
            # plt.show()

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
            plt.show()

            grid = utils.make_grid(
                torch.cat([test_cumsum_img[:30], real_cumsum_img[:30]], axis=0), nrow=30
            ).clamp(0, 1)
            plt.imshow(grid.cpu().detach().permute(1, 2, 0))
            plt.axis("off")
            plt.show()

            model.train()
            print(running_loss / (idx + 1))

    epoch_loss = running_loss / len(trainloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
