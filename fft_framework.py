import torch
from torchvision import transforms
from einops import rearrange
from PIL import Image


class simpDiff:
    def __init__(self, image_size, slice_points, device="cpu"):
        h, w = image_size
        width = w // 2 + 1
        self.device = device
        self.image_size = image_size
        self.slice_points = slice_points
        self.max_t = len(self.get_masks(torch.tensor([width])))
        self.fund_freq = fund_freq = transforms.ToTensor()(Image.open("fund_freq_2.png"))

    def slice_mask(self, slice_num):
        h, w = self.image_size

        width = w // 2 + 1

        # Create coordinate grid
        y, x = torch.meshgrid(torch.arange(h), torch.arange(width), indexing="ij")

        # Center the coordinates
        x = x - (width)  # Center is at n//2 + 1
        y = y - h // 2

        # Calculate angles and convert to [0, π] range (left half only)
        angles = torch.atan2(y, x) % torch.pi

        # Convert angles to slice indices (as if there were 2m slices in a full circle)
        slice_indices = (angles / torch.pi * slice_num).long()

        # Create a list to store individual slice masks
        slice_masks = []

        # Generate mask for each slice
        for i in range(slice_num):
            slice_mask = slice_indices == i
            slice_masks.append(slice_mask)

        return torch.stack(slice_masks, dim=0).float()

    def get_masks(self, radius):
        h, w = self.image_size

        width = w // 2 + 1

        if self.slice_points == []:
            slice_masks = torch.ones(1, 1, *self.image_size).to(self.device)
        else:
            slice_masks = rearrange(self.slice_mask(2), "n h w -> n 1 1 h w").to(
                self.device
            )

        center = (h // 2, w // 2)
        Y, X = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing="ij",
        )
        dist_from_center = (X - center[1]) ** 2 + (Y - center[0]) ** 2
        dist_from_center = rearrange(dist_from_center, "h w -> 1 1 h w")

        cutoff_masks = []
        for rad in range(radius)[1:]:
            cutoff_masks.append(
                (
                    (dist_from_center >= (rad - 0.5) ** 2)
                    & (dist_from_center <= (rad + 0.5) ** 2)
                ).float()[:, :, :, :width]
            )

        out = []
        num_slices = 2
        for i, mask in enumerate(cutoff_masks):
            if i in self.slice_points:
                num_slices *= 2
                slice_masks = rearrange(
                    self.slice_mask(num_slices), "n h w -> n 1 1 h w"
                ).to(self.device)

            for slice_mask in slice_masks:
                out.append(mask.flip(-1) * slice_mask.flip(-1))

        return torch.stack(out, dim=0)

    def to_fft(self, img, radius):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        b, c, h, w = img.shape
        radius = rearrange(radius, "b -> b 1 1 1")
        width = w // 2 + 1

        # Perform fft on image
        images_fft = torch.fft.rfft2(img)
        images_fft_shifted = torch.fft.fftshift(images_fft, dim=(-2))

        center = (h // 2, w // 2)

        fund_freq = torch.zeros(
            b, 3, h, width, dtype=torch.complex64, device=img.device
        )
        fund_freq[:, :, center[0], 0] = images_fft_shifted[:, :, center[0], 0]
        removed_freqs = [fund_freq.to(self.device)]

        if radius > 1:
            masks = self.get_masks(radius)
            for mask in masks:
                removed_freqs.append(images_fft_shifted.to(self.device) * mask)

        removed_freqs = torch.stack(removed_freqs, dim=1)
        removed_freqs = rearrange(removed_freqs, "b s c h w -> b s (c h w)")

        real_freqs = removed_freqs.real
        imag_freqs = removed_freqs.imag

        freqs = torch.cat([real_freqs, imag_freqs], dim=-1)
        return freqs

    def from_fft(self, fft_img):
        # fft_img = torch.complex(*torch.chunk(fft_img, 2, axis=-1))
        fft_img = fft_img.sum(axis=1)
        fft_img = rearrange(
            fft_img,
            "b (c h w) -> b c h w",
            h=self.image_size[0],
            w=self.image_size[1] // 2 + 1,
        )
        fft_unshifted = torch.fft.ifftshift(fft_img, dim=(-2))
        return torch.fft.irfft2(fft_unshifted, s=self.image_size)

    @torch.no_grad()
    def sample(self, model, c, omega, device="cpu"):
        h, w = self.image_size
        width = h // 2 + 1

        # init_freq = torch.zeros(6, h, width)
        # init_freq[:3, h//2, 0] = torch.randn(3) * 400
        # init_freq = rearrange(init_freq, 'c h w -> 1 1 (c h w)').to(self.device)
        init_freq = self.to_fft(self.fund_freq.unsqueeze(0), torch.tensor([1]))

        seq = torch.empty(1, 0, 3 * 2 * h * width).to(device)
        seq = torch.cat([seq, init_freq], axis=-2)
        for _ in range(self.max_t):
            pred = model(seq, c.to(device))[:, -1, :].unsqueeze(0)
            pred_w = model(seq, torch.tensor([10]).to(device))[:, -1, :].unsqueeze(0)
            pred = (1 + omega) * pred - omega * pred_w
            seq = torch.cat([seq, pred], axis=-2)
        real, imag = torch.chunk(seq, 2, axis=-1)
        out = torch.complex(real, imag)
        image = self.from_fft(out)
        fourier = rearrange(out, "b s (c h w) -> b s c h w", c=3, h=h, w=w)
        return image, fourier

    @torch.no_grad()
    def filtered_sample(self, model, c, omega, device="cpu"):
        h, w = self.image_size
        width = w // 2 + 1

        # init_freq = torch.zeros(6, h, width)
        # init_freq[:3, h//2, 0] = torch.randn(3)
        # init_freq = rearrange(init_freq, 'c h w -> 1 1 (c h w)').to(self.device)
        init_freq = self.to_fft(self.fund_freq.unsqueeze(0), torch.tensor([1]))
        if len(c) > 1:
            init_freq = init_freq.repeat(len(c), 1, 1)

        seq = torch.empty(len(c), 0, 3 * 2 * h * width).to(device)
        seq = torch.cat([seq, init_freq], axis=-2)

        # Create filter masks
        masks = self.get_masks(torch.tensor([16]))

        for i in range(self.max_t):
            pred = model(seq, c.to(device))[:, -1, :].unsqueeze(1)
            pred_w = model(seq, torch.tensor([10]).to(device))[:, -1, :].unsqueeze(1)
            pred = (1 + omega) * pred - omega * pred_w
            if i < 5:
                pred = pred * 0.8 + 0.2 * torch.randn_like(pred)
            pred = rearrange(pred, "b s (c h w) -> b s c h w", c=6, h=h, w=width)
            pred = pred * masks[i]
            pred = rearrange(pred, "b s c h w -> b s (c h w)")
            seq = torch.cat([seq, pred], axis=-2)
            print(f"On step {i} / {self.max_t}")
        real, imag = torch.chunk(seq, 2, axis=-1)
        out = torch.complex(real, imag)
        image = self.from_fft(out)
        fourier = rearrange(out, "b s (c h w) -> b s c h w", c=3, h=h, w=width)
        return image, fourier
