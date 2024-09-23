import torch
from model import adaptive_penis_net
from fft_framework import simpDiff
from torchvision import utils
import matplotlib.pyplot as plt


def main():
    model = adaptive_penis_net(img_size=(31, 31), num_classes=11, hidden_dim=1024, slice_points=[1, 3, 8, 16])
    diff_obj = simpDiff(image_size=(31, 31), slice_points=[1, 3, 8, 16], device="cpu")

    state_dict = torch.load('model_120.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    sample_class = torch.arange(10)
    #sample_class = torch.tensor([1])
    out, fourier = diff_obj.filtered_sample(model, sample_class, omega=2.5, device="cpu")
    #grid = utils.make_grid(fourier[0].abs(), nrow=163)
    grid = utils.make_grid(out, nrow=10).clamp(0, 1)
    plt.imshow(grid.detach().permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    

if __name__ == '__main__':
    main()
