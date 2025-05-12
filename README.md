# DEAL: Deep Attentive Least Squares for Image Reconstruction

[![arXiv](https://img.shields.io/badge/arXiv-2502.04079-b31b1b.svg)](https://arxiv.org/abs/2502.04079)
[![Project Website](https://img.shields.io/badge/Website-DEAL-blue)](https://mehrsapo.github.io/DEAL/)

This repository contains the official PyTorch implementation of **DEAL (Deep Attentive Least Squares)** — an interpretable framework for solving inverse problems in image reconstruction.

📄 **Paper**: [DEALing with Image Reconstruction: Deep Attentive Least Squares (arXiv:2502.04079)](https://arxiv.org/abs/2502.04079)  
🌐 **Project Website**: [mehrsapo.github.io/DEAL](https://mehrsapo.github.io/DEAL/)

---

## 🔧 Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mehrsapo/DEAL.git
   cd DEAL
   ```

2. **Install dependencies:**
   ```bash
   conda create -n deal python=3.11.2
   conda activate deal
   conda install pip
   pip install -r requirements.txt
   ```
---

## 🧠 Load the Model

```python
import torch
from deal import DEAL 

device = 'cuda:0' # change to your device name e.g. 'cpu', 'cuda:3', etc.
path_ckp = "trained_models/deal_color.pth" # change to deal_gray.pth for grayscale setup
ckp = torch.load(path_ckp, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

model = DEAL(color=True) # change to color=False for grayscale setup
model.to(device)
model.load_state_dict(ckp['state_dict'])
model.eval()
```

---

## 🔁 Reconstruct an Image

### 🧠 Reconstructing an Image with `solve_inverse_problem`

To reconstruct an image using DEAL, call the `solve_inverse_problem` method as follows:

```python
out_deal = model.solve_inverse_problem(y_torch, H, Ht, sigma_denoiser, lambda_)
```

#### 🔹 Arguments:

- `y_torch` (`torch.Tensor`): The observed measurement, shape **(1, C, H_y, W_y)**, where:
  - `C` is the number of channels (1 for grayscale, 3 for color)
  - `H_y`, `W_y` are the image height and width of the measurements.

- `H` (`callable`): Forward operator, a Python function that applies the degradation process to an image tensor of shape **(1, C, H, W)**.

- `Ht` (`callable`): Adjoint (transpose) of the forward operator.

- `sigma_denoiser` (`float`): Noise level assumed by the learned denoiser. A value of **15/255** is recommended for most tasks.

- `lambda_` (`float`): Regularization parameter controlling the strength of the prior. Needs tuning depending on the task and corruption severity.

#### 🔹 Returns:

- A reconstructed image tensor of shape **(1, C, H, W)**, clipped to the range **[0, 1]**.


---

## 📄 License

This project is released under the MIT License.

---

## 📫 Citation

If you use this code, please consider citing our paper:

```
@article{pourya2025dealing,
  title={DEALing with Image Reconstruction: Deep Attentive Least Squares},
  author={Pourya, Mehrsa and Kobler, Erich and Unser, Michael and Neumayer, Sebastian},
  journal={arXiv preprint arXiv:2502.04079},
  year={2025}
}
```




