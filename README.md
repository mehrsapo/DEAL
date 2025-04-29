# DEAL: Deep Attentive Least Squares for Image Reconstruction

This repository contains the official PyTorch implementation of **DEAL (Deep Attentive Least Squares)**, a model designed for solving inverse problems in image reconstruction by blending iterative solvers with learned attention-based denoisers.

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
from model import DEAL  # Make sure DEAL is defined in your 'model.py'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_ckp = "trained_models/deal_color.pth"
ckp = torch.load(path_ckp, map_location={'cuda:0': device, 'cuda:1': device, 'cuda:2': device, 'cuda:3': device})

model = DEAL(color=True)
model.to(device)
model.load_state_dict(ckp['state_dict'])
model.eval()
```

---

## 🔁 Reconstruct an Image

To reconstruct an image using DEAL, call the `solve_inverse_problem` method:

```python
out_deal = model.solve_inverse_problem(y_torch, H, Ht, sigma_denoiser, lambda_)
```

- `y_torch`: torch.Tensor of the corrupted measurement  
- `H`: Forward operator (implemented as a Python function)  
- `Ht`: Adjoint of the forward operator  
- `sigma_denoiser`: Suggested value is `15`  
- `lambda_`: Needs to be tuned depending on the problem

---

## 📄 License

This project is released under the MIT License.

---

## 📫 Citation

If you use this code, please consider citing our paper:

```
@inproceedings{pourya2025deal,
  title={DEALing with Image Reconstruction: Deep Attentive Least Squares},
  author={Pourya, Mehrsa and et al.},
  booktitle={ArXIV},
  year={2025}
}
```
