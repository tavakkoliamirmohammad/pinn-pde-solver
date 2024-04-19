import torch
import torch.nn as nn
import numpy as np


class HelmholtzPINN(nn.Module):
    def __init__(self):
        super(HelmholtzPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, y):
        inputs = torch.cat([x, y], axis=1)
        return self.net(inputs)

def manufactured_solution(x, y):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)

def helmholtz_loss(model, x, y, k):
    x.requires_grad = True
    y.requires_grad = True

    u = model(x, y)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]

    laplacian_u = u_xx + u_yy
    f = laplacian_u + k**2 * u  # Helmholtz equation
    f_true = (k**2 - 2 * np.pi**2) * manufactured_solution(x, y)

    return torch.mean((f - f_true)**2)

import matplotlib.pyplot as plt

resolutions = [50, 100, 200, 400, 1000]
k = 3.0
final_losses = []

for res in resolutions:
    model = HelmholtzPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x_train = torch.Tensor(np.linspace(-1, 1, res)).unsqueeze(1)
    y_train = torch.Tensor(np.linspace(-1, 1, res)).unsqueeze(1)

    losses = []
    for epoch in range(10000):
        optimizer.zero_grad()
        loss = helmholtz_loss(model, x_train, y_train, k)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            losses.append(loss.item())

    plt.plot(losses, label=f'Res {res}')
    final_losses.append(losses[-1])
    print(f"Res {res}: {final_losses[-1]}")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs at Different Resolutions')
plt.legend()
plt.show()

plt.figure()
plt.plot(resolutions, final_losses, marker='o')
plt.xlabel('Resolution')
plt.ylabel('Final Loss')
plt.title('Final Loss at Different Resolutions')
plt.show()
