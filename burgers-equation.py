import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], axis=1)
        return self.net(inputs)

def manufactured_solution(x, t, nu):
    return torch.sin(np.pi * x) * torch.exp(-nu * np.pi**2 * t)

def manufactured_solution_derivatives(x, t, nu):
    u = manufactured_solution(x, t, nu)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    return u, u_x, u_xx, u_t

def burgers_loss(model, x, t, nu):
    x.requires_grad = True
    t.requires_grad = True

    u_pred = model(x, t).squeeze()

    u_t_pred = torch.autograd.grad(u_pred.sum(), t, create_graph=True)[0]
    u_x_pred = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
    u_xx_pred = torch.autograd.grad(u_x_pred.sum(), x, create_graph=True)[0]

    u, u_x, u_xx, u_t = manufactured_solution_derivatives(x, t, nu)
    f_pred = u_t_pred + u_pred * u_x_pred - nu * u_xx_pred
    f_true = u_t + u * u_x - nu * u_xx

    loss = torch.mean((f_pred - f_true)**2)
    return loss


nu = 0.01 / np.pi
resolutions = [50, 100, 200, 400, 1000]
final_losses = []

for res in resolutions:
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x_train = torch.linspace(-1, 1, res).unsqueeze(-1).requires_grad_(True)
    t_train = torch.linspace(0, 1, res).unsqueeze(-1).requires_grad_(True)

    losses = []
    for epoch in range(10000):
        optimizer.zero_grad()
        loss = burgers_loss(model, x_train, t_train, nu)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            losses.append(loss.item())

    final_losses.append(losses[-1])
    plt.plot(losses, label=f'Res {res}')
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
