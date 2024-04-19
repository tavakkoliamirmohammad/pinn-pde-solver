import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)

def manufactured_solution(x, y, t, alpha):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-2 * np.pi**2 * alpha * t)

def temporal_derivative(x, y, t, alpha):
    return -2 * np.pi**2 * alpha * torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-2 * np.pi**2 * alpha * t)

def spatial_second_derivative(x, y, t, alpha):
    return -np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-2 * np.pi**2 * alpha * t) * (2 * np.pi**2 * t + 1)

def modified_physics_informed_loss(model, x, y, t, alpha):
    u_pred = model(x, y, t)
    
    # First derivatives
    u_t_pred = torch.autograd.grad(u_pred.sum(), t, create_graph=True)[0]
    u_x_pred = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
    u_y_pred = torch.autograd.grad(u_pred.sum(), y, create_graph=True)[0]
    
    # Second derivatives
    u_xx_pred = torch.autograd.grad(u_x_pred.sum(), x, create_graph=True)[0]
    u_yy_pred = torch.autograd.grad(u_y_pred.sum(), y, create_graph=True)[0]

    f_pred = u_t_pred - alpha * (u_xx_pred + u_yy_pred)
    
    # For the true solution
    u_true = manufactured_solution(x, y, t, alpha)
    u_t_true = temporal_derivative(x, y, t, alpha)
    u_xx_yy_true = spatial_second_derivative(x, y, t, alpha)

    f_true = u_t_true - alpha * u_xx_yy_true

    loss = torch.mean((f_pred - f_true)**2)
    return loss


model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
alpha = torch.tensor(100.0)

resolutions = [50, 100, 200, 400, 1000]
resolution_losses = []

for res in resolutions:
    x_train = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    y_train = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    t_train = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32).unsqueeze(1).requires_grad_(True)

    losses = []
    for epoch in range(3000):
        optimizer.zero_grad()
        loss = modified_physics_informed_loss(model, x_train, y_train, t_train, alpha)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    resolution_losses.append(losses[-1])
    print(f"Res {res}: {resolution_losses[-1]}")
    plt.plot(losses, label=f'Res {res}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs at Different Resolutions')
plt.legend()
plt.show()

plt.figure()
plt.plot(resolutions, resolution_losses, marker='o')
plt.xlabel('Resolution')
plt.ylabel('Final Loss')
plt.title('Final Loss at Different Resolutions')
plt.show()