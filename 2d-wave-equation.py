import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class WavePINN(nn.Module):
    def __init__(self):
        super(WavePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], axis=1)
        return self.net(inputs)


def manufactured_solution(x, y, t, c, omega=np.pi):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.cos(omega * t)

def manufactured_solution_tt(x, y, t, c, omega=np.pi):
    return -omega**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.cos(omega * t)

def manufactured_solution_xx_yy(x, y, t, c, omega=np.pi):
    return -2 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.cos(omega * t)

def wave_equation_loss(model, x, y, t, c):
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True

    u_pred = model(x, y, t)

    u_t = torch.autograd.grad(u_pred.sum(), t, create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]

    u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u_pred.sum(), y, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]

    f = u_tt - c**2 * (u_xx + u_yy)

    u_true_tt = manufactured_solution_tt(x, y, t, c)
    u_true_xx_yy = manufactured_solution_xx_yy(x, y, t, c)

    f_true = u_true_tt - c**2 * u_true_xx_yy

    return torch.mean((f - f_true)**2)


c = torch.tensor(0.5)

resolutions = [50, 100, 200, 400, 1000]
final_losses = []

for res in resolutions:
    model = WavePINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create the grid
    x_train = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32).unsqueeze(1)
    t_train = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32).unsqueeze(1)
    c = torch.tensor(0.5)

    losses = []
    for epoch in range(5000):
        optimizer.zero_grad()
        loss = wave_equation_loss(model, x_train, y_train, t_train, c)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    final_losses.append(losses[-1])
    print(f"Res {res}: {final_losses[-1]}")
    plt.plot(losses, label=f'Res {res}')

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
