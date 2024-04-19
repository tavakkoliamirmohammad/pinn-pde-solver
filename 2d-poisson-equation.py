import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PoissonPINN(nn.Module):
    def __init__(self):
        super(PoissonPINN, self).__init__()
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
    return torch.sin(np.pi * x) * torch.cos(np.pi * y)

def manufactured_rhs(x, y):
    return -2 * np.pi**2 * torch.sin(np.pi * x) * torch.cos(np.pi * y)


def poisson_loss(model, x, y):
    x.requires_grad = True
    y.requires_grad = True

    u_pred = model(x, y)
    f_true = manufactured_rhs(x, y)

    u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u_pred.sum(), y, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]

    laplacian_u_pred = u_xx + u_yy

    return torch.mean((laplacian_u_pred - f_true)**2)


import matplotlib.pyplot as plt

model = PoissonPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
resolutions = [50, 100, 200, 400, 1000]
final_losses = []

for res in resolutions:
    model = PoissonPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_train = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32).unsqueeze(1)

    losses = []
    for epoch in range(7000):
        optimizer.zero_grad()
        loss = poisson_loss(model, x_train, y_train)
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