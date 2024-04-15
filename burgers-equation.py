import torch
import torch.nn as nn
import numpy as np


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


def burgers_loss(model, x, t, nu):
    x.requires_grad = True
    t.requires_grad = True

    u = model(x, t)
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

    f = u_t + u * u_x - nu * u_xx

    return torch.mean(f**2)

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x_train = torch.Tensor(np.random.rand(1000, 1) * 2 - 1)  # x in [-1, 1]
t_train = torch.Tensor(np.random.rand(1000, 1) * 1)      # t in [0, 1]
nu = 0.01 / np.pi  # Viscosity

for epoch in range(5000):
    optimizer.zero_grad()
    loss = burgers_loss(model, x_train, t_train, nu)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')