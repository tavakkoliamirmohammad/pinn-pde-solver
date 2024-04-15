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

    return torch.mean(f**2)


model = HelmholtzPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x_train = torch.Tensor(np.random.rand(1000, 1) * 2 - 1)  # x in [-1, 1]
y_train = torch.Tensor(np.random.rand(1000, 1) * 2 - 1)  # y in [-1, 1]
k = 3.0  # Wave number

for epoch in range(5000):
    optimizer.zero_grad()
    loss = helmholtz_loss(model, x_train, y_train, k)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')