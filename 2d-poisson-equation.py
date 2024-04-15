import torch
import torch.nn as nn
import numpy as np

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


def poisson_loss(model, x, y, f):
    x.requires_grad = True
    y.requires_grad = True

    u = model(x, y)

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]

    laplacian_u = u_xx + u_yy

    return torch.mean((laplacian_u - f)**2)


model = PoissonPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)
y_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)
f_train = torch.tensor(np.sin(x_train) * np.cos(y_train), dtype=torch.float32)

for epoch in range(2000):
    optimizer.zero_grad()
    loss = poisson_loss(model, x_train, y_train, f_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
