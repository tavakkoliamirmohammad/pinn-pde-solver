import torch
import torch.nn as nn
import numpy as np


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
        inputs = torch.cat([x, y, t], axis=1)
        return self.net(inputs)

def physics_informed_loss(model, x, y, t, alpha):
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True

    u = model(x, y, t)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]

    f = u_t - alpha * (u_xx + u_yy)  # Residual of the heat equation

    return torch.mean(f**2)  # Mean squared error on the PDE residual


model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
alpha = torch.tensor(100)

x_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)
y_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)
t_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)

# Training loop
for epoch in range(2000):
    optimizer.zero_grad()
    loss = physics_informed_loss(model, x_train, y_train, t_train, alpha)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
