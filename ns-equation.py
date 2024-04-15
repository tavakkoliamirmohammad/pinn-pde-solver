import torch
import torch.nn as nn
import numpy as np

class NSPINN(nn.Module):
    def __init__(self):
        super(NSPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3)
        )

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], axis=1)
        return self.net(inputs)


def navier_stokes_loss(model, x, y, t, rho, nu):
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True

    output = model(x, y, t)
    u = output[:, 0]
    v = output[:, 1]
    p = output[:, 2]

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]

    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]

    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]

    # Compute second derivatives
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]

    # Continuity equation loss
    continuity = u_x + v_y

    # Momentum equations loss
    u_momentum = u_t + u * u_x + v * u_y + (1/rho) * p_x - nu * (u_xx + u_yy)
    v_momentum = v_t + u * v_x + v * v_y + (1/rho) * p_y - nu * (v_xx + v_yy)

    return torch.mean(continuity**2) + torch.mean(u_momentum**2) + torch.mean(v_momentum**2)


model = NSPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)
y_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)
t_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)
rho = 1.0  # Density (assuming constant)
nu = 0.01  # Kinematic viscosity (assuming constant)

for epoch in range(3000):
    optimizer.zero_grad()
    loss = navier_stokes_loss(model, x_train, y_train, t_train, rho, nu)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
