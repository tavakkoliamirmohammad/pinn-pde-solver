import torch
import torch.nn as nn
import numpy as np

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

def wave_equation_loss(model, x, y, t, c):
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True

    u = model(x, y, t)

    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]

    u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
    u_xx = torch.autograd.grad(torch.autograd.grad(u.sum(), x, create_graph=True)[0].sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(torch.autograd.grad(u.sum(), y, create_graph=True)[0].sum(), y, create_graph=True)[0]

    f = u_tt - c**2 * (u_xx + u_yy)

    return torch.mean(f**2)

model = WavePINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)
y_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)
t_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)
c = torch.tensor(0.5)

for epoch in range(2000):
    optimizer.zero_grad()
    loss = wave_equation_loss(model, x_train, y_train, t_train, c)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
