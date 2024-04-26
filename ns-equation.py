import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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


def manufactured_solution(x, y, t, nu):
    u = -torch.sin(np.pi * x) * torch.cos(np.pi * y) * torch.exp(-2 * np.pi**2 * nu * t)
    v = torch.cos(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-2 * np.pi**2 * nu * t)
    p = torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-4 * np.pi**2 * nu * t)
    return u, v, p

def derivatives(x, y, t, nu):
    u, v, p = manufactured_solution(x, y, t, nu)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    return u, v, p, u_x, u_y, u_t, u_xx, u_yy, v_x, v_y, v_t, v_xx, v_yy, p_x, p_y

def navier_stokes_loss(model, x, y, t, rho, nu):
    u, v, p, u_x, u_y, u_t, u_xx, u_yy, v_x, v_y, v_t, v_xx, v_yy, p_x, p_y = derivatives(x, y, t, nu)

    output = model(x, y, t)
    pred_u = output[:, 0]
    pred_v = output[:, 1]
    pred_p = output[:, 2]

    pred_u_x = torch.autograd.grad(pred_u.sum(), x, create_graph=True)[0]
    pred_u_y = torch.autograd.grad(pred_u.sum(), y, create_graph=True)[0]
    pred_u_t = torch.autograd.grad(pred_u.sum(), t, create_graph=True)[0]
    pred_u_xx = torch.autograd.grad(pred_u_x.sum(), x, create_graph=True)[0]
    pred_u_yy = torch.autograd.grad(pred_u_y.sum(), y, create_graph=True)[0]

    pred_v_x = torch.autograd.grad(pred_v.sum(), x, create_graph=True)[0]
    pred_v_y = torch.autograd.grad(pred_v.sum(), y, create_graph=True)[0]
    pred_v_t = torch.autograd.grad(pred_v.sum(), t, create_graph=True)[0]
    pred_v_xx = torch.autograd.grad(pred_v_x.sum(), x, create_graph=True)[0]
    pred_v_yy = torch.autograd.grad(pred_v_y.sum(), y, create_graph=True)[0]

    pred_p_x = torch.autograd.grad(pred_p.sum(), x, create_graph=True)[0]
    pred_p_y = torch.autograd.grad(pred_p.sum(), y, create_graph=True)[0]

    continuity_residual = pred_u_x + pred_v_y
    u_momentum_residual = pred_u_t + pred_u * pred_u_x + pred_v * pred_u_y - nu * (pred_u_xx + pred_u_yy) + pred_p_x / rho
    v_momentum_residual = pred_v_t + pred_u * pred_v_x + pred_v * pred_v_y - nu * (pred_v_xx + pred_v_yy) + pred_p_y / rho

    loss = torch.mean(continuity_residual**2) + torch.mean(u_momentum_residual**2) + torch.mean(v_momentum_residual**2)
    return loss

resolutions = [50, 100, 200, 400]
rho = 1.0
nu = 0.01
final_losses = []

for res in resolutions:
    model = NSPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x_train = torch.linspace(-1, 1, res).unsqueeze(1).requires_grad_(True)
    y_train = torch.linspace(-1, 1, res).unsqueeze(1).requires_grad_(True)
    t_train = torch.linspace(0, 1, res).unsqueeze(1).requires_grad_(True)

    losses = []
    for epoch in range(3000):
        optimizer.zero_grad()
        loss = navier_stokes_loss(model, x_train, y_train, t_train, rho, nu)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    plt.plot(losses, label=f'Res {res}')
    final_losses.append(losses[-1])
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
