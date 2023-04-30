import torch
import matplotlib.pyplot as plt

# initialization of the split interval x
x1 = torch.arange(-2, -0.1, 0.05)
x2 = torch.arange(0.1, 2.05, 0.05)
x = torch.cat((x1, x2))
xlen = len(x)

# calculation of function degeneracies on the partition x
# function y=sqrt(x^2+1)/x
y = torch.zeros(xlen)
for n in range(xlen):
    y[n] = torch.sqrt(x[n]**2 + 1) / x[n]

# initialization of the network
P = x.unsqueeze(1)
T = y.unsqueeze(1)

# Tanh - activation function of the first hidden layer
# Identity - activation function of the output layer
net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.Tanh(),
    torch.nn.Linear(10, 1),
    torch.nn.Identity()
)

# setting up training parameters
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_fn = torch.nn.MSELoss()

# training of the network
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = net(P)
    loss = loss_fn(outputs, T)
    loss.backward()
    optimizer.step()

# simulation of network training results
a = net(P)

# plotting of graphs
plt.xlabel('time')
plt.ylabel('output')
plt.title('Sim Fn')
# output function
plt.plot(P.detach().numpy(), T.detach().numpy(), 'x')
#simulation
plt.plot(P.detach().numpy(), a.detach().numpy())
# error
plt.plot(P.detach().numpy(), a.detach().numpy() - T.detach().numpy())
plt.grid()
plt.show()