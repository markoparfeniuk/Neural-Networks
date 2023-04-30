% initialization of the split interval x
x1 = -2:0.05:-0.1;
x2 = 0.1:0.05:2;
x = [x1 x2];
xlen = length(x);
% calculation of function degeneracies on the partition x
% function y=sqrt(x^2+1)/x
y = zeros(1, xlen);
for n = 1:xlen
y(n) = sqrt(x(n)^2+1)/x(n);
end
% initialization of the network
P = x; T = y;
% [-4 4] - the interval at which the approximation will be performed
% [10 1] - two layers of the network:
% 10 - number of neurons in the first hidden layer
% 1 - the number of neurons in the output layer
% {'tansig', 'purelin'} - activation functions of the corresponding layers. here
% tansig - activation function of the first hidden layer
% purelin - activation function of the output layer
% traingd - training scheme. here is the gradient method
net = newff([-2 2], [10 1], {'tansig', 'purelin'}, 'traingd');;
% setting up training parameters
net.trainParam.show = 50; %.
net.trainParam.lr = 0.05; %
net.trainParam.epochs = 1000; % maximum number of epochs (iterations)
net.trainParam.goal = 1e-3; % stopping criterion
% training of the network
net1 = train(net, P, T);
% simulation of network training results
a = sim(net1, P);
% plotting of graphs
hold on;
xlabel('time'); ylabel('output'); title('Sim Fn');
% output function
plot(P, T, 'x');
% simulation
plot(P, a);
% error
plot(P, a-T);
grid;
hold off;