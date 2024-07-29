% Deterministic Optimal Growth Model : Discrete state space approximation
% Code from Chris Edmond's lecture notes on Macroeconomics (http://www.chrisedmond.net/phd2019.html)

clc; clear; close all;


%%%%% Economic parameters

alpha = 1/3;          %% capital's share in production function
beta  = 0.95;         %% time discount factor
delta = 0.05;         %% depreciation rate
sigma = 1;            %% CRRA (=1/IES)
rho   = (1/beta) - 1; %% implied rate of time preference

kstar = (alpha / (rho + delta))^(1/(1-alpha)); %% steady state
kbar  = (1/delta)^(1/(1-alpha));


%%%%% Numerical parameters

max_iter = 500;   %% maximum number of iterations
tol      = 1e-7;  %% treat numbers smaller than this as zero
penalty  = 10^16; %% for penalizing constraint violations


%%%%% Setting up the grid of capital stocks

n    = 1001; %% number of nodes for k grid
kmin = tol;  %% effectively zero
kmax = kbar; %% effective upper bound on k grid

k = linspace(kmin, kmax, n); %% linearly spaced k grid


%%%%% Return function

c = zeros(n, n);

for j = 1:n

    c(:, j) = k.^alpha + (1 - delta) * k - k(j);
end


%%%%% Penalize violations of feasibility conditions

violations = (c <= 0);

c = c.*(c>=0) + tol;

if sigma == 1
    u = log(c) - penalty * violations;
else
    u = (1/(1-sigma))*(c.^(1-sigma) - 1) - penalty * violations;
end


%%%%% Bellman iterations

% initial guess
v = zeros(n, 1);

% iterate on Bellman operator
for i=1:max_iter
    
    obj = u + beta * kron(ones(n, 1), v'); %% RHS of Bellman equation
    [Tv, argmax] = max(obj, [], 2);        %% maximize over obj to get Tv
    g = k(argmax);                         %% policy that attains the maximum
    
    % Check if converged
    error = norm(Tv-v, inf);
    fprintf('%4i %6.2e \n', [i, error])

    if error < tol
        break
    end;

    % If not converged, update and try again
    v = Tv;

end


