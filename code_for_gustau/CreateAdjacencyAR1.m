function A = CreateAdjacencyAR1(N,rho)

A = zeros(N,N);

for i = 1:N
    for j = 1:N
        A(i,j) = rho^abs((i-1)-(j-1));
    end
end