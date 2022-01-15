function g = g_SRMCC(x,D,muK)
mu = muK(1);
K = muK(2);
[N1,P] = size(D);
Xh = [D(1:N1-1,:); ones(1,P)];
y = D(N1,:);
W = reshape(x,N1,K);
g = zeros(N1*K,1);
for k = 1:K
    ink = find(y == k);
    gk = -sum(Xh(:,ink),2)/P;
    wk = W(:,k);
    for p = 1:P
        xp = Xh(:,p);
        tp = exp(xp'*wk);
        t0 = P*sum(exp(xp'*W));
        gk = gk + (tp/t0)*xp;
    end
    g((k-1)*N1+1:k*N1) = gk;
end
g = g + mu*W(:);