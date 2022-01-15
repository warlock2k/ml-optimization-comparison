function f = f_SRMCC(x,D,muK)
mu = muK(1);
K = muK(2);
[N1,P] = size(D);
Xh = [D(1:N1-1,:); ones(1,P)];
y = D(N1,:);
W = reshape(x,N1,K);
f = 0;
for p = 1:P
    xp = Xh(:,p);
    t0 = sum(exp(xp'*W));
    tp = exp(xp'*W(:,y(p)));
    f = f + log(tp/t0);
end
xw1 = W(:);
f = -f/P + 0.5*mu*(xw1'*xw1);