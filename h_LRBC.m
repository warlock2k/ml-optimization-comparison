function H = h_LRBC(w,X)
[N1,P] = size(X);
q1 = exp(X'*w);
q = q1./((1+q1).^2);
H = zeros(N1,N1);
for p = 1:P
    xp = X(:,p);
    H = H + q(p)*(xp*xp');
end
H = H/P;