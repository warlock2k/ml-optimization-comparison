% Logistic regression for binary classification using Newton algorithm.
% Inputs:
% X: input data matrix where input samples are given as colum vectors.
% y: a vector whose pth component is the label for the pth input sample,
%    which is set to 1 if the pth sample belongs to class P, or set to
%    -1 if the pth sample belongs to class N.
% K: number of Newton iterations.
% Outputs:
% ws: a column vector of optimized weight w* and bias b*.
% C2: 2 x 2 confusion matrix.
% Written by W.-S. Lu, University of Victoria.
% Last modified: May 16, 2020.
% Example: 
% Dp = [1.5 3.5 5 6.9 8.4 10 11.2; 6.5 6.5 5 3.7 2.6 5 1.5];
% Dn = [1 2.1 3.1 4 5.9 7.9 9 10.5; 4.5 3.5 4.9 4.2 2.7 2.2 1.6 0.8];
% X = [Dp Dn];
% y = [ones(1,7) -ones(1,8)];
% [ws,C2] = LRBC_newton(X,y,5);
function [ws,C2] = LRBC_newton(X,y,K)
y = y(:)';
[N,P] = size(X);
N1 = N + 1;
Ine = 1e-10*eye(N1);
ind = 1:1:P;
indp = find(y > 0);
indn = setdiff(ind,indp);
Xh = [X; ones(1,P)];
Xp = Xh(:,indp);
Xn = Xh(:,indn);
Xw = [Xp -Xn];
P1 = length(indp);
k = 0;
wk = zeros(N1,1);
while k < K
  gk = feval('g_LRBC',wk,Xw);
  Hk = feval('h_LRBC',wk,Xw) + Ine;
  dk = -Hk\gk;
  ak = bt_lsearch2019(wk,dk,'f_LRBC','g_LRBC',Xw);
  wk = wk + ak*dk;
  k = k + 1;
  fk = feval('f_LRBC',wk, Xw);
  fprintf('iter %1i: loss = %8.2e\n',k,fk)
end
ws = wk;
yt = sign(ws'*[Xp Xn]);
er = abs(y-yt)/2;
erp = sum(er(1:P1));
ern = sum(er(P1+1:P));
C2 = [P1-erp ern; erp P-P1-ern];