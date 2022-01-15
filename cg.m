% To implement CG algorithm with Polak-Ribiere-Polyak-(plus)'s beta.
% Example:
% [xs,fs,k] = cg('f_rosen','g_rosen',[-1;-1],1e-6);
function [Ws,fs,k] = cg(fname,gname,wk,X,y,epsi)
format compact
format long
[N,P] = size(X);
ind = 1:1:P;
indp = find(y > 0);
indn = setdiff(ind,indp);
Xh = [X; ones(1,P)];
Xp = Xh(:,indp);
Xn = Xh(:,indn);
Xw = [Xp -Xn];

n = length(wk);
k = 0;
gk = feval(gname,wk, Xw);
dk = -gk;
er = norm(gk);

while er >= epsi
    ak = bt_lsearch2019(wk,dk,fname,gname, Xw);
    wk_new = wk + ak*dk;  
    gk_new = feval(gname,wk_new, Xw);
    gmk = gk_new - gk;
    bk = max((gk_new'*gmk)/(gk'*gk),0);
    dk = -gk_new + bk*dk;
    wk = wk_new;
    gk = gk_new;
    er = norm(gk);
    k = k + 1;
    fk_new = feval(fname,wk,Xw);
    fprintf('iter %1i: loss = %8.2e\n',k, fk_new)
end
% disp('solution:')
Ws = wk;
disp('objective function at solution point:')
fs = feval(fname,Ws, Xw);
format short
disp('number of iterations at convergence:')
k