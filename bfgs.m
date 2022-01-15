% To implement BFGS algorithm.
% Example:
% [xs,fs,k] = bfgs('f200','g200',[zeros(100,1);-ones(100,1)],1e-7);
function [Ws,fs,k] = bfgs(fname,gname,wk, X,y, iter)
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
I = eye(n);
k = 1;
Sk = I;
fk = feval(fname,wk,Xw);
gk = feval(gname,wk,Xw);
dk = -Sk*gk;
ak = bt_lsearch2019(wk,dk,fname,gname,Xw);
dtk = ak*dk;
wk_new = wk + dtk;
fk_new = feval(fname,wk_new,Xw);

while k < iter
      gk_new = feval(gname,wk_new,Xw);
      gmk = gk_new - gk;
      D = dtk'*gmk;
      if D <= 0
         Sk = I;
      else
         sg = Sk*gmk;
         sw0 = (1+(gmk'*sg)/D)/D;
         sw1 = dtk*dtk';
         sw2 = sg*dtk';
         Sk = Sk + sw0*sw1 - (sw2'+sw2)/D;
      end
      fk = fk_new;
      gk = gk_new;
      wk = wk_new;
      dk = -Sk*gk;
      ak = bt_lsearch2019(wk,dk,fname,gname,Xw);
      dtk = ak*dk;
      wk_new = wk + dtk;
      fk_new = feval(fname,wk_new,Xw);
      k = k + 1;
      fprintf('iter %1i: loss = %8.2e\n',k,fk_new)
end
disp('solution:')
Ws = wk_new
disp('objective function at solution point:')
fs = feval(fname,Ws,Xw)
format short
disp('number of iterations at convergence:')
k