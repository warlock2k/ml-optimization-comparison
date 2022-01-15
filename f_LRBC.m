function f = f_LRBC(w,X)
P = size(X,2);
f = sum(log(1+exp(-X'*w)))/P;