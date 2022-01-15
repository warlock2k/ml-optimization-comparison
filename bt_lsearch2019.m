%  Program: bt_lsearch2019.m
%  Implements line learch by backtracking.
%  Description: Implements inexact line search described in 
%  Reference: Sec. 9.2 of Boyd and Vanderberghe's book.
%  Input:
%     x:  initial point
%     d:  search direction
% fname:  objective function to be minimized along the direction of s  
% gname:  gradient of the objective function.
%    p1:  user-defined parameter vector whose components mast have been 
%         numerically specified. The order in which the components of p1 
%         appear must be the same as what they appear in fname and gname.
%         Note: p1 is an optional input.
%    p2:  2nd user-defined parameter vector whose components mast have been 
%         numerically specified. The order in which the components of p2 
%         appear must be the same as what they appear in fname and gname.
%         Note: p2 is an optional input.
% Output:
%     a:  acceptable value of alpha.
% Written by W.-S. Lu, University of Victoria.
% Last modified: July 28, 2019.
function a = bt_lsearch2019(x,d,fname,gname,p1,p2)
rho = 0.1;
gma = 0.5;
x = x(:);
d = d(:);
a = 1;
xw = x + a*d;
parameterstring ='';
if nargin == 5
   if ischar(p1)
      eval([p1 ';']);
   else
      parameterstring = ',p1';
   end
end
if nargin == 6
   if ischar(p1)
      eval([p1 ';']);
   else
      parameterstring = ',p1';
   end
   if ischar(p2)
      eval([p2 ';']);
   else
      parameterstring = ',p1,p2';
   end
end
eval(['f0 = ' fname '(x' parameterstring ');']);
eval(['g0 = ' gname '(x' parameterstring ');']);
eval(['f1 = ' fname '(xw' parameterstring ');']);
t0 = rho*(g0'*d);
f2 = f0 + a*t0;
er = f1 - f2;
while er > 0
     a = gma*a;
     xw = x + a*d;
     eval(['f1 = ' fname '(xw' parameterstring ');']);
     f2 = f0 + a*t0;
     er = f1 - f2;
end
if a < 1e-5
   a = min([1e-5, 0.1/norm(d)]); 
end 