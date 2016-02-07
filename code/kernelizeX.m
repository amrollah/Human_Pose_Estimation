%
% usage: F = kernelizeX(X,Xbases)
%
% X: dxn matrix of n data points, Xbases: dxp matrix of p training data points
%
% warning: this function contains some hard-coded parameters. It is
% intended to be used only for kernelizing the 100D shape descriptors X1
% and X2.

function F = kernelizeX(X,Xbases)

F = L2_distance(Xbases,X);

beta = 0.1;
variance = 100.0092;
F = exp(-beta*(F.*F)/variance);
