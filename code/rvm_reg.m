%
% Relevance Vector Machine Regresion
%
% usage: A = rvm_reg(Y,F,nu)
% solves Y = A.F using nu as a pruning term
%
% (Note that this is a simple special case of rvm_lk_regress)

function A = rvm_reg(Y,F,nu)

A = rvm_lk_regress1(Y,F,2,nu,1,[],[]);