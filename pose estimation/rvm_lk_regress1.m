% Multivariate nonlinear regression under the L_k norm using a
% Bayesian "Relevance Vector Machine" approach, i.e. a sparsifying
% prior on the coefficients that is scale-invariant and hence
% infinitely peaked at zero, so that small / unnecessary coefficients
% are aggressively pruned.
%
% The model fitted is y = A*f(x), where x and y are respectively input
% and output vectors, f() is a vector of nonlinear functions (possibly
% including the constant function, the components of x, etc). The
% coefficient matrix A is estimated by minimizing
%
% logprior(A) + 1/2 sum_i || y_i - A*f(x_i) ||^k
%
% over a set of training samples (x_i,y_i), where ||-||^k denotes the
% kth power of the L_k norm (the sum of kth powers of components).
% Logprior(A) is the -logarithm of an |x|^nu prior on A. Two forms are
% supported: a component-wise version nu*sum_ij log|A_ij|, which
% allows the RVM to select different basis functions to model each
% component of y, and a column-wise version nu*sum_j log||A_*j||,
% where A_*j is the jth column of A, which forces the RVM to choose
% the same set of basis functions for all components of y.
%
% The algorithm is not the one advocated by Tipping et al. Instead,
% the scale parameters (tau or sigma) are integrated out to produce a
% closed form prior, and a continuation method based on local
% convexification of the prior with a quadratic function is used to
% prevent premature trapping of weights at zero. The method maintains
% a running estimate of the norm of each variable or column of
% variables, and at each iteration approximates the nonconvez
% nu*log(norm) prior with a convex quadratic, 0.5*alpha*norm^2 + const,
% with the same derivative, i.e. alpha = nu/norm^2.
%
% The input is the training data Y = [y1,y2...] and
% F=[f(x1),f(x2)...], the norm strength k, the prior strength nu, the
% prior type colwise, and optionally an inital guess for A, and an
% initial scale for the priors. 
%
% If no initial guess for A is supplied, standard least squares
% regression is used to initialize A. However, note that for many
% problems, particularly ill-conditioned or very sparse ones, starting
% with A=0 is often a better strategy.

function [A] = rvm_lk_regress1(Y,F,k,nu,colwise,A,Ascale)

global live_list

verbose = 0;
maxit = 1000;
maxit = 30; % normally takes more iterations to converge, but set to 30 for demo purposes
% maxminor = 10;
[nf,np]= size(F);
[ny,np1] = size(Y);
if np1~=np, error('Incompatible dimensions for Y,F'); end
if verbose
    fprintf(1,'ny=%d nf=%d np=%d\n',ny,nf,np); %fflush(1);
end
% [U,S,V] = svd(F); diag(S)'

% Default initialization: solve Y = A*F in least squares.
if nargin<6, A = []; end
if isempty(A)
   A = Y / F;
end

% Estimate inital scales for continuation (quadratic approximation
% of prior on A). Model: we assume that terms of y=A*f add like
% independent random variables, so each component contributes
% O(1/sqrt(nf)) of the total, and we estimate typical scales for
% rows of f and y as their mean square entries over input
% points. Also, the scale must be at least as large as the inital
% A, if any. [Beware - an ill conditioned solution A=Y/F might make
% this ridiculous].

if nargin<7, Ascale = []; end
if isempty(Ascale)
    Yscale = sum(Y.^2,2)/np;
    Fscale = sum(F.^2,2)/np;
    Yscale = max(Yscale,1e-5*norm(Yscale,1));
    Fscale = max(Fscale,1e-5*norm(Fscale,1));
    Ascale = (Yscale * (max(Fscale,1e-10).^-1)') / sqrt(nf);
    Ascale = max(Ascale,abs(A));    
    % Boost initial working scale to reduce early trapping.
    Ascale = 5*Ascale;    
end
colscale = sqrt(sum(Ascale.^2,1));

% Columnwise method is same as componentwise one, but with
% all scales in column locked to scale of ||A_*k||
if colwise
    Ascale = ones(ny,1)*colscale;
end
if verbose  
    fprintf(1,'0: residual=%.3g live=%d\n',norm(Y-A*F,1),nf); 
    %fflush(1);    
end

% Main loop, continuation over norms of variables, replacing
% nu*log(norm) priors with 0.5*nu*(var/norm)^2 quadratic ones and
% minimizing (which is just solving a linear system for k=2 case).

live = [1:nf];
S = Ascale;
Sprevious = S;
for it = 1:maxit
    A0 = A;
      
    if k == 2
    	F2 = F*F';
	    if (~colwise),
            for i = 1:ny
                % Solve for each row of A independently, including
	            % diagonal Hessian contribution from prior.
	            A(i,:) = ((F2 + diag(nu*S(i,:).^-2)) \ F*Y(i,:)')';
            end
        else
            A = ((F2 + diag(nu*S(1,:).^-2)) \ F*Y')';    
        end    
    else
        error('k~=2 not yet implemented');    
    end

    % Update live functions and run convergence test.  We've 
    % converged if there was no change in live variables and the
    % change in A is relatively small, or if there are no live
    % variables left to play with.

    delta = norm((A-A0)./Ascale,1);
    % [sqrt(sum(A.^2,1));colscale; sqrt(sum(A.^2,1))./colscale]
    ilive = find(sqrt(sum(A.^2,1))./colscale > 1e-6);
    if verbose
        fprintf(1,'%d: delta=%.3g live=%d/%d\n',it,delta,size(ilive,2),size(live,2));
        %fflush(1);
    end

    if size(ilive,2) == size(live,2)
        if delta < 1e-5, break; end         % REMEMBER THE ORIGINAL CONDITION WAS 1e-8    
    elseif size(ilive,2) == 0
        A = zeros(ny,nf);
        return;    
    else
        % Remove unneeded functions for efficiency (i.e. those for
	    % which all coefficients vanish).
	    live = live(ilive);
	    F = F(ilive,:);
	    A = A(:,ilive);
	    Ascale = Ascale(:,ilive);
	    colscale = colscale(ilive);    
    end
    % Update running scale for prior, to norm of new A variables. 
    if colwise
        S = ones(ny,1)*sqrt(sum(A.^2,1));    
    else
        S = abs(A);         % try to take max of this and a 1e-06 matrix to avoid warning in comp-wise case
    end
    %S = max(S,0.50*Sprevious(:,ilive)); % adding this line to control rate of killing  
    Sprevious = S;
    % S = max(S,1e-5*norm(S,1));    
end
   
% Reinsert dead coefficients for return.
A0 = A;
A = zeros(ny,nf);
A(:,live) = A0;   
live_list = live;

%end
