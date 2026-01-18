%
%  BFGS with optional early-stop callback.
%
%  the same interface as the original BFGS plus one extra argument:
%      stopFcn   - handle  @(x) logical , evaluated each main iteration.
%                   If true, the loop terminates immediately.
%   also has some numerical tweaks to help get unstuck
%
function [xminEstimate, fminEstimate, k] = ...
             bfgsEarlyStop(f, gradf, x0, H0, tol1, tol2, T, stopFcn)

maxTime = 60;
tStart = tic;                                   %starts timer
k  = 0;                                %initialize iteration counter
xk = x0;  xk_old = x0;  H_old = H0;

tinyTol = 1e-13;
normTol = 1e-3;

while ( norm(feval(gradf,xk)) >= tol1 )
    % EARLY-STOP ----------------------------------------------------------
    if nargin>7 && stopFcn(xk), break; end
    % --------------------------------------------------------------------
    % in degenerate cases where t > 30 seconds, abort
    if toc(tStart) > maxTime
        fprintf("Aborting BFGS, time limit %d seconds reached.\n", maxTime);
        break;
    end
    % H_old = H_old / max(max(H_old));         %prevent blow-up
    dk    = transpose(-H_old*transpose(feval(gradf,xk)));   %search dir
    
    [a,b]      = multiVariableHalfOpen(f,xk,dk,T);          %bounds
    [tmin,~]   = multiVariableGoldenSectionSearch( ...
                     f,a,b,tol2,xk,dk);                     %line-search
    
    % modified to have an early stop if t is near 0
    if abs(tmin) < tinyTol
        if norm(feval(gradf,xk)) < normTol
            fprintf('tiny step & small grad, stop\n');
            break                      % convergence
        else
            fprintf('tiny step only,  reset H, retry\n');
            H_old = eye(size(H_old));  % restart
            T     = T*2;               % widen bracket
            continue                   % go back to while-loop
        end
    end

    k    = k+1;

    % resetting the hessian every so often helps get unstuck
    if mod(k,200)==0, H_old = eye(size(H_old)); end


    xk   = xk + tmin*dk;   xk_new = xk_old + tmin*dk;
    
    sk   = (xk_new-xk_old)';                                %BFGS update
    gk   = (feval(gradf,xk_new)-feval(gradf,xk_old))';
    rk   = (H_old*gk)/(sk'*gk);
    H_new= H_old + (1+rk'*gk)/(sk'*gk)*(sk*sk') - (sk*rk') - (rk*sk');
    
    xk_old = xk_new;   H_old = H_new;
end

xminEstimate = xk;
fminEstimate = feval(f,xk);
end