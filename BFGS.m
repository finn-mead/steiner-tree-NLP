function [xminEstimate, fminEstimate, k] = BFGS( ...
    f, gradf, x0, H0, tolerance1, constraintFuns)
% BFGS   -  BFGS with Armijo backtracking + feasibility check instead of GSS
%
%   [xminEstimate,fminEstimate,k] = BFGS(f, gradf, x0, H0, tol1, tol2, T, constraintFuns)
%     f             - objective, takes row-vector x (1xn)
%     gradf         - gradient, takes row-vector x, returns row-gradient (1xn)
%     x0            - starting point (row vector 1xn)
%     H0            - initial Hessian approx (nxn)
%     tolerance1    - stopping tol for norm of grad
%     tolerance2    - line-search tol (ignored here)
%     constraintFuns - array of handles g_i <= 0
%
%  Each Armijo step will also reject any x_trial that violates g_i(x) < 0

    tic;
    k = 0;
    tinyTol = 1e-11; % identify when armijo step size is no good

    initial = 1.0;   % initial step size, tau
    shrink_factor    = 0.5;
    sigma     = 1e-4;
    maxSteps  = 30;    % max steps to prevent getting stuck forever

    % Initialize
    xk     = x0;   % row vector
    xk_old = xk;
    H_old  = H0;

    while norm(feval(gradf, xk),2) >= tolerance1
        % Compute BFGS search direction
        H_old = H_old / max(max(H_old));   % dont let H be too large
        dk    = - feval(gradf, xk) * H_old;

        fxk  = feval(f, xk);
        gkR  = feval(gradf, xk); % row gradient used in Armijo
        
        % armijo backtracking + feasibility check 
        t = initial;
        for step = 1:maxSteps
            x_trial = xk + t*dk;
            % check feasibility of ALL constraints
            gVals = cellfun(@(h) h(x_trial'), constraintFuns);
            if any(gVals >= 0)
                % infeasible, shrink and retry
                t = t * shrink_factor;
                continue;
            end

            % objective at trial point
            f_trial = feval(f, x_trial);
            % check Armijo condition
            if f_trial <= fxk + sigma * t * (gkR * dk')
                break
            end

            % otherwise shrink
            t = t * shrink_factor;
            if t < tinyTol
                t = 0;
                break
            end
        end

        % if no progress, exit
        if t < tinyTol
            break
        end

        % update iterate and Hessian
        k = k + 1;
        xk_new = xk + t * dk;              
        sk     = (xk_new - xk_old)';       
        gk_new = feval(gradf, xk_new)';    
        yk     = gk_new - feval(gradf, xk_old)';


        % for very small values this form works better
        rho   = 1/(yk'*sk);
        I      = eye(size(H_old));
        H_new  = (I - rho*(sk*yk')) * H_old * (I - rho*(yk*sk')) + rho*(sk*sk');

        % shift for next iter
        xk_old = xk_new;
        xk     = xk_new;
        H_old  = H_new;
    end

    xminEstimate = xk;
    fminEstimate = feval(f, xk);
end