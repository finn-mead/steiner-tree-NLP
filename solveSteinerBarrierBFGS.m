% solveSteinerBarrierBFGS.m (Corrected to use steinerNLP output)
% Solves Steiner tree via log-barrier + BFGS.
%
% Inputs:
%   termXY   Nx2 terminals
%   edgeList Mx2 connections (neg=terminal, pos=Steiner)
%   b        scalar max edge length
%   P        # of Steiner points
%   x0       initial 2Px1 guess (column, must be strictly feasible)
%   gradTol  gradient norm tolerance (default 1e-6)
%   outerTol outer-loop tol (default 1e-5)
%   maxOuter max outer iterations (default 8)
%
% Outputs:
%   xOpt    optimal Steiner coords (column)
%   fOpt    objective at xOpt
%   nOuter  # of barrier outer loops executed

function [xOpt,fOpt,nOuter] = solveSteinerBarrierBFGS(termXY,edgeList,b,P,x0,gradTol,lineTol,outerTol, maxOuter)

    if nargin<6,  gradTol  = 1e-4; end % occasional crashing for smaller tolerances
    if nargin<7,  outerTol = 1e-5; end % Tolerance for norm(x_new - x_old)
    if nargin<8,  maxOuter = 100;   end % rarely needs more than a handful of iterations

    feasTol = 1e-10;

    % fprintf("STARTING AT: ");
    % disp(x0(1:6));
    % 
    % dbgG0 = buildGvec(x0, termXY, edgeList, b, P);
    % fprintf('[Phase-2] max g at start = %.3e\n', max(dbgG0));

    % 1. Get base objective and constraint function handles from steinerNLP
    %    original_objFun expects/returns column x.
    %    original_constraintFuns is a cell array of handles, each expects/returns column x.
    [original_objFun, original_constraintFuns] = steinerNLP(termXY, edgeList, b, P);

    checkG  = @(xcol) cellfun(@(h) h(xcol), original_constraintFuns);

    if any(checkG(x0) >= 0)
        fprintf("INFEASIBLE START!\n");
        error("Infeasible Start");
    end

    % theoretically shouldnt need this because armijo implementation
    % already explicitly checks constraints. but to be safe...
    function v = safeLogBarrier(xCol, alpha, objHandle)
        gVals = checkG(xCol);
        if any(gVals >= -feasTol)
            fprintf(1, ...
              '[safeLogBarrier] Infeasible!  x = [%s],  max g = %g\n', ...
              num2str(xCol','% .4g '), max(gVals));
            v = Inf;       % force line‐search to reject
            return;
        end
        v = objHandle(xCol) - (1/alpha)*sum(log(-gVals));
    end

    % 2. Initialize Log-Barrier Loop
    xk_col = x0(:); % Ensure x0 is a column vector
    alpha  = 10;    % Initial alpha
                    

    xk_prev_outer_col = xk_col; % For checking outer loop convergence
    total_bfgs_iters = 0;

    fprintf('Starting log-barrier + BFGS (Phase 2)\n');

    for nOuter = 1:maxOuter

        % Penalized objective for BFGS (expects row vector input, returns scalar)
        % original_objFun and original_constraintFuns expect column vectors.
        penObj  = @(xr_bfgs) safeLogBarrier(xr_bfgs', alpha, original_objFun);

        % Gradient of penalized objective for BFGS (expects row vector input, returns row vector gradient)
        % gradBarrierObjective expects column vector input, returns column vector gradient.
        penGrad = @(xr_bfgs) gradBarrierObjective(xr_bfgs', termXY, edgeList, b, alpha, P)';

        H0 = eye(2*P);
        
        
        % scale gradient tolerance dynamically.
        % idea: as alpha increases and we can get closer to the barrier,
        % we can ask for a greater degree of precision
        gradTol_bfgs = min(gradTol, 1/alpha);

        current_bfgs_iters = 0;

        [xk_row_new, f_penalized_val, current_bfgs_iters] = BFGS(...
            penObj, ...
            penGrad, ...
            xk_col', ... % Pass current xk as a ROW vector
            H0, ...
            gradTol, ...
            original_constraintFuns);
            

        xk_col = xk_row_new';
        
        total_bfgs_iters = total_bfgs_iters + current_bfgs_iters;
        
        % Check convergence of the outer log-barrier loop

        % first, explicit feasibility check
        gVals   = cellfun(@(g) g(xk_col), original_constraintFuns);
        maxViol = max(gVals);

        fprintf('  alpha=%g  maxViol=%.2e\n', alpha, maxViol);

        % outer loop stops only if constraints strictly satisfied AND small gradient
        if (maxViol <= feasTol) && (norm(penGrad(xk_col')) <= gradTol_bfgs)
            fprintf('  Converged: Feasible and stationary solution found.\n');
            break;
        end
        
        xk_prev_outer_col = xk_col; % Update for next iteration's comparison
        
        % increase alpha
        alpha  = alpha * 2;
    end

    xOpt = xk_col;
    fOpt = original_objFun(xOpt);
    
    fprintf('Log-Barrier BFGS Done: outer iters = %d, total BFGS iters = %d, fOpt = %.6f\n', ...
            nOuter, total_bfgs_iters, fOpt);
end
