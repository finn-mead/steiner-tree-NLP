% findFeasibleStart  -  Phase-1 feasibility via log-barrier + early-stop BFGS
% highly sensitive to parameter values, strongly recommend not touching them.
% -------------------------------------------------------------------------
function [xStrictFeas, sOpt, success] = ...
    findFeasibleStart(terminalCoords, edgeList, bOriginal, numSteinerPoints, ...
                      xGuess, epsFeas)

fprintf('  Phase 1 (BFGS): searching for strictly feasible x ...\n');

% Starting slack so (xGuess,s0) is strictly feasible by taking s = the max
% constraint violation
g0 = buildGvec(xGuess, terminalCoords, edgeList, bOriginal, numSteinerPoints);
s0 = max(0, max(g0)) + 0.001;  % small margin to prevent log(0)

% introduce a constraint that s <= s0 (initial s) to make solver behave
epsBound = 0.001;  % another slim margin to prevent log(0)
s_upper_bound = s0 + epsBound;

y0 = [xGuess ; s0];
y0_row = y0';

% experimentally, alpha = 10 has been giving more consistent results
% dont need to increase alpha and see what it converges to, any feasible
% point is good enough and we can exit early
alpha = 10

% Build objective and gradient
[objFun, gradFun, gVecFun] = setupPhase1NLP( ...
        terminalCoords, edgeList, bOriginal, numSteinerPoints, alpha, s_upper_bound);

% make wrappers for bfgs implementation which uses row vectors not column
fPhase1_for_bfgs = @(y_row) objFun(y_row');
gPhase1_for_bfgs = @(y_row) transpose(gradFun(y_row'));

% BFGS settings ------------------------------------------------------------
H0      = eye(numel(y0_row));
tol1    = 1e-6;
tol2    = 1e-13;
Tline   = 2;
stopFcn = @(y) y(end) < -epsFeas;          % early stop when s < -epsFeas (roughly zero)

% Run BFGS ------------------------------------------------------------------
[yOpt_row, ~, ~] = bfgsEarlyStop(fPhase1_for_bfgs, gPhase1_for_bfgs, y0_row, H0, ...
                             tol1, tol2, Tline, stopFcn);

sOpt       = yOpt_row(end);
success    = (sOpt < -epsFeas);

% sometimes the minimum violates constraints, manual check.
gVals   = gVecFun(yOpt_row(1:end-1)');            % all g_i(x) at final x
maxViol = max(gVals);
if maxViol > -epsFeas
    success = false;          % if any edge still long
end

% if its invalid set to zero vector as a flag (backup to success flag)
xStrictFeas = success * yOpt_row(1:2*numSteinerPoints)';

% Report -------------------------------------------------------------------
if success
    fprintf('    SUCCESS: s = %.4g < %.1e\n', sOpt, -epsFeas);
else
    fprintf('    FAIL: final s = %.4g (not strictly negative)\n', sOpt);
end
fprintf('  Phase 1 done.\n');
end
