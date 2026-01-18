function obj_vs_bTightness()
    % OBJ_VS_BTIGHTNESS  Compare BFGS vs fmincon for varying b‐tightness.
    rng(2);
    
    % problem & experiment params
    nT    = 20;      p = nT-2;
    Nwant = 100;      % desired valid instances
    fac   = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5];  
    L     = numel(fac);
    tolF  = 1e-9;    
    maxGen = Nwant*100;
    
    % storage for relative errors
    relErr      = nan(L,Nwant);
    % Phase-1 counters: [agreeFeas, agreeInf, falseNeg, falsePos]
    p1cnt       = zeros(L,4);
    % Phase-2 BFGS counters
    p2bfgs_run  = zeros(L,1);
    p2bfgs_fail = zeros(L,1);
    % Phase-2 fmincon counters
    p2fmc_run   = zeros(L,1);
    p2fmc_fail  = zeros(L,1);
    
    fprintf('== OBJ VS B-TIGHTNESS (nT=%d) ==\n',nT);
    
    %  build Nwant valid bases using slack-based B_min
    bases = cell(Nwant,1); 
    V = 0; 
    tries = 0;
    while V < Nwant && tries < maxGen
        tries = tries + 1;
        [T,~,E,x0,b0] = generateSteinerInstance(nT,100,[0.8,1.5]);
        [sF, xF, okF]  = p1_fmin(T,E,b0,p,x0,tolF);
        if ~okF, continue; end
        % compute B_min from slack:
        B_min = sqrt( max(0, b0^2 + sF) );
        if B_min <= 1e-9, continue; end
        V = V + 1;
        bases{V} = struct('T',T,'E',E,'x0',x0,'B',B_min);
    end
    if V < Nwant
        warning('Only %d/%d valid bases generated.', V, Nwant);
    end
    bases = bases(1:V);
    fprintf('  %d valid bases generated.\n', V);
    
    %% 2) Sweep through b‐tightness levels
    for k = 1:L
        for i = 1:V
            d = bases{i};
            btest = max(d.B*(1+fac(k)),1e-4);
            
            %---- Phase-1 comparison ----
            [okB, okF, xB] = phase1Eval(d.T,d.E,btest,p,d.x0,tolF);
            p1cnt(k,:) = p1cnt(k,:) + [okB&&okF, ~okB&&~okF, ~okB&&okF, okB&&~okF];
            
            %---- Phase-2 BFGS ----
            ok2 = false;
            if okB && ~isempty(xB)
                p2bfgs_run(k) = p2bfgs_run(k) + 1;
                [fB, ok2] = phase2BFGS(d.T,d.E,btest,p,xB,tolF);
                if ~ok2
                    p2bfgs_fail(k) = p2bfgs_fail(k) + 1;
                end
            else
                fB = NaN;
            end
            
            % ---- Phase-2 fmincon with debug ----
p2fmc_run(k) = p2fmc_run(k) + 1;

% Set up the NLP for fmincon
[obj2, cn2] = steinerNLP(d.T, d.E, btest, p);
nonl2       = @(x) edgeNonlcon(x, cn2);
x0_fmc      = zeros(2*p,1);
opts2       = optimoptions('fmincon', ...
                 'Display','off', ...
                 'ConstraintTolerance',1e-7);

try
    [xF2, fF2, efF2] = fmincon(obj2, x0_fmc, [],[],[],[],[],[], nonl2, opts2);
    gF2 = buildGvec(xF2, d.T, d.E, btest, p);
    okF2 = (efF2 >= 0) && all(gF2 < -tolF);

    if ~okF2
        % debug print
        fprintf('[DEBUG P2-fmin] level %d, instance %d: exitflag=%d, max(g)=%.4g, obj=%.4g\n', ...
                k, i, efF2, max(gF2), fF2);
        p2fmc_fail(k) = p2fmc_fail(k) + 1;
    end

catch ME
    fprintf('[DEBUG P2-fmin] level %d, instance %d CRASH: %s\n', ...
            k, i, ME.message);
    okF2 = false;
    fF2  = NaN;
    p2fmc_fail(k) = p2fmc_fail(k) + 1;
end

% finally assign into loop variables
fF   = fF2;
okF2 = okF2;
            
            %---- Collect relative error if both succeeded ----
            if ok2 && okF2 && abs(fF) > eps
                relErr(k,i) = 100*(fB - fF)/abs(fF);
            end
        end
    end
    
    %% 3) Plot Avg % difference
    xs = fac*100;                     
    mu = mean(relErr,2,'omitnan');    
    figure;
    plot(xs, mu, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    yline(0, '--k', 'LineWidth', 1);
    xlabel('b relaxation (%)');
    ylabel('Avg % difference (BFGS vs fmincon)');
    title('Avg. % increase in objective against b tightness');
    grid on;
    
    %% 4) Print summary tables
    fprintf('\nP1 lvl | agreeF | agreeI | falseNeg | falsePos\n');
    for k=1:L
        fprintf('%4d | %6d | %6d | %8d | %8d\n', k, p1cnt(k,:));
    end
    
    fprintf('\nP2-BFGS lvl | runs | fails | success rate\n');
    for k=1:L
        runs = p2bfgs_run(k);
        fails= p2bfgs_fail(k);
        if runs > 0
            sr = 100*(runs-fails)/runs;
        else
            sr = NaN;
        end
        fprintf('%4d       | %4d | %5d | %7.2f%%\n', k, runs, fails, sr);
    end
    
    fprintf('\nP2-FMC lvl | runs | fails | success rate\n');
    for k=1:L
        runs = p2fmc_run(k);
        fails= p2fmc_fail(k);
        if runs > 0
            sr = 100*(runs-fails)/runs;
        else
            sr = NaN;
        end
        fprintf('%4d       | %4d | %5d | %7.2f%%\n', k, runs, fails, sr);
    end
end


%% --- helpers ---

function [s,x,ok] = p1_fmin(T,E,b,p,x0,tolF)
    [f1,c1] = setupPhase1NLP_fmincon(T,E,b,p);
    x0 = x0(:);
    g0 = buildGvec(x0,T,E,b,p);
    y0 = [x0; max(0,max(g0(~isnan(g0))))+1e-3];
    opts = optimoptions('fmincon','Display','off','ConstraintTolerance',1e-7);
    ok = false; s = Inf; x = [];
    try
        [y,~,ef] = fmincon(f1,y0,[],[],[],[],[],[],c1,opts);
        % accept ef>=0 (converged or hit limit) if point is strictly feasible
        if ef >= 0 && y(end) < -tolF
            xc = y(1:end-1);
            if all(buildGvec(xc,T,E,b,p) < -tolF)
                ok = true; s = y(end); x = xc;
            end
        end
    catch
        % nothing
    end
end

function [okB,okF,xB] = phase1Eval(T,E,b,p,x0,tolF)
    % BFGS Phase-1
    [xB,sB,okB] = findFeasibleStart(T,E,b,p,x0,tolF);
    if okB && any(buildGvec(xB,T,E,b,p) >= -tolF)
        okB = false; 
        xB  = [];
    end
    
    % fmincon Phase-1
    [sF,xF,okF] = p1_fmin(T,E,b,p,x0,tolF);
    if ~okF, xF = []; end
end

function [f,ok] = phase2BFGS(T,E,b,p,x0,tolF)
    f = NaN; ok = false;
    if isempty(x0), return; end
    try
        [x,f0,~] = solveSteinerBarrierBFGS(T,E,b,p,x0);
        if all(buildGvec(x,T,E,b,p) < -tolF)
            ok = true; f = f0;
        end
    catch
    end
end

function [f,ok] = p2_fmin0(T,E,b,p)
    [obj,cn] = steinerNLP(T,E,b,p);
    nonl     = @(x) edgeNonlcon(x,cn);
    opts     = optimoptions('fmincon','Display','off','ConstraintTolerance',1e-7);
    f = NaN; ok = false;
    try
        [x,f0,ef] = fmincon(obj,zeros(2*p,1),[],[],[],[],[],[],nonl,opts);
        if ef >= 0 && all(buildGvec(x,T,E,b,p) < -1e-6)
            ok = true; f = f0;
        end
    catch
    end
end