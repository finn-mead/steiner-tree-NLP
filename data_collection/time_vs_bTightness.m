function time_vs_bTightness()
    % TIME_VS_BTIGHTNESS  Compare BFGS (P1+P2) vs fmincon (P2 from zeros)
    % for varying b-tightness, recording runtimes and failure rates.

    rng(4);
    nT      = 20;
    p       = nT - 2;
    Nwant   = 100;
    factors = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5];
    L       = numel(factors);
    tolF    = 1e-9;

    % Preallocate
    tB    = nan(L,Nwant);   % BFGS times (P1+P2)
    tF    = nan(L,Nwant);   % fmincon times (P2 only)
    runB  = zeros(L,1);     failB = zeros(L,1);
    runF  = zeros(L,1);     failF = zeros(L,1);

    fprintf('== TIME VS B-TIGHTNESS (nT=%d) ==\n', nT);

    % 1) Build Nwant valid base problems via slack-based B_min
    bases = cell(Nwant,1); V=0; tries=0;
    while V<Nwant && tries<5*Nwant
        tries = tries + 1;
        [T,~,E,x0,b0] = generateSteinerInstance(nT,100,[0.8,1.5]);
        [sF, xF, okF] = p1_fmin(T,E,b0,p,x0,tolF);
        if ~okF, continue; end
        Bmin = sqrt(max(0, b0^2 + sF));
        if Bmin <= 1e-9, continue; end
        V = V + 1;
        bases{V} = struct('T',T,'E',E,'x0',x0,'B',Bmin);
    end
    bases = bases(1:V);
    fprintf('  %d valid bases generated.\n', V);

    % 2) Sweep through b-tightness factors
    for k = 1:L
        for i = 1:V
            d = bases{i};
            btest = max(d.B*(1+factors(k)),1e-4);

            % ---- BFGS pipeline timing ----
            runB(k) = runB(k) + 1;

            % Phase-1 via phase1Eval (returns xB only if okB truly succeeded)
            tic1 = tic;
            [okB, ~, xB] = phase1Eval(d.T, d.E, btest, p, d.x0, tolF);
            t1 = toc(tic1);

            % Phase-2 only if P1 succeeded & xB nonempty
            if okB && ~isempty(xB)
                tic2 = tic;
                [xOptB, ~, ~] = solveSteinerBarrierBFGS(d.T, d.E, btest, p, xB);
                t2 = toc(tic2);
                % check feasibility of the returned xOptB
                okP2 = all(buildGvec(xOptB, d.T, d.E, btest, p) < -tolF);
            else
                t2 = NaN;
                okP2 = false;
            end

            if okB && okP2
                tB(k,i) = t1 + t2;
            else
                failB(k) = failB(k) + 1;
            end

            % ---- fmincon Phase-2 timing (from zeros) ----
            runF(k) = runF(k) + 1;
            ticF = tic;
            [objF, cnF] = steinerNLP(d.T, d.E, btest, p);
            nlF = @(x) edgeNonlcon(x, cnF);
            optsF = optimoptions('fmincon','Display','off','ConstraintTolerance',1e-7);
            try
                [xF2, ~, efF2] = fmincon(objF, zeros(2*p,1), [],[],[],[],[],[], nlF, optsF);
                tFval = toc(ticF);
                okF2 = (efF2 >= 0) && all(buildGvec(xF2, d.T, d.E, btest, p) < -tolF);
            catch
                tFval = toc(ticF);
                okF2 = false;
            end

            if okF2
                tF(k,i) = tFval;
            else
                failF(k) = failF(k) + 1;
            end
        end
    end

    % 3) Plot
    xs  = factors * 100;
    muB = mean(tB, 2, 'omitnan');
    muF = mean(tF, 2, 'omitnan');

    figure; hold on;
      plot(xs, muB, '-o', 'LineWidth',1.5, 'DisplayName','BFGS (P1+P2)');
      plot(xs, muF, '-s', 'LineWidth',1.5, 'DisplayName','fmincon (P2)');
    hold off;
    set(gca,'YScale','log');
    xlabel('b relaxation (%)');
    ylabel('Avg. time (s)');
    title(sprintf('Time vs. b-Tightness (nT=%d)', nT));
    legend('Location','NorthWest');
    grid on;

    % 4) Print failure tables
    fprintf('\nBFGS pipeline failures:\n lvl | runs | fails | fail(%%)\n');
    for k=1:L
      fprintf('%3d | %4d | %5d | %6.2f\n', k, runB(k), failB(k), 100*failB(k)/runB(k));
    end
    fprintf('\nfmincon P2 failures:\n lvl | runs | fails | fail(%%)\n');
    for k=1:L
      fprintf('%3d | %4d | %5d | %6.2f\n', k, runF(k), failF(k), 100*failF(k)/runF(k));
    end
end


% --- helpers ---

function [okB, okF, xB] = phase1Eval(T,E,b,p,x0,tolF)
    % BFGS Phase-1
    [xB,sB,okB] = findFeasibleStart(T,E,b,p,x0,tolF);
    if okB && any(buildGvec(xB,T,E,b,p) >= -tolF)
        okB = false; xB = [];
    end

    % fmincon Phase-1 
    [sF, xF, okF] = p1_fmin(T,E,b,p,x0,tolF);
    % flag when its not feasible
    if ~okF, xF = []; end
end

function [s,x,ok] = p1_fmin(T,E,b,p,x0,tolF)
    % fmincon Phase-1 for B_min generation; return slack s and x only
    [f1,c1] = setupPhase1NLP_fmincon(T,E,b,p);
    x0 = x0(:);
    g0 = buildGvec(x0,T,E,b,p);
    y0 = [x0; max(0,max(g0(~isnan(g0))))+1e-3];
    opts = optimoptions('fmincon','Display','off','ConstraintTolerance',1e-7);
    ok=false; s=Inf; x=[];
    try
        [y,~,ef] = fmincon(f1,y0,[],[],[],[],[],[],c1,opts);
        if ef>=0 && y(end)<-tolF
            xc = y(1:end-1);
            if all(buildGvec(xc,T,E,b,p)<-tolF)
                ok=true; s=y(end); x=xc;
            end
        end
    catch
    end
end