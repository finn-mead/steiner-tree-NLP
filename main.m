% main.m  -  random tests using generateSteinerTopology
clc; clear; rng("shuffle");

nCases     = 300;
epsFeas    = 1e-6;

agreeFeas  = 0;
agreeInfeas= 0;
bfgsFP     = 0;
bfgsFN     = 0;

t = tic;

falseNegs = struct(...
    'caseNum',{},'nTerm',{},'P',{},'bVal',{},...
    'termXY',{},'nodeIDs',{},'edgeList',{},'xGuess',{},...
    'sOptB',{},'sOptF',{},'tBFGS',{});

phase2InfeasCount = 0;

for c = 1:nCases
    fprintf("number %d/%d", c, nCases);
    % pick a random number of terminals
    % nTerm = randi([10 10]);
    nTerm = 10;
    P = nTerm - 2;
    [termXY,nodeIDs,edgeList,xGuess,bVal] = generateSteinerInstance(nTerm, 100, [0.4, 0.5]);

    % Phase-1 via BFGS log-barrier
    tB = tic;
    [xFeas, sOptB, okB] = findFeasibleStart(termXY, edgeList, bVal, P, xGuess, epsFeas);
    tBFGS = toc(tB);

    % --- Phase-1 via fmincon benchmark --
    [objF, conF] = setupPhase1NLP_fmincon(termXY, edgeList, bVal, P);
    y0      = [xGuess; max(0,max(buildGvec(xGuess,termXY,edgeList,bVal,P))) + 1e-3];
    opts    = optimoptions('fmincon','Algorithm','interior-point','Display','off', ...
                           'ConstraintTolerance',1e-7,'StepTolerance',1e-7);

    try
        [yOptF, ~, exitflagF] = fmincon(objF, y0, [],[],[],[],[],[], conF, opts);
    catch
        exitflagF = -1;
        yOptF     = y0;    % fallback to initial guess
    end

    % pull out slack + Steiner coords
    sOptF = yOptF(end);
    xOptF = yOptF(1:end-1);

    % --- robust feasibility check ---
    gValsF   = buildGvec(xOptF, termXY, edgeList, bVal, P);
    maxViolF = max(gValsF);

    % success if slack is negative AND every g_i(x)<0, regardless of exitflagF
    okF = (sOptF < -epsFeas) && (maxViolF < -epsFeas);
    if ~okF && exitflagF>=0 && (sOptF < -epsFeas) && (maxViolF < epsFeas)
        % solver hit iter limit (exitflagF==0) but is actually feasible within tolerance
        fprintf('  Note: fmincon exitflag=%d but max g=%.3e -> treating as feasible\n', ...
                exitflagF, maxViolF);
        okF = true;
    end

    

    % tally Phase-1 agreement
    if     okB && okF,  agreeFeas   = agreeFeas   + 1;
    elseif ~okB && ~okF, agreeInfeas = agreeInfeas + 1;
    elseif okB,         bfgsFP      = bfgsFP      + 1;
    else                 bfgsFN      = bfgsFN      + 1;
    end

    % collect false negatives
    if (~okB && okF)
        fn = struct();
        fn.caseNum  = c;
        fn.nTerm    = nTerm;
        fn.P        = P;
        fn.bVal     = bVal;
        fn.termXY   = termXY;
        fn.nodeIDs  = nodeIDs;
        fn.edgeList = edgeList;
        fn.xGuess   = xGuess;
        fn.sOptB    = sOptB;
        fn.sOptF    = sOptF;
        fn.tBFGS    = tBFGS;
        falseNegs(end+1) = fn;
    end

    fprintf('Case %4d | nT=%3d P=%3d | BFGS_ok=%d  sB=% .2e  tB=%.3fs | ', ...
            c, nTerm, P, okB, sOptB, tBFGS);
    fprintf('fmc_ok=%d  sF=% .2e \n', okF, sOptF);
    fprintf('   current false negatives: %d of %d\n', bfgsFN, c);

    if okB
        % BFGS Phase-2 solver 
        tB2 = tic;
        [xOptB2, fOptB2, ~] = solveSteinerBarrierBFGS(termXY, edgeList, bVal, P, xFeas);
        tB2 = toc(tB2);
        phase2TimesB(c) = tB2;
        phase2ObjsB(c)  = fOptB2;

        %check actual feasibility of xOptB2 %
       gValsB2   = buildGvec(xOptB2, termXY, edgeList, bVal, P);
       maxViolB2 = max(gValsB2);
       if maxViolB2 > epsFeas
          phase2InfeasCount = phase2InfeasCount + 1;
          fprintf(' >> Phase-2 BFGS infeasible on case %d: max-g = %.3e\n', c, maxViolB2);
       end

        % fmincon Phase-2 benchmark, same barrier NLP but using xFeas start
        [ objF2, constraintFuns ] = steinerNLP(termXY, edgeList, bVal, P);
        nonlcon2 = @(x) edgeNonlcon(x, constraintFuns);
        
        opts2 = optimoptions('fmincon', ...
            'Algorithm','interior-point', ...
            'Display','off', ...
            'ConstraintTolerance',1e-7, ...
            'StepTolerance',1e-7, ...
            'SpecifyObjectiveGradient',false);
        
        try
            tF2 = tic;
            x0_fmc2 = zeros(2*P, 1);
            [ xOptF2, fOptF2, exitflagF2 ] = ...
                fmincon(objF2, x0_fmc2, [],[],[],[],[],[], nonlcon2, opts2);
            tF2 = toc(tF2);
        catch ME
            warning("Phase-2 fmincon failed: %s", ME.message);
            xOptF2 = x0_fmc2;
            fOptF2 = objF2(xFeas);
            tF2    = NaN;
        end

        fprintf(  '\nPhase-2 comparison:\n' );
        fprintf(  '  BFGS:     obj = %.6g   time = %.3fs\n', fOptB2, tB2 );
        fprintf(  '  fmincon:  obj = %.6g   time = %.3fs\n', fOptF2, tF2 );
    
        % Optional sanity check:
        if fOptB2 > fOptF2 + 1e-8
            fprintf('  >> Warning: BFGS obj is %.3g higher than fmincon!\n\n', fOptB2 - fOptF2);
        else
            fprintf('  (OK: within tolerance)\n\n');
        end
        
        phase2TimesF(c) = tF2;
        phase2ObjsF(c)  = fOptF2;
        
    end
    % if okB
    %     figure('Color','w');
    %     hold on; axis equal; grid on;
    %     title('Steiner Tree Solution','FontSize',14);
    % 
    %     % Plot terminals
    %     hT = plot(termXY(:,1), termXY(:,2), 'ro', ...
    %               'MarkerSize',8, 'LineWidth',2);
    % 
    %     % Extract and plot Steiner points
    %     S = reshape(xOptB2, 2, []).';   % Px2 matrix
    %     hS = plot(S(:,1), S(:,2), 'bs', ...
    %               'MarkerSize',8, 'LineWidth',2);
    % 
    % 
    %     % Draw edges
    %     for e = 1:size(edgeList,1)
    %         % get coords for endpoints
    %         aID = edgeList(e,1);
    %         bID = edgeList(e,2);
    %         if aID < 0, A = termXY(-aID,:);
    %         else        A = S(aID,:);    end
    %         if bID < 0, B = termXY(-bID,:);
    %         else        B = S(bID,:);    end
    % 
    %         plot([A(1), B(1)], [A(2), B(2)], '-k', 'LineWidth',1.5);
    %     end
    % 
    % 
    %     legend([hT hS], {'Terminals','Steiner pts'}, 'Location','best');
    %     xlabel('x'); ylabel('y');
    %     hold off;
    % end
    %% ——— Debug dump of entire problem ———
    fprintf("\n=== Problem %d/%d — nTerm=%d, P=%d, bVal=%.6g ===\n", c, nCases, nTerm, P, bVal);
    fprintf("terminalCoords (termXY):\n");    disp(termXY);
    fprintf("nodeIDs:\n");                   disp(nodeIDs);
    fprintf("edgeList (each row = [a b]):\n"); disp(edgeList);
    fprintf("initial guess (xGuess):\n");    disp(xGuess);
    fprintf("epsFeas = %.1e\n\n", epsFeas);
end



% --- print all false-negatives at once -----------------------------------
if isempty(falseNegs)
    fprintf('No false-negative cases.\n');
else
    fprintf('Detailed false-negative cases:\n\n');
    for i = 1:numel(falseNegs)
        fn = falseNegs(i);
        fprintf('%% ==================== Case %d ====================\n', fn.caseNum);
        fprintf('%% nTerm = %d, P = %d, bVal = %.6g\n', fn.nTerm, fn.P, fn.bVal);

        fprintf('%% termXY = [\n');
        fprintf('%%   %.6g   %.6g\n', fn.termXY');
        fprintf('%% ];\n');

        fprintf('%% nodeIDs = [ ');
        fprintf('%d ', fn.nodeIDs);
        fprintf('];\n');

        fprintf('%% edgeList = [\n');
        fprintf('%%   %3d   %3d\n', fn.edgeList');
        fprintf('%% ];\n');

        if isempty(fn.xGuess)
            fprintf('%% xGuess = []\n');
        else
            fprintf('%% xGuess = [\n');
            fprintf('%%   %.6g\n', fn.xGuess);
            fprintf('%% ];\n');
        end

        fprintf('%% sOptB = %.6g, sOptF = %.6g, tBFGS = %.3fs\n\n',...
                fn.sOptB, fn.sOptF, fn.tBFGS);
    end
end

fprintf('\nSummary over %d cases:\n', nCases);
fprintf('  agree feasible   : %d\n', agreeFeas);
fprintf('  agree infeasible : %d\n', agreeInfeas);
fprintf('  false + (BFGS)   : %d\n', bfgsFP);
fprintf('  false - (BFGS)   : %d\n', bfgsFN);
fprintf('TIME: %d\n', toc(t));
fprintf('\nPhase-2 BFGS infeasible count: %d of %d cases\n', phase2InfeasCount, nCases);