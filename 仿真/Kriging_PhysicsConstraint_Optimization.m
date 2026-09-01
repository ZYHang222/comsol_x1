%% Kriging_PhysicsConstraint_Optimization
% 在已有 Local Ordinary Kriging 高分辨率三分量结果的基础上，执行物理约束优化。
% 优化变量：Kriging 输出分辨率网格上的 [Bx, By, Bz]，初值严格来自 Kriging 输出。
%
% 对照实验包含三组结果：Kriging 初值、标准一阶 Tikhonov 正则化、物理约束重建。
% 数学正则化：L_math = (wPrior/2)*||B-B_kriging||_F^2
%                  + (wSample/2)*||S(B)-B_sample||_F^2
%                  + (lambdaTik/2)*||grad(B)||_F^2。
% 物理约束：  L_physics = L_math + (lambdaCurl/2)*||dBy/dx-dBx/dy||_F^2。
% Lprior：二次 Kriging 先验保真项，保持 Kriging 的宏观空间结构；
% Lsample：原始低分辨率采样点保真项，保证后处理不偏离真实观测；
% LTik  ：标准一阶 Tikhonov 正则，即三个磁场分量空间一阶导数的平方和；
% Lcurl ：二维可观测无旋残差 |dBy/dx-dBx/dy|，来自无自由电流区域的 curl(B)=0。
%
% 重要物理说明：当前输入仅有一个 z 平面，无法直接得到 Bzz=dBz/dz，
% 因而不强行采用 Bxx+Byy+Bzz=0 或假设 Bzz=0。二维无旋残差只涉及 Bx、By 的
% 平面导数，适用于测量平面位于无自由电流区域的磁异常实验。

%% ==================== 1. 用户参数 ====================
clear;
clc;
close all;

% 直接读取主重建脚本导出的 Kriging TXT，避免重复执行耗时的 Local Ordinary Kriging。
% 可指定任意已有重建分辨率，例如 100×100、500×500、1000×1000 或 2000×2000。
% 请在主脚本运行后，将此路径改为对应结果目录内的 Kriging_重建数据.txt。
krigingResultFile =  'D:\YAN\大论文\x1\仿真\RBF_results_10-10_100x100\Kriging_重建数据.txt';
% 原始低分辨率采样文件，仅用于检验高分辨率结果是否仍贴合真实观测点。
% 使用 10×10 重建时填写 10-10.txt；使用 100×100 重建时填写 100-100.txt。
% 它作为 Lsample 数据保真项参与两种优化；Ground Truth 始终只用于结果评价。
originalSampleFile = 'D:\YAN\大论文\x1\数据\10-10.txt';
groundTruthFile = 'D:\YAN\大论文\x1\数据\1000-1000.txt';

saveFigures = true;
figureResolution = 300;

% 标准一阶 Tikhonov / Physics Constraint 参数。PCG 直接求解二次正规方程，
% 不使用 TV 的非线性迭代和回溯步长。lambdaTik 推荐试验范围为 1e-4~1e-2。
physicsConfig.pcgTolerance = 1e-8;        % PCG 相对残差阈值。
physicsConfig.pcgMaxIterations = 300;     % PCG 最大迭代次数；高分辨率可增大到 500。
physicsConfig.dataWeight = 1.00;          % Kriging 二次数据保真权重。
physicsConfig.originalSampleWeightMultiplier = 1.00; % 原始采样保真相对强度；1 表示总权重与 Kriging 全场项相当。
physicsConfig.tikhonovWeight = 1e-3;      % 一阶 Tikhonov 强度；越大则重建越平滑，原始采样点由 Lsample 锚定。
physicsConfig.curlWeight = 5e-4;          % 二维无旋物理约束强度；数学对照组自动设为 0。

if ~isfile(krigingResultFile)
    error('找不到 Kriging 结果文件：%s', krigingResultFile);
end
if ~isfile(originalSampleFile)
    error('找不到原始采样文件：%s', originalSampleFile);
end
if ~isfile(groundTruthFile)
    error('找不到 Ground Truth 文件：%s', groundTruthFile);
end

resultFolder = fullfile(fileparts(krigingResultFile), 'PhysicsConstraint_Results');
if ~isfolder(resultFolder)
    mkdir(resultFolder);
end

fprintf('============================================================\n');
fprintf('Kriging + 标准一阶 Tikhonov + Physics Constraint 高分辨率重建\n');
fprintf('Kriging 先验：%s\n', krigingResultFile);
fprintf('原始采样：%s\n', originalSampleFile);
fprintf('Ground Truth：%s\n', groundTruthFile);
fprintf('输出目录：%s\n', resultFolder);
fprintf('============================================================\n');

%% ==================== 2. 读取 Kriging 初值与 Ground Truth ====================
krigingPrior = readKrigingPrior(krigingResultFile);
originalSamples = readOriginalSamplePoints(originalSampleFile);
validateOriginalSamplesAgainstGrid(originalSamples, krigingPrior.x, krigingPrior.y, krigingPrior.z_mm);
groundTruth = readComsolGroundTruth1000(groundTruthFile);
groundTruth.Bmag = sqrt(groundTruth.Bx.^2 + groundTruth.By.^2 + groundTruth.Bz.^2);

% 物理优化严格在 Kriging 原始输出网格上进行，绝不先改写为 1000×1000。
optimizationX = krigingPrior.x;
optimizationY = krigingPrior.y;
krigingField = cat(3, krigingPrior.Bx, krigingPrior.By, krigingPrior.Bz);
assertFinite(krigingField, 'Kriging 优化初值');
krigingBmag = vectorMagnitude(krigingField);
fprintf('物理优化网格：%d×%d（x=%d 点，y=%d 点）\n', ...
    size(krigingField, 1), size(krigingField, 2), numel(optimizationX), numel(optimizationY));
fprintf('原始采样点：%d 个；将作为 Lsample 数据项并用于一致性检查。\n', numel(originalSamples.x));

%% ==================== 3. 构造先验并执行两类优化 ====================
% 使用一个全局磁场幅值归一化三分量，使 Tikhonov 权重可在不同量级数据间复用。
fieldScale = max(abs(krigingField), [], 'all');
if ~isfinite(fieldScale) || fieldScale <= eps
    error('Kriging 三分量全局幅值无效，无法进行物理约束优化。');
end

priorNormalized = krigingField / fieldScale;
[dxNormalized, dyNormalized] = normalizedGridSpacing(optimizationX, optimizationY);
% S 是从高分辨率网格回采样至原始实测坐标的双线性算子。
% 将所有原始采样点的总权重标定为与 Kriging 全场数据项相当，
% 因而 10×10 与 100×100 源文件在不同重建分辨率下具有可比约束强度。
physicsConfig.sampleOperator = createBilinearSampleOperator( ...
    optimizationX, optimizationY, originalSamples.x, originalSamples.y);
physicsConfig.sampleValuesNormalized = [originalSamples.Bx, originalSamples.By, originalSamples.Bz] / fieldScale;
physicsConfig.originalSampleDataWeight = physicsConfig.dataWeight * ...
    numel(optimizationX) * numel(optimizationY) / numel(originalSamples.x) * ...
    physicsConfig.originalSampleWeightMultiplier;
% 无旋项在归一化坐标中计算；以下比例将其恢复到 x/y 实际物理长度的相对尺度。
curlReferenceLength = sqrt((optimizationX(end) - optimizationX(1)) * ...
    (optimizationY(end) - optimizationY(1)));
physicsConfig.curlXFactor = curlReferenceLength / (optimizationX(end) - optimizationX(1));
physicsConfig.curlYFactor = curlReferenceLength / (optimizationY(end) - optimizationY(1));

% 数学对照组与物理组使用完全相同的初值、二次数据保真项和一阶 Tikhonov 项。
% 唯一区别是数学对照组 curlWeight=0，物理组加入二维无旋约束。
mathematicalConfig = physicsConfig;
mathematicalConfig.curlWeight = 0;

fprintf('\n[1] 开始标准一阶 Tikhonov 正则化：L = Lprior + Lsample + LTik。\n');
fprintf('Kriging/原始采样/Tikhonov 权重 = %.4g / %.4g / %.4g\n', ...
    mathematicalConfig.dataWeight, mathematicalConfig.originalSampleDataWeight, ...
    mathematicalConfig.tikhonovWeight);
[mathematicalNormalized, mathematicalSolverHistory, mathematicalSolverInfo] = optimizePhysicsConstrainedField( ...
    priorNormalized, dxNormalized, dyNormalized, mathematicalConfig);

fprintf('\n[2] 开始物理约束优化：L = Lprior + Lsample + LTik + Lcurl。\n');
fprintf('Kriging/原始采样/Tikhonov/无旋权重 = %.4g / %.4g / %.4g / %.4g\n', ...
    physicsConfig.dataWeight, physicsConfig.originalSampleDataWeight, ...
    physicsConfig.tikhonovWeight, physicsConfig.curlWeight);
fprintf('无旋残差定义：dBy/dx - dBx/dy（单位：T/mm）。\n');
[physicsNormalized, physicsSolverHistory, physicsSolverInfo] = optimizePhysicsConstrainedField( ...
    priorNormalized, dxNormalized, dyNormalized, physicsConfig);

mathematicalField = mathematicalNormalized * fieldScale;
assertFinite(mathematicalField, '标准一阶 Tikhonov 正则化后的三分量场');
physicsField = physicsNormalized * fieldScale;
assertFinite(physicsField, '物理约束优化后的三分量场');

bxMathematical = mathematicalField(:, :, 1);
byMathematical = mathematicalField(:, :, 2);
bzMathematical = mathematicalField(:, :, 3);
mathematicalBmag = vectorMagnitude(mathematicalField);
bxPhysics = physicsField(:, :, 1);
byPhysics = physicsField(:, :, 2);
bzPhysics = physicsField(:, :, 3);
physicsBmag = vectorMagnitude(physicsField);

% 原始采样已通过 Lsample 参与优化。此处再把三种重建回采样至观测坐标，
% 用于独立报告其拟合误差，并量化两次后处理在观测点上的实际改变量。
krigingAtSamples = sampleFieldAtOriginalPoints(krigingField, optimizationX, optimizationY, originalSamples);
mathematicalAtSamples = sampleFieldAtOriginalPoints(mathematicalField, optimizationX, optimizationY, originalSamples);
physicsAtSamples = sampleFieldAtOriginalPoints(physicsField, optimizationX, optimizationY, originalSamples);
sampleMetricsKriging = calculateOriginalSampleMetrics(krigingAtSamples, originalSamples);
sampleMetricsMathematical = calculateOriginalSampleMetrics(mathematicalAtSamples, originalSamples);
sampleMetricsPhysics = calculateOriginalSampleMetrics(physicsAtSamples, originalSamples);
sampleConsistency = buildOriginalSampleConsistencyTable( ...
    sampleMetricsKriging, sampleMetricsMathematical, sampleMetricsPhysics);
sampleTikhonovCorrection = mathematicalAtSamples.Bmag - krigingAtSamples.Bmag;
samplePhysicsCorrection = physicsAtSamples.Bmag - mathematicalAtSamples.Bmag;
sampleCorrectionSummary = table(["First-order Tikhonov - Kriging"; "Physics - First-order Tikhonov"], ...
    [sqrt(mean(sampleTikhonovCorrection.^2)); sqrt(mean(samplePhysicsCorrection.^2))], ...
    [max(abs(sampleTikhonovCorrection)); max(abs(samplePhysicsCorrection))], ...
    'VariableNames', {'Correction', 'RMS_Bmag_T', 'MaxAbs_Bmag_T'});
fprintf('\n原始采样点一致性（高分辨率结果回采样至原始 %d 个观测坐标）：\n', numel(originalSamples.x));
disp(sampleConsistency);
fprintf('原始采样点上的方法改变量：\n');
disp(sampleCorrectionSummary);
reportOriginalSampleInfluence(sampleMetricsKriging, sampleCorrectionSummary, ...
    max(abs(originalSamples.Bmag)), mathematicalConfig, physicsConfig);

%% ==================== 4. Ground Truth 指标与物理残差 ====================
% Ground Truth 仅用于训练完成后的评价，绝不进入优化损失函数。
% 重建结果始终保持在原优化网格；只将原始 Ground Truth 重采样至该网格进行逐点误差计算。
[groundTruthEvaluation, evaluationGridOperation] = resampleGroundTruthToTargetGrid( ...
    groundTruth, optimizationX, optimizationY);
fprintf('%s\n', evaluationGridOperation);
krigingMetrics = calculateComparisonMetrics(krigingField, groundTruthEvaluation);
mathematicalMetrics = calculateComparisonMetrics(mathematicalField, groundTruthEvaluation);
physicsMetrics = calculateComparisonMetrics(physicsField, groundTruthEvaluation);

[ctKriging, curlKriging] = calculatePhysicalFields(krigingField, optimizationX, optimizationY);
[ctMathematical, curlMathematical] = calculatePhysicalFields(mathematicalField, optimizationX, optimizationY);
[ctPhysics, curlPhysics] = calculatePhysicalFields(physicsField, optimizationX, optimizationY);

metricSummary = table(["Kriging"; "First-order Tikhonov"; "Physics constrained"], ...
    [krigingMetrics.Bmag.RMSE; mathematicalMetrics.Bmag.RMSE; physicsMetrics.Bmag.RMSE], ...
    [krigingMetrics.Bmag.MAE; mathematicalMetrics.Bmag.MAE; physicsMetrics.Bmag.MAE], ...
    [krigingMetrics.Bmag.MaxError; mathematicalMetrics.Bmag.MaxError; physicsMetrics.Bmag.MaxError], ...
    [krigingMetrics.Bmag.PeakValueError; mathematicalMetrics.Bmag.PeakValueError; physicsMetrics.Bmag.PeakValueError], ...
    [krigingMetrics.Bmag.PeakPositionError_mm; mathematicalMetrics.Bmag.PeakPositionError_mm; physicsMetrics.Bmag.PeakPositionError_mm], ...
    [mean(abs(curlKriging), 'all'); mean(abs(curlMathematical), 'all'); mean(abs(curlPhysics), 'all')], ...
    [max(abs(curlKriging), [], 'all'); max(abs(curlMathematical), [], 'all'); max(abs(curlPhysics), [], 'all')], ...
    'VariableNames', {'Method', 'RMSE_Bmag_T', 'MAE_Bmag_T', 'MaxError_Bmag_T', ...
    'PeakValueError_T', 'PeakPositionError_mm', ...
    'MeanAbsCurlZ_T_per_mm', 'MaxAbsCurlZ_T_per_mm'});

componentMetrics = buildComponentMetricTable(krigingMetrics, mathematicalMetrics, physicsMetrics);
fprintf('\n[3] |B| 误差与二维无旋残差均在实际优化网格 %d×%d 计算。\n', ...
    numel(optimizationY), numel(optimizationX));
disp(metricSummary);
fprintf('[4] Bx、By、Bz、|B| 分量误差指标\n');
disp(componentMetrics);

%% ==================== 5. 图像：Kriging、数学正则化与物理重建对比 ====================
% 所有图均使用实际优化网格；Ground Truth 已被重采样到该坐标，便于逐点比较。
errorKriging = krigingBmag - groundTruthEvaluation.Bmag;
errorMathematical = mathematicalBmag - groundTruthEvaluation.Bmag;
errorPhysics = physicsBmag - groundTruthEvaluation.Bmag;
bmagLimits = sharedLimits([groundTruthEvaluation.Bmag(:); originalSamples.Bmag(:); ...
    krigingBmag(:); mathematicalBmag(:); physicsBmag(:)]);
errorLimit = max(abs([errorKriging(:); errorMathematical(:); errorPhysics(:)]));
if errorLimit <= eps
    errorLimit = 1;
end
% 重建方法之间的修正量通常远小于 Ground Truth 误差，必须使用独立色轴。
% 若沿用 errorLimit，真实的正则化改动会在图上被压缩为近乎纯白。
mathematicalCorrection = mathematicalBmag - krigingBmag;
physicsCorrection = physicsBmag - mathematicalBmag;
correctionLimit = max(abs([mathematicalCorrection(:); physicsCorrection(:)]));
if correctionLimit <= eps
    correctionLimit = max(1e-12, 1e-9 * max(abs(krigingBmag), [], 'all'));
end
fprintf(['标准一阶 Tikhonov 相对 Kriging 的 |B| 修正：RMS=%.6g T，最大值=%.6g T；' ...
    'Physics 相对数学组修正：RMS=%.6g T，最大值=%.6g T。\n'], ...
    sqrt(mean(mathematicalCorrection.^2, 'all')), max(abs(mathematicalCorrection), [], 'all'), ...
    sqrt(mean(physicsCorrection.^2, 'all')), max(abs(physicsCorrection), [], 'all'));

figBmag = figure(1);
set(figBmag, 'Color', 'w', 'Position', [35, 70, 1920, 850], ...
    'Name', 'Kriging、标准一阶 Tikhonov 与 Physics Constraint 的 |B| 对比');
clf(figBmag);
layoutBmag = tiledlayout(figBmag, 2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
plotFieldTile(layoutBmag, optimizationX, optimizationY, groundTruthEvaluation.Bmag, ...
    'Ground Truth |B|（重采样至重建网格）', '|B| (T)', bmagLimits, parula(256));
plotFieldTile(layoutBmag, optimizationX, optimizationY, krigingBmag, ...
    'Kriging 初值 |B|', '|B| (T)', bmagLimits, parula(256));
plotFieldTile(layoutBmag, optimizationX, optimizationY, mathematicalBmag, ...
    'First-order Tikhonov |B|', '|B| (T)', bmagLimits, parula(256));
plotFieldTile(layoutBmag, optimizationX, optimizationY, physicsBmag, ...
    'Physics Constraint 重建 |B|', '|B| (T)', bmagLimits, parula(256));
plotSignedFieldTile(layoutBmag, optimizationX, optimizationY, errorKriging, ...
    'Kriging Error: Kriging - Ground Truth', 'Error |B| (T)', errorLimit);
plotSignedFieldTile(layoutBmag, optimizationX, optimizationY, errorMathematical, ...
    'Tikhonov Error: Tikhonov - Ground Truth', 'Error |B| (T)', errorLimit);
plotSignedFieldTile(layoutBmag, optimizationX, optimizationY, errorPhysics, ...
    'Physics Error: Physics - Ground Truth', 'Error |B| (T)', errorLimit);
plotSignedFieldTile(layoutBmag, optimizationX, optimizationY, physicsCorrection, ...
    'Physics - Mathematical Correction', 'Correction |B| (T)', correctionLimit);
title(layoutBmag, sprintf(['Kriging 初值、标准一阶 Tikhonov 与二维无旋物理约束的 |B| 对比；' ...
    '方法间修正色轴为 ±%.4g T'], correctionLimit));
saveFigure(figBmag, resultFolder, '01_Kriging_Mathematical_Physics_Bmag_Comparison', saveFigures, figureResolution);

ctLimits = sharedLimits([ctKriging(:); ctMathematical(:); ctPhysics(:)]);
curlLimits = sharedLimits(abs([curlKriging(:); curlMathematical(:); curlPhysics(:)]));
figPhysics = figure(2);
set(figPhysics, 'Color', 'w', 'Position', [65, 100, 1660, 850], ...
    'Name', 'Ct_2D 与二维无旋残差对比');
clf(figPhysics);
layoutPhysics = tiledlayout(figPhysics, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
% Ct_2D 用于观察一阶正则化的结构变化；无旋残差是物理组直接优化的物理量。
plotFieldTile(layoutPhysics, optimizationX, optimizationY, ctKriging, ...
    'Kriging Ct_2D', 'Ct_2D (T^2/mm^2)', ctLimits, turbo(256));
plotFieldTile(layoutPhysics, optimizationX, optimizationY, ctMathematical, ...
    'First-order Tikhonov Ct_2D', 'Ct_2D (T^2/mm^2)', ctLimits, turbo(256));
plotFieldTile(layoutPhysics, optimizationX, optimizationY, ctPhysics, ...
    'Physics Ct_2D', 'Ct_2D (T^2/mm^2)', ctLimits, turbo(256));
plotFieldTile(layoutPhysics, optimizationX, optimizationY, abs(curlKriging), ...
    '|dBy/dx-dBx/dy|：Kriging', '|Curl_z| (T/mm)', curlLimits, hot(256));
plotFieldTile(layoutPhysics, optimizationX, optimizationY, abs(curlMathematical), ...
    '|dBy/dx-dBx/dy|：First-order Tikhonov', '|Curl_z| (T/mm)', curlLimits, hot(256));
plotFieldTile(layoutPhysics, optimizationX, optimizationY, abs(curlPhysics), ...
    '|dBy/dx-dBx/dy|：Physics constrained', '|Curl_z| (T/mm)', curlLimits, hot(256));
title(layoutPhysics, 'Ct_2D 结构对比与无自由电流区域的二维无旋物理约束');
saveFigure(figPhysics, resultFolder, '02_Kriging_Mathematical_Physics_CT_Curl_Comparison', saveFigures, figureResolution);

figLoss = figure(3);
set(figLoss, 'Color', 'w', 'Position', [150, 120, 1240, 520], ...
    'Name', 'Tikhonov 与物理约束 PCG 求解诊断');
clf(figLoss);
lossLayout = tiledlayout(figLoss, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile(lossLayout);
semilogy(mathematicalSolverHistory.Iteration, mathematicalSolverHistory.RelativeResidual, ...
    'b-', 'LineWidth', 1.5); hold on;
semilogy(physicsSolverHistory.Iteration, physicsSolverHistory.RelativeResidual, ...
    'r-', 'LineWidth', 1.5); grid on;
xlabel('PCG iteration'); ylabel('Relative linear residual'); title('PCG 收敛历史');
legend('First-order Tikhonov', 'Physics constrained', 'Location', 'best');
nexttile(lossLayout);
objectiveTerms = [mathematicalSolverInfo.finalObjective.Data, mathematicalSolverInfo.finalObjective.OriginalSample, ...
    mathematicalSolverInfo.finalObjective.Tikhonov, mathematicalSolverInfo.finalObjective.Curl; ...
    physicsSolverInfo.finalObjective.Data, physicsSolverInfo.finalObjective.OriginalSample, ...
    physicsSolverInfo.finalObjective.Tikhonov, physicsSolverInfo.finalObjective.Curl];
bar(categorical({'First-order Tikhonov', 'Physics constrained'}), objectiveTerms, 'grouped'); grid on;
ylabel('Objective term value'); title('最终二次目标函数分项');
legend('Kriging prior', 'Original samples', 'Tikhonov', 'Curl_z', 'Location', 'best');
saveFigure(figLoss, resultFolder, '03_Tikhonov_Physics_PCG_Diagnostics', saveFigures, figureResolution);

% Figure 4：以原始低分辨率采样为基准，检查三种结果的采样点一致性和实际修正量。
sampleCorrectionLimit = max(abs([sampleTikhonovCorrection; samplePhysicsCorrection]));
if sampleCorrectionLimit <= eps
    sampleCorrectionLimit = max(1e-12, 1e-9 * max(abs(originalSamples.Bmag)));
end
figSamples = figure(4);
set(figSamples, 'Color', 'w', 'Position', [120, 90, 1440, 830], ...
    'Name', '原始采样点一致性与后处理改变量');
clf(figSamples);
plotOriginalSampleDiagnostics(figSamples, originalSamples, krigingAtSamples, ...
    mathematicalAtSamples, physicsAtSamples, sampleTikhonovCorrection, ...
    samplePhysicsCorrection, bmagLimits, sampleCorrectionLimit);
saveFigure(figSamples, resultFolder, '04_OriginalSample_Consistency_Diagnostics', saveFigures, figureResolution);

%% ==================== 6. 保存结果 ====================
physicsResults = struct();
physicsResults.krigingResultFile = krigingResultFile;
physicsResults.originalSampleFile = originalSampleFile;
physicsResults.groundTruthFile = groundTruthFile;
physicsResults.optimizationGrid = [numel(optimizationY), numel(optimizationX)];
physicsResults.evaluationGridOperation = evaluationGridOperation;
physicsResults.mathematicalConfig = mathematicalConfig;
physicsResults.physicsConfig = physicsConfig;
physicsResults.x_mm = optimizationX;
physicsResults.y_mm = optimizationY;
physicsResults.evaluationX_mm = optimizationX;
physicsResults.evaluationY_mm = optimizationY;
physicsResults.kriging.Bx = krigingField(:, :, 1);
physicsResults.kriging.By = krigingField(:, :, 2);
physicsResults.kriging.Bz = krigingField(:, :, 3);
physicsResults.kriging.Bmag = krigingBmag;
physicsResults.kriging.Ct2D = ctKriging;
physicsResults.kriging.CurlZ = curlKriging;
physicsResults.mathematical.Bx = bxMathematical;
physicsResults.mathematical.By = byMathematical;
physicsResults.mathematical.Bz = bzMathematical;
physicsResults.mathematical.Bmag = mathematicalBmag;
physicsResults.mathematical.Ct2D = ctMathematical;
physicsResults.mathematical.CurlZ = curlMathematical;
physicsResults.physics.Bx = bxPhysics;
physicsResults.physics.By = byPhysics;
physicsResults.physics.Bz = bzPhysics;
physicsResults.physics.Bmag = physicsBmag;
physicsResults.physics.Ct2D = ctPhysics;
physicsResults.physics.CurlZ = curlPhysics;
physicsResults.evaluation.KrigingBx = krigingField(:, :, 1);
physicsResults.evaluation.KrigingBy = krigingField(:, :, 2);
physicsResults.evaluation.KrigingBz = krigingField(:, :, 3);
physicsResults.evaluation.KrigingBmag = krigingBmag;
physicsResults.evaluation.MathematicalBx = bxMathematical;
physicsResults.evaluation.MathematicalBy = byMathematical;
physicsResults.evaluation.MathematicalBz = bzMathematical;
physicsResults.evaluation.MathematicalBmag = mathematicalBmag;
physicsResults.evaluation.PhysicsBx = physicsField(:, :, 1);
physicsResults.evaluation.PhysicsBy = physicsField(:, :, 2);
physicsResults.evaluation.PhysicsBz = physicsField(:, :, 3);
physicsResults.evaluation.PhysicsBmag = physicsBmag;
physicsResults.groundTruth.Bx = groundTruthEvaluation.Bx;
physicsResults.groundTruth.By = groundTruthEvaluation.By;
physicsResults.groundTruth.Bz = groundTruthEvaluation.Bz;
physicsResults.groundTruth.Bmag = groundTruthEvaluation.Bmag;
physicsResults.error.KrigingBmag = errorKriging;
physicsResults.error.MathematicalBmag = errorMathematical;
physicsResults.error.PhysicsBmag = errorPhysics;
physicsResults.correction.MathematicalMinusKrigingBmag = mathematicalCorrection;
physicsResults.correction.PhysicsMinusMathematicalBmag = physicsCorrection;
physicsResults.correction.colorLimit_T = correctionLimit;
physicsResults.originalSamples = originalSamples;
physicsResults.originalSampleEvaluation.Kriging = krigingAtSamples;
physicsResults.originalSampleEvaluation.FirstOrderTikhonov = mathematicalAtSamples;
physicsResults.originalSampleEvaluation.Physics = physicsAtSamples;
physicsResults.originalSampleConsistency = sampleConsistency;
physicsResults.originalSampleCorrectionSummary = sampleCorrectionSummary;
physicsResults.metricSummary = metricSummary;
physicsResults.componentMetrics = componentMetrics;
physicsResults.mathematicalSolverHistory = mathematicalSolverHistory;
physicsResults.physicsSolverHistory = physicsSolverHistory;
physicsResults.mathematicalSolverInfo = mathematicalSolverInfo;
physicsResults.physicsSolverInfo = physicsSolverInfo;

save(fullfile(resultFolder, 'Kriging_Mathematical_Physics_Results.mat'), 'physicsResults', '-v7.3');
writetable(metricSummary, fullfile(resultFolder, 'Kriging_Mathematical_Physics_Bmag_Metrics.csv'), 'Encoding', 'UTF-8');
writetable(componentMetrics, fullfile(resultFolder, 'Kriging_Mathematical_Physics_Component_Metrics.csv'), 'Encoding', 'UTF-8');
writetable(sampleConsistency, fullfile(resultFolder, 'OriginalSamplingPoint_Consistency.csv'), 'Encoding', 'UTF-8');
writetable(sampleCorrectionSummary, fullfile(resultFolder, 'OriginalSamplingPoint_Correction.csv'), 'Encoding', 'UTF-8');
writetable(mathematicalSolverHistory, fullfile(resultFolder, 'FirstOrder_Tikhonov_PCG_Residual.csv'), 'Encoding', 'UTF-8');
writetable(physicsSolverHistory, fullfile(resultFolder, 'PhysicsConstraint_PCG_Residual.csv'), 'Encoding', 'UTF-8');
writePhysicsSummary(fullfile(resultFolder, 'Kriging_Mathematical_Physics_Summary.txt'), ...
    physicsResults, metricSummary);

fprintf('\n完成：Kriging、标准一阶 Tikhonov 与 Physics Constraint 结果已保存至：%s\n', resultFolder);

%% ==================== 局部函数：文件与数据 ====================
function prior = readKrigingPrior(filePath)
% 功能：只读取主重建脚本导出的 Kriging 三分量 TXT 先验。
    [~, ~, extension] = fileparts(filePath);
    if ~strcmpi(extension, '.txt')
        error('Kriging 先验文件必须是主脚本导出的 Kriging_重建数据.txt：%s', filePath);
    end
    prior = readKrigingPointTable(filePath);
end

function prior = readKrigingPointTable(filePath)
% 功能：读取主重建脚本输出的 Kriging_重建数据.txt，并恢复为 [y,x] 网格矩阵。
% 输入格式：前 9 行为文本头；其后六列为 x(mm)、y(mm)、z(mm)、Bx(T)、By(T)、Bz(T)。
    rawData = readmatrix(filePath, 'FileType', 'text', 'NumHeaderLines', 9);
    if size(rawData, 2) < 6
        error('Kriging TXT 数值列少于 6 列，无法读取 x、y、z、Bx、By、Bz：%s', filePath);
    end
    rawData = rawData(:, 1:6);
    if isempty(rawData) || any(~isfinite(rawData), 'all')
        error('Kriging TXT 含空数据、NaN 或 Inf，无法作为 Physics 优化初值：%s', filePath);
    end

    x = rawData(:, 1);
    y = rawData(:, 2);
    z = rawData(:, 3);
    xValues = unique(x, 'sorted')';
    yValues = unique(y, 'sorted')';
    zValues = unique(z, 'sorted')';
    expectedCount = numel(xValues) * numel(yValues);
    if numel(zValues) ~= 1 || numel(x) ~= expectedCount
        error(['Kriging TXT 必须为完整单一 z 平面网格；实际为 %d 行、%d 个 x、' ...
            '%d 个 y、%d 个 z。'], numel(x), numel(xValues), numel(yValues), numel(zValues));
    end
    if ~isStrictlyIncreasingAxis(xValues) || ~isStrictlyIncreasingAxis(yValues)
        error('Kriging TXT 的 x/y 坐标必须严格递增。');
    end

    [~, xIndex] = ismember(x, xValues);
    [~, yIndex] = ismember(y, yValues);
    gridSize = [numel(yValues), numel(xValues)];
    prior.x = xValues;
    prior.y = yValues;
    prior.z_mm = zValues(1);
    prior.Bx = accumarray([yIndex, xIndex], rawData(:, 4), gridSize, @mean, NaN);
    prior.By = accumarray([yIndex, xIndex], rawData(:, 5), gridSize, @mean, NaN);
    prior.Bz = accumarray([yIndex, xIndex], rawData(:, 6), gridSize, @mean, NaN);
    assertFinite(prior.Bx, 'Kriging TXT Bx（同时检查缺失或重复坐标）');
    assertFinite(prior.By, 'Kriging TXT By（同时检查缺失或重复坐标）');
    assertFinite(prior.Bz, 'Kriging TXT Bz（同时检查缺失或重复坐标）');
    fprintf('已从 TXT 读取 Kriging 先验：%d×%d；z=%.12g mm\n', ...
        numel(yValues), numel(xValues), prior.z_mm);
end

function samples = readOriginalSamplePoints(filePath)
% 功能：读取原始 10×10 或 100×100 COMSOL 采样点，不将其重构为高分辨率网格。
% 输入格式：前 9 行为文本头；其后六列为 x(mm)、y(mm)、z(mm)、Bx(T)、By(T)、Bz(T)。
    rawData = readmatrix(filePath, 'FileType', 'text', 'NumHeaderLines', 9);
    if size(rawData, 2) < 6
        error('原始采样文件数值列少于 6 列，无法读取 x、y、z、Bx、By、Bz：%s', filePath);
    end
    rawData = rawData(:, 1:6);
    if isempty(rawData) || any(~isfinite(rawData), 'all')
        error('原始采样文件含空数据、NaN 或 Inf：%s', filePath);
    end
    samples.x = rawData(:, 1);
    samples.y = rawData(:, 2);
    samples.z = rawData(:, 3);
    samples.Bx = rawData(:, 4);
    samples.By = rawData(:, 5);
    samples.Bz = rawData(:, 6);
    samples.Bmag = sqrt(samples.Bx.^2 + samples.By.^2 + samples.Bz.^2);
    zValues = unique(samples.z, 'sorted');
    if numel(zValues) ~= 1
        error('原始采样文件必须为单一 z 平面；实际检测到 %d 个 z 坐标。', numel(zValues));
    end
    if size(unique([samples.x, samples.y], 'rows'), 1) ~= numel(samples.x)
        error('原始采样文件含重复 x/y 坐标，无法进行逐点一致性检查。');
    end
    samples.z_mm = zValues(1);
end

function validateOriginalSamplesAgainstGrid(samples, xGrid, yGrid, zGrid)
% 功能：保证原始采样点位于 Kriging 重建网格范围内且处于同一 z 平面。
% 说明：采样点可不与高分辨率节点重合，后续统一用双线性插值进行回采样。
    coordinateScale = max([1; abs(xGrid(:)); abs(yGrid(:)); abs(samples.x(:)); abs(samples.y(:))]);
    tolerance = 1e-9 * coordinateScale;
    outsideDomain = samples.x < xGrid(1) - tolerance | samples.x > xGrid(end) + tolerance | ...
        samples.y < yGrid(1) - tolerance | samples.y > yGrid(end) + tolerance;
    if any(outsideDomain)
        error(['原始采样点存在 %d 个点位于 Kriging 重建范围外，不能回采样比较。' ...
            '请确认原始采样文件与 Kriging_重建数据.txt 使用同一空间范围。'], sum(outsideDomain));
    end
    if abs(samples.z_mm - zGrid) > tolerance
        error(['原始采样 z=%.12g mm 与 Kriging 结果 z=%.12g mm 不一致，' ...
            '不能将不同测量平面的磁场直接比较。'], samples.z_mm, zGrid);
    end
end

function sampled = sampleFieldAtOriginalPoints(field, xGrid, yGrid, samples)
% 功能：在原始采样点坐标，以线性插值从高分辨率三分量场回采样。
% 说明：先回采样 Bx/By/Bz，再计算 |B|，避免直接插值 |B| 带来的非线性偏差。
    interpolantBx = griddedInterpolant({yGrid, xGrid}, field(:, :, 1), 'linear', 'none');
    interpolantBy = griddedInterpolant({yGrid, xGrid}, field(:, :, 2), 'linear', 'none');
    interpolantBz = griddedInterpolant({yGrid, xGrid}, field(:, :, 3), 'linear', 'none');
    sampled.Bx = interpolantBx(samples.y, samples.x);
    sampled.By = interpolantBy(samples.y, samples.x);
    sampled.Bz = interpolantBz(samples.y, samples.x);
    sampled.Bmag = sqrt(sampled.Bx.^2 + sampled.By.^2 + sampled.Bz.^2);
    assertFinite(sampled.Bx, '原始采样点回采样 Bx');
    assertFinite(sampled.By, '原始采样点回采样 By');
    assertFinite(sampled.Bz, '原始采样点回采样 Bz');
end

function operator = createBilinearSampleOperator(xGrid, yGrid, xSamples, ySamples)
% 功能：构造从 [y,x] 高分辨率网格到原始采样坐标的稀疏双线性回采样矩阵 S。
% 说明：S*B 给出采样点预测值；S'*r 将采样残差回投影到高分辨率网格。
%       因此可在 PCG 中精确使用 Lsample=(wSample/2)||S(B)-Bsample||_F^2。
    xGrid = xGrid(:)';
    yGrid = yGrid(:)';
    xSamples = xSamples(:);
    ySamples = ySamples(:);
    xPosition = interp1(xGrid, 1:numel(xGrid), xSamples, 'linear');
    yPosition = interp1(yGrid, 1:numel(yGrid), ySamples, 'linear');
    if any(~isfinite(xPosition)) || any(~isfinite(yPosition))
        error('存在原始采样点无法映射到 Kriging 重建网格，不能建立采样保真项。');
    end

    [xLeft, xFraction] = interpolationCell(xPosition, numel(xGrid));
    [yLower, yFraction] = interpolationCell(yPosition, numel(yGrid));
    xRight = xLeft + 1;
    yUpper = yLower + 1;
    sampleCount = numel(xSamples);
    rowIndex = repmat((1:sampleCount)', 4, 1);
    columnIndex = [sub2ind([numel(yGrid), numel(xGrid)], yLower, xLeft); ...
        sub2ind([numel(yGrid), numel(xGrid)], yLower, xRight); ...
        sub2ind([numel(yGrid), numel(xGrid)], yUpper, xLeft); ...
        sub2ind([numel(yGrid), numel(xGrid)], yUpper, xRight)];
    weights = [(1 - xFraction) .* (1 - yFraction); ...
        xFraction .* (1 - yFraction); ...
        (1 - xFraction) .* yFraction; ...
        xFraction .* yFraction];
    operator = sparse(rowIndex, columnIndex, weights, sampleCount, numel(xGrid) * numel(yGrid));
end

function [lowerIndex, fraction] = interpolationCell(position, pointCount)
% 功能：将连续网格索引转换为双线性插值单元及单元内比例。
    position = min(max(position, 1), pointCount);
    lowerIndex = floor(position);
    fraction = position - lowerIndex;
    atUpperBoundary = lowerIndex >= pointCount;
    lowerIndex(atUpperBoundary) = pointCount - 1;
    fraction(atUpperBoundary) = 1;
    lowerIndex = max(lowerIndex, 1);
end

function groundTruth = readComsolGroundTruth1000(filePath)
% 功能：读取 COMSOL 直接导出的 1000×1000 单 z 平面真值，不做任何插值。
    rawData = readmatrix(filePath, 'FileType', 'text', 'NumHeaderLines', 9);
    if size(rawData, 2) < 6
        error('Ground Truth 数值列少于 6 列，无法读取 x、y、z、Bx、By、Bz。');
    end
    rawData = rawData(:, 1:6);
    if any(~isfinite(rawData), 'all')
        error('Ground Truth 包含 NaN 或 Inf，不能用于逐点误差评价。');
    end
    x = rawData(:, 1); y = rawData(:, 2); z = rawData(:, 3);
    xValues = unique(x, 'sorted')';
    yValues = unique(y, 'sorted')';
    zValues = unique(z, 'sorted')';
    if numel(x) ~= 1000000 || numel(xValues) ~= 1000 || numel(yValues) ~= 1000 || numel(zValues) ~= 1
        error(['Ground Truth 必须是完整 1000×1000 单一 z 平面；实际为 %d 点、%d 个 x、' ...
            '%d 个 y、%d 个 z。'], numel(x), numel(xValues), numel(yValues), numel(zValues));
    end
    % COMSOL 文本坐标可能因小数位截断呈现 0.198/0.199 mm 交替间隔。
    % Ground Truth 在这里仅作为 griddedInterpolant 的源数据，源轴无需严格等间距；
    % 真正用于差分和物理约束的是后续 Kriging 的规则优化网格。
    if ~isStrictlyIncreasingAxis(xValues) || ~isStrictlyIncreasingAxis(yValues)
        error(['Ground Truth 的 x/y 坐标必须严格递增，才能重采样至重建网格。' ...
            '请检查 COMSOL 导出文件中的坐标列。']);
    end
    [~, xIndex] = ismember(x, xValues);
    [~, yIndex] = ismember(y, yValues);
    gridSize = [1000, 1000];
    groundTruth.x = xValues;
    groundTruth.y = yValues;
    groundTruth.z_mm = zValues;
    groundTruth.Bx = accumarray([yIndex, xIndex], rawData(:, 4), gridSize, @mean, NaN);
    groundTruth.By = accumarray([yIndex, xIndex], rawData(:, 5), gridSize, @mean, NaN);
    groundTruth.Bz = accumarray([yIndex, xIndex], rawData(:, 6), gridSize, @mean, NaN);
    assertFinite(groundTruth.Bx, 'Ground Truth Bx');
    assertFinite(groundTruth.By, 'Ground Truth By');
    assertFinite(groundTruth.Bz, 'Ground Truth Bz');
end

function [evaluationTruth, operation] = resampleGroundTruthToTargetGrid(sourceTruth, xTarget, yTarget)
% 功能：仅为误差评价，将原始 1000×1000 Ground Truth 三分量重采样至重建网格。
% 说明：重建结果与物理优化变量不重采样；它们保持用户指定的重建分辨率。
    if coordinateVectorsMatch(sourceTruth.x, xTarget) && coordinateVectorsMatch(sourceTruth.y, yTarget)
        evaluationTruth = sourceTruth;
        operation = 'Ground Truth 坐标与重建网格一致：直接使用原始 Ground Truth 进行误差评价。';
        return;
    end
    [xMesh, yMesh] = meshgrid(xTarget, yTarget);
    bxInterpolant = griddedInterpolant({sourceTruth.y, sourceTruth.x}, sourceTruth.Bx, 'linear', 'nearest');
    byInterpolant = griddedInterpolant({sourceTruth.y, sourceTruth.x}, sourceTruth.By, 'linear', 'nearest');
    bzInterpolant = griddedInterpolant({sourceTruth.y, sourceTruth.x}, sourceTruth.Bz, 'linear', 'nearest');
    evaluationTruth.x = xTarget;
    evaluationTruth.y = yTarget;
    evaluationTruth.z_mm = sourceTruth.z_mm;
    evaluationTruth.Bx = bxInterpolant(yMesh, xMesh);
    evaluationTruth.By = byInterpolant(yMesh, xMesh);
    evaluationTruth.Bz = bzInterpolant(yMesh, xMesh);
    evaluationTruth.Bmag = sqrt(evaluationTruth.Bx.^2 + evaluationTruth.By.^2 + evaluationTruth.Bz.^2);
    assertFinite(evaluationTruth.Bx, '重采样 Ground Truth Bx');
    assertFinite(evaluationTruth.By, '重采样 Ground Truth By');
    assertFinite(evaluationTruth.Bz, '重采样 Ground Truth Bz');
    operation = sprintf(['已将原始 Ground Truth 从 %d×%d 线性重采样至重建网格 %d×%d；' ...
        'Kriging、数学正则化和 Physics 重建结果未进行重采样。'], ...
        numel(sourceTruth.y), numel(sourceTruth.x), numel(yTarget), numel(xTarget));
end

%% ==================== 局部函数：物理约束优化 ====================
function [field, history, solverInfo] = optimizePhysicsConstrainedField(prior, dx, dy, config)
% 功能：用 PCG 求解带原始采样保真、标准一阶 Tikhonov 和二维无旋约束的二次正规方程。
% 说明：curlWeight=0 时就是纯一阶 Tikhonov 数学正则化对照组。
%       原始采样项通过稀疏回采样矩阵作用，不改变优化变量的高分辨率网格。
%       目标函数为严格二次凸函数，不存在 TV 非线性和回溯步长停滞问题。
    fieldSize = size(prior);
    rightHandSideField = config.dataWeight * prior;
    for component = 1:3
        sampleBackProjection = config.sampleOperator' * config.sampleValuesNormalized(:, component);
        rightHandSideField(:, :, component) = rightHandSideField(:, :, component) + ...
            config.originalSampleDataWeight * reshape(sampleBackProjection, fieldSize(1), fieldSize(2));
    end
    rightHandSide = rightHandSideField(:);
    normalOperator = @(fieldVector) applyTikhonovNormalOperator( ...
        fieldVector, fieldSize, dx, dy, config);
    initialObjective = calculateTikhonovObjective(prior, prior, dx, dy, config);

    [solutionVector, pcgFlag, relativeResidual, iterationCount, residualVector] = pcg( ...
        normalOperator, rightHandSide, config.pcgTolerance, config.pcgMaxIterations, [], [], prior(:));
    field = reshape(solutionVector, fieldSize);
    assertFinite(field, 'Tikhonov/Physics PCG 解');
    finalObjective = calculateTikhonovObjective(field, prior, dx, dy, config);
    residualVector = residualVector(:);
    if isempty(residualVector)
        residualVector = norm(normalOperator(solutionVector) - rightHandSide);
    end
    initialResidual = max(residualVector(1), eps);
    history = table((0:numel(residualVector)-1)', residualVector, residualVector / initialResidual, ...
        'VariableNames', {'Iteration', 'LinearResidual', 'RelativeResidual'});
    solverInfo.flag = pcgFlag;
    solverInfo.relativeResidual = relativeResidual;
    solverInfo.iterationCount = iterationCount;
    solverInfo.initialObjective = initialObjective;
    solverInfo.finalObjective = finalObjective;

    if pcgFlag == 0
        fprintf('  PCG 收敛：%d 次迭代，相对残差=%.3g，目标函数 %.6g -> %.6g。\n', ...
            iterationCount, relativeResidual, initialObjective.Total, finalObjective.Total);
    else
        warning(['PCG 未在设定阈值内完全收敛（flag=%d，相对残差=%.3g，迭代=%d）。' ...
            '将保留当前 PCG 解；可增大 pcgMaxIterations 或放宽 pcgTolerance。'], ...
            pcgFlag, relativeResidual, iterationCount);
    end
end

function outputVector = applyTikhonovNormalOperator(inputVector, fieldSize, dx, dy, config)
% 功能：实现 A*B，其中 A 对应 Tikhonov 和无旋二次目标的正规方程矩阵。
% 说明：不显式构造稀疏大矩阵，避免 1000×1000 或 2000×2000 网格的内存膨胀。
    field = reshape(inputVector, fieldSize);
    output = config.dataWeight * field;
    gradientX = zeros(fieldSize);
    gradientY = zeros(fieldSize);
    for component = 1:3
        componentValues = field(:, :, component);
        sampleForwardProjection = config.sampleOperator * componentValues(:);
        sampleNormalProjection = config.sampleOperator' * sampleForwardProjection;
        output(:, :, component) = output(:, :, component) + config.originalSampleDataWeight * ...
            reshape(sampleNormalProjection, fieldSize(1), fieldSize(2));
        [gradientX(:, :, component), gradientY(:, :, component)] = ...
            forwardGradient(componentValues, dx, dy);
        output(:, :, component) = output(:, :, component) + config.tikhonovWeight * ...
            adjointForwardGradient(gradientX(:, :, component), gradientY(:, :, component), dx, dy);
    end

    if config.curlWeight ~= 0
        curlZ = config.curlXFactor * gradientX(:, :, 2) - ...
            config.curlYFactor * gradientY(:, :, 1);
        output(:, :, 1) = output(:, :, 1) + config.curlWeight * ...
            adjointForwardGradient(zeros(size(curlZ)), -config.curlYFactor * curlZ, dx, dy);
        output(:, :, 2) = output(:, :, 2) + config.curlWeight * ...
            adjointForwardGradient(config.curlXFactor * curlZ, zeros(size(curlZ)), dx, dy);
    end
    outputVector = output(:);
end

function objective = calculateTikhonovObjective(field, prior, dx, dy, config)
% 功能：计算标准一阶 Tikhonov 和无旋约束的各二次目标项，用于报告和验证。
    residual = field - prior;
    sampleResidualEnergy = 0;
    tikhonovEnergy = 0;
    gradientX = zeros(size(field));
    gradientY = zeros(size(field));
    for component = 1:3
        componentValues = field(:, :, component);
        sampleResidual = config.sampleOperator * componentValues(:) - ...
            config.sampleValuesNormalized(:, component);
        sampleResidualEnergy = sampleResidualEnergy + sum(sampleResidual.^2);
        [gradientX(:, :, component), gradientY(:, :, component)] = ...
            forwardGradient(componentValues, dx, dy);
        tikhonovEnergy = tikhonovEnergy + sum(gradientX(:, :, component).^2 + ...
            gradientY(:, :, component).^2, 'all');
    end
    curlZ = config.curlXFactor * gradientX(:, :, 2) - ...
        config.curlYFactor * gradientY(:, :, 1);
    objective.Data = 0.5 * config.dataWeight * sum(residual.^2, 'all');
    objective.OriginalSample = 0.5 * config.originalSampleDataWeight * sampleResidualEnergy;
    objective.Tikhonov = 0.5 * config.tikhonovWeight * tikhonovEnergy;
    objective.Curl = 0.5 * config.curlWeight * sum(curlZ.^2, 'all');
    objective.Total = objective.Data + objective.OriginalSample + objective.Tikhonov + objective.Curl;
end

function [gx, gy] = forwardGradient(values, dx, dy)
% 功能：前向一阶差分；矩阵行是 y、列是 x。
    gx = zeros(size(values));
    gy = zeros(size(values));
    gx(:, 1:end-1) = diff(values, 1, 2) / dx;
    gy(1:end-1, :) = diff(values, 1, 1) / dy;
end

function values = adjointForwardGradient(px, py, dx, dy)
% 功能：前向差分的严格伴随算子，用于 Tikhonov 和二维无旋项的正规方程。
    values = zeros(size(px));
    values(:, 1) = values(:, 1) - px(:, 1) / dx;
    values(:, 2:end-1) = values(:, 2:end-1) + ...
        (px(:, 1:end-2) - px(:, 2:end-1)) / dx;
    values(:, end) = values(:, end) + px(:, end-1) / dx;
    values(1, :) = values(1, :) - py(1, :) / dy;
    values(2:end-1, :) = values(2:end-1, :) + ...
        (py(1:end-2, :) - py(2:end-1, :)) / dy;
    values(end, :) = values(end, :) + py(end-1, :) / dy;
end

function [dx, dy] = normalizedGridSpacing(xValues, yValues)
% 功能：将物理坐标归一化到 [0,1] 后得到相邻节点间距。
    xNormalized = (xValues - xValues(1)) / (xValues(end) - xValues(1));
    yNormalized = (yValues - yValues(1)) / (yValues(end) - yValues(1));
    dx = median(diff(xNormalized));
    dy = median(diff(yNormalized));
    if dx <= 0 || dy <= 0 || ~isfinite(dx) || ~isfinite(dy)
        error('无法建立物理优化所需的归一化规则网格。');
    end
end

%% ==================== 局部函数：指标、绘图与保存 ====================
function metrics = calculateComparisonMetrics(field, groundTruth)
% 功能：计算 Bx、By、Bz、|B| 相对于 Ground Truth 的误差及 |B| 峰值误差。
    quantities = {'Bx', 'By', 'Bz'};
    for index = 1:3
        metrics.(quantities{index}) = basicMetrics(field(:, :, index), groundTruth.(quantities{index}));
    end
    bmag = vectorMagnitude(field);
    metrics.Bmag = basicMetrics(bmag, groundTruth.Bmag);
    [truthPeak, truthIndex] = max(groundTruth.Bmag, [], 'all');
    [fieldPeak, fieldIndex] = max(bmag, [], 'all');
    [truthRow, truthColumn] = ind2sub(size(groundTruth.Bmag), truthIndex);
    [fieldRow, fieldColumn] = ind2sub(size(bmag), fieldIndex);
    metrics.Bmag.PeakValueError = abs(fieldPeak - truthPeak);
    metrics.Bmag.PeakPositionError_mm = hypot(groundTruth.x(fieldColumn) - groundTruth.x(truthColumn), ...
        groundTruth.y(fieldRow) - groundTruth.y(truthRow));
end

function metric = basicMetrics(values, truth)
% 功能：计算逐点 RMSE、MAE、最大绝对误差；不忽略 NaN 或 Inf。
    assertFinite(values, '待评价重建结果');
    assertFinite(truth, 'Ground Truth');
    residual = values - truth;
    metric.RMSE = sqrt(mean(residual.^2, 'all'));
    metric.MAE = mean(abs(residual), 'all');
    metric.MaxError = max(abs(residual), [], 'all');
end

function metrics = calculateOriginalSampleMetrics(reconstructedSamples, originalSamples)
% 功能：计算高分辨率结果回采样至原始观测坐标后的分量和模值误差。
% 说明：这些指标衡量“保留原始采样信息”的程度，不替代 Ground Truth 全场评价。
    metrics.Bx = basicMetrics(reconstructedSamples.Bx, originalSamples.Bx);
    metrics.By = basicMetrics(reconstructedSamples.By, originalSamples.By);
    metrics.Bz = basicMetrics(reconstructedSamples.Bz, originalSamples.Bz);
    metrics.Bmag = basicMetrics(reconstructedSamples.Bmag, originalSamples.Bmag);
end

function summary = buildOriginalSampleConsistencyTable(krigingMetrics, tikhonovMetrics, physicsMetrics)
% 功能：将三种结果在原始采样点的 Bx、By、Bz、|B| 误差汇总为一个表格。
    quantity = repmat(["Bx"; "By"; "Bz"; "|B|"], 3, 1);
    method = [repmat("Kriging", 4, 1); ...
        repmat("First-order Tikhonov", 4, 1); ...
        repmat("Physics constrained", 4, 1)];
    source = {krigingMetrics.Bx, krigingMetrics.By, krigingMetrics.Bz, krigingMetrics.Bmag, ...
        tikhonovMetrics.Bx, tikhonovMetrics.By, tikhonovMetrics.Bz, tikhonovMetrics.Bmag, ...
        physicsMetrics.Bx, physicsMetrics.By, physicsMetrics.Bz, physicsMetrics.Bmag};
    rmse = zeros(12, 1); mae = zeros(12, 1); maxError = zeros(12, 1);
    for index = 1:12
        rmse(index) = source{index}.RMSE;
        mae(index) = source{index}.MAE;
        maxError(index) = source{index}.MaxError;
    end
    summary = table(quantity, method, rmse, mae, maxError, ...
        'VariableNames', {'Quantity', 'Method', 'RMSE_at_OriginalSamples_T', ...
        'MAE_at_OriginalSamples_T', 'MaxError_at_OriginalSamples_T'});
end

function reportOriginalSampleInfluence(krigingMetrics, correctionSummary, sampleMagnitudeScale, mathematicalConfig, physicsConfig)
% 功能：解释原始采样项为何可能不改变 Kriging，并提示应查看哪些实际改变量。
    krigingSampleRmse = krigingMetrics.Bmag.RMSE;
    tikhonovCorrection = correctionSummary.RMS_Bmag_T(1);
    physicsCorrection = correctionSummary.RMS_Bmag_T(2);
    tolerance = max(1e-12, 1e-8 * sampleMagnitudeScale);
    if krigingSampleRmse <= tolerance
        fprintf(['提示：Kriging 在原始采样点的 |B| RMSE=%.3g T，已近似严格通过原始采样。\n' ...
            '因此 Lsample 主要防止后处理偏离观测点，不会单独产生明显场形修正。\n'], ...
            krigingSampleRmse);
    else
        fprintf(['提示：Kriging 在原始采样点的 |B| RMSE=%.3g T；Lsample 会将优化结果' ...
            '向真实原始观测拉回。\n'], krigingSampleRmse);
    end
    fprintf(['原始采样点上的 |B| 修正 RMS：Tikhonov=%.3g T，Physics=%.3g T；' ...
        '当前 lambdaTik=%.3g，lambdaCurl=%.3g。\n'], ...
        tikhonovCorrection, physicsCorrection, mathematicalConfig.tikhonovWeight, ...
        physicsConfig.curlWeight);
end

function tableOutput = buildComponentMetricTable(krigingMetrics, mathematicalMetrics, physicsMetrics)
% 功能：将四个物理量在 Kriging、数学正则化、Physics Constraint 三组中的误差排成表格。
    quantity = repmat(["Bx"; "By"; "Bz"; "|B|"], 3, 1);
    method = [repmat("Kriging", 4, 1); ...
        repmat("First-order Tikhonov", 4, 1); ...
        repmat("Physics constrained", 4, 1)];
    source = {krigingMetrics.Bx, krigingMetrics.By, krigingMetrics.Bz, krigingMetrics.Bmag, ...
        mathematicalMetrics.Bx, mathematicalMetrics.By, mathematicalMetrics.Bz, mathematicalMetrics.Bmag, ...
        physicsMetrics.Bx, physicsMetrics.By, physicsMetrics.Bz, physicsMetrics.Bmag};
    rmse = zeros(12, 1); mae = zeros(12, 1); maxError = zeros(12, 1);
    for index = 1:12
        rmse(index) = source{index}.RMSE;
        mae(index) = source{index}.MAE;
        maxError(index) = source{index}.MaxError;
    end
    tableOutput = table(quantity, method, rmse, mae, maxError, ...
        'VariableNames', {'Quantity', 'Method', 'RMSE_T', 'MAE_T', 'MaxError_T'});
end

function [ct, curlZ] = calculatePhysicalFields(field, xValues, yValues)
% 功能：在真实 mm 坐标下计算 Ct_2D 和二维可观测无旋残差 dBy/dx-dBx/dy。
    dx = median(diff(xValues));
    dy = median(diff(yValues));
    ct = zeros(size(field, 1), size(field, 2));
    gx = zeros(size(field)); gy = zeros(size(field));
    for component = 1:3
        [gx(:, :, component), gy(:, :, component)] = forwardGradient(field(:, :, component), dx, dy);
        ct = ct + gx(:, :, component).^2 + gy(:, :, component).^2;
    end
    curlZ = gx(:, :, 2) - gy(:, :, 1);
end

function magnitude = vectorMagnitude(field)
% 功能：由 Bx、By、Bz 三分量计算磁场模值 |B|。
    magnitude = sqrt(sum(field.^2, 3));
end

function plotFieldTile(layout, xValues, yValues, values, titleText, colorLabel, limits, map)
% 功能：绘制普通二维场图，保证不同方法使用统一色轴。
    nexttile(layout);
    imagesc(xValues, yValues, values); set(gca, 'YDir', 'normal'); axis image;
    xlim([xValues(1), xValues(end)]); ylim([yValues(1), yValues(end)]);
    caxis(limits); colormap(gca, map); xlabel('x (mm)'); ylabel('y (mm)');
    title(titleText, 'Interpreter', 'none');
    colorbarHandle = colorbar; colorbarHandle.Label.String = colorLabel;
end

function plotSignedFieldTile(layout, xValues, yValues, values, titleText, colorLabel, limit)
% 功能：绘制以零为中心的误差或修正图。
    nexttile(layout);
    imagesc(xValues, yValues, values); set(gca, 'YDir', 'normal'); axis image;
    xlim([xValues(1), xValues(end)]); ylim([yValues(1), yValues(end)]);
    caxis([-limit, limit]); colormap(gca, blueWhiteRedMap(256));
    xlabel('x (mm)'); ylabel('y (mm)'); title(titleText, 'Interpreter', 'none');
    colorbarHandle = colorbar; colorbarHandle.Label.String = colorLabel;
end

function plotOriginalSampleDiagnostics(fig, originalSamples, krigingSamples, tikhonovSamples, ...
    physicsSamples, tikhonovCorrection, physicsCorrection, bmagLimits, correctionLimit)
% 功能：显示原始观测与回采样结果的符合程度，以及后处理在观测点的实际改变量。
% 布局：第一行显示原始观测及三种回采样 |B|；第二行突出显示两种后处理改变量。
    layout = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    plotSampleValues(layout, originalSamples, originalSamples.Bmag, ...
        '原始采样 |B|', bmagLimits, parula(256), '原始 |B| (T)');
    plotSampleValues(layout, originalSamples, krigingSamples.Bmag, ...
        'Kriging 回采样 |B|', bmagLimits, parula(256), '回采样 |B| (T)');
    plotSampleValues(layout, originalSamples, tikhonovSamples.Bmag, ...
        'First-order Tikhonov 回采样 |B|', bmagLimits, parula(256), '回采样 |B| (T)');
    plotSampleValues(layout, originalSamples, physicsSamples.Bmag, ...
        'Physics Constraint 回采样 |B|', bmagLimits, parula(256), '回采样 |B| (T)');
    plotSampleValues(layout, originalSamples, tikhonovCorrection, ...
        'Tikhonov - Kriging（原始采样点）', [-correctionLimit, correctionLimit], ...
        blueWhiteRedMap(256), '\Delta|B| (T)');
    plotSampleValues(layout, originalSamples, physicsCorrection, ...
        'Physics - Tikhonov（原始采样点）', [-correctionLimit, correctionLimit], ...
        blueWhiteRedMap(256), '\Delta|B| (T)');
    title(layout, '原始采样点数据保真与后处理改变量：原始采样参与 Lsample，Ground Truth 未参与优化');
end

function plotSampleValues(layout, originalSamples, values, titleText, limits, map, colorLabel)
% 功能：在原始观测坐标上显示标量值，统一控制色轴以便方法间直接比较。
    nexttile(layout);
    scatter(originalSamples.x, originalSamples.y, 56, values, 'filled', 'MarkerEdgeColor', 'k');
    axis image; xlim([min(originalSamples.x), max(originalSamples.x)]);
    ylim([min(originalSamples.y), max(originalSamples.y)]); caxis(limits); colormap(gca, map);
    xlabel('x (mm)'); ylabel('y (mm)'); title(titleText, 'Interpreter', 'none');
    colorbarHandle = colorbar; colorbarHandle.Label.String = colorLabel;
end

function map = blueWhiteRedMap(count)
% 功能：创建用于 signed error 的蓝-白-红发散色图。
    halfCount = floor(count / 2);
    blueToWhite = [linspace(0, 1, halfCount)', linspace(0.25, 1, halfCount)', ones(halfCount, 1)];
    whiteToRed = [ones(count - halfCount, 1), linspace(1, 0.15, count - halfCount)', ...
        linspace(1, 0.10, count - halfCount)'];
    map = [blueToWhite; whiteToRed];
end

function limits = sharedLimits(values)
% 功能：生成统一色轴范围，并防止常量场导致 caxis 报错。
    values = values(isfinite(values));
    limits = [min(values), max(values)];
    if limits(1) == limits(2)
        delta = max(abs(limits(1)) * 0.01, 1);
        limits = limits + [-delta, delta];
    end
end

function saveFigure(fig, folder, name, shouldSave, resolution)
% 功能：根据开关保存 PNG 图像。
    drawnow;
    if shouldSave
        exportgraphics(fig, fullfile(folder, [name '.png']), 'Resolution', resolution);
    end
end

function writePhysicsSummary(filePath, results, summary)
% 功能：写入数学正则化对照、无旋物理约束和三组评价指标的文本摘要。
    fileId = fopen(filePath, 'w');
    if fileId < 0
        warning('无法写入物理约束优化摘要：%s', filePath);
        return;
    end
    cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
    fprintf(fileId, 'Kriging + First-order Tikhonov + Physics Constraint 摘要\n');
    fprintf(fileId, 'Kriging 先验：%s\n', results.krigingResultFile);
    fprintf(fileId, '原始采样：%s\n', results.originalSampleFile);
    fprintf(fileId, 'Ground Truth：%s\n', results.groundTruthFile);
    fprintf(fileId, '统一优化与评价网格：%d×%d\n', results.optimizationGrid(1), results.optimizationGrid(2));
    fprintf(fileId, '%s\n\n', results.evaluationGridOperation);
    fprintf(fileId, ['标准一阶 Tikhonov 目标：L = (%.12g/2)||B-B_kriging||_F^2 + ' ...
        '(%.12g/2)||S(B)-B_sample||_F^2 + (%.12g/2)||grad(B)||_F^2\n'], ...
        results.mathematicalConfig.dataWeight, ...
        results.mathematicalConfig.originalSampleDataWeight, ...
        results.mathematicalConfig.tikhonovWeight);
    fprintf(fileId, ['物理约束目标：在标准一阶 Tikhonov 基础上增加 ' ...
        '(%.12g/2)||dBy/dx-dBx/dy||_F^2\n'], ...
        results.physicsConfig.curlWeight);
    fprintf(fileId, ['物理限制：单一 z 平面不能直接计算 Bzz=dBz/dz，因此不采用 Bzz=0 假设。' ...
        '物理项仅使用可观测的 curl_z=dBy/dx-dBx/dy，并假定测量平面无自由电流。\n\n']);
    fprintf(fileId, '原始采样点一致性（高分辨率结果线性回采样至原始采样坐标）：\n');
    for index = 1:height(results.originalSampleConsistency)
        row = results.originalSampleConsistency(index, :);
        fprintf(fileId, '  %s / %s: RMSE=%.12g T；MAE=%.12g T；MaxError=%.12g T\n', ...
            row.Method, row.Quantity, row.RMSE_at_OriginalSamples_T, ...
            row.MAE_at_OriginalSamples_T, row.MaxError_at_OriginalSamples_T);
    end
    fprintf(fileId, '\n原始采样点上相邻方法的 |B| 改变量：\n');
    for index = 1:height(results.originalSampleCorrectionSummary)
        row = results.originalSampleCorrectionSummary(index, :);
        fprintf(fileId, '  %s: RMS=%.12g T；MaxAbs=%.12g T\n', ...
            row.Correction, row.RMS_Bmag_T, row.MaxAbs_Bmag_T);
    end
    fprintf(fileId, '\n');
    for index = 1:height(summary)
        row = summary(index, :);
        fprintf(fileId, '%s\n', row.Method);
        fprintf(fileId, '  RMSE(|B|)=%.12g T；MAE(|B|)=%.12g T；MaxError(|B|)=%.12g T\n', ...
            row.RMSE_Bmag_T, row.MAE_Bmag_T, row.MaxError_Bmag_T);
        fprintf(fileId, '  PeakValueError=%.12g T；PeakPositionError=%.12g mm\n', ...
            row.PeakValueError_T, row.PeakPositionError_mm);
        fprintf(fileId, '  MeanAbsCurlZ=%.12g T/mm；MaxAbsCurlZ=%.12g T/mm\n', ...
            row.MeanAbsCurlZ_T_per_mm, row.MaxAbsCurlZ_T_per_mm);
    end
end

function isIncreasing = isStrictlyIncreasingAxis(values)
% 功能：检查插值源坐标是否严格递增。
% 说明：griddedInterpolant 支持非等距坐标轴，因此不要求 COMSOL 文本中的坐标
%       间隔完全相同。这样可兼容有限小数位导出造成的坐标量化误差。
    values = values(:);
    isIncreasing = numel(values) >= 2 && all(isfinite(values)) && all(diff(values) > 0);
end

function isMatch = coordinateVectorsMatch(firstValues, secondValues)
% 功能：在浮点容差内判断两个坐标向量是否逐点相同。
    isMatch = isequal(size(firstValues), size(secondValues));
    if ~isMatch
        return;
    end
    tolerance = max(1e-9, 1e-10 * max([1; abs(firstValues(:)); abs(secondValues(:))]));
    isMatch = all(abs(firstValues(:) - secondValues(:)) <= tolerance);
end

function assertFinite(values, name)
% 功能：阻止 NaN 或 Inf 进入优化、误差和绘图计算。
    nanCount = sum(isnan(values), 'all');
    infCount = sum(isinf(values), 'all');
    if nanCount > 0 || infCount > 0
        error('%s 含有 %d 个 NaN 和 %d 个 Inf，不能继续计算。', name, nanCount, infCount);
    end
end
