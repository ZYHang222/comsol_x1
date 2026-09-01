%% RBF_100_100：100×100 全数据训练的 Gaussian RBF 超分辨率重建
% 训练数据：原始 COMSOL 100×100 网格（共 10000 个点）
% 输出数据：可选 1000×1000 或 2000×2000 高分辨率网格
%
% 重要物理与算法约束：
%   1. 仅对 Bx、By、Bz 三个分量分别执行 Gaussian RBF。
%   2. |B| 由三分量重建结果计算，不直接对 |B| 插值。
%   3. Ct_2D 由三分量的 x/y 导数计算，不直接对 Ct 插值。
%   4. 数据只有 z=-20 mm 单一平面，Ct_2D 不包含 z 方向导数，
%      因而不能解释为完整三维磁梯度张量 Ct。
%
% 数值实现说明：规则 100×100 网格上的二维 Gaussian 核可以分离为
% K((x,y),(xi,yi)) = Kx(x,xi) * Ky(y,yi)。本脚本利用这一性质，
% 仅求解两个 100×100 核矩阵，数学上等价于使用全部 10000 个中心的
% 全局二维 Gaussian RBF，但避免构造难以承受的 10000×10000 稠密矩阵。

%% 1. 参数设置
clear;
clc;
close all;

inputFile = 'D:\YAN\大论文\x1\数据\100-100.txt';

% 可选超分辨率：1000 或 2000。1000×1000 共 100 万点，
% 2000×2000 共 400 万点，需要更多内存与运行时间。
reconstructionResolution = 1000;
allowedResolutions = [1000, 2000];

% epsilon 作用在归一化后的 x-y 坐标。多个候选值根据 |B| 与 Ct_2D
% 在原始 100×100 网格上的误差进行评价。
epsilonMultipliers = [0.50, 0.75, 1.00, 1.50, 2.00];
saveFigures = true;
figureResolution = 300;

if ~ismember(reconstructionResolution, allowedResolutions)
    error('reconstructionResolution 只能设置为 1000 或 2000。');
end
if ~isfile(inputFile)
    [fileName, filePath] = uigetfile({'*.txt', 'Text files'}, ...
        '选择 COMSOL 磁场数据文件');
    if isequal(fileName, 0)
        error('未选择输入数据文件，程序终止。');
    end
    inputFile = fullfile(filePath, fileName);
end

scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder)
    scriptFolder = pwd;
end
[~, baseName, ~] = fileparts(inputFile);
resultFolder = fullfile(scriptFolder, ...
    sprintf('RBF_results_%s_%dx%d', baseName, reconstructionResolution, reconstructionResolution));
if ~isfolder(resultFolder)
    mkdir(resultFolder);
end

fprintf('============================================================\n');
fprintf('100×100 全数据训练 Gaussian RBF 超分辨率重建\n');
fprintf('输入文件：%s\n', inputFile);
fprintf('训练网格：100×100；重建网格：%d×%d\n', ...
    reconstructionResolution, reconstructionResolution);
fprintf('============================================================\n');

%% 2. 读取 COMSOL 文本数据并检查实际采样网格
% 文件前 9 行为 COMSOL 元数据与变量名，第 10 行起为六列数值：
% x, y, z, Bx, By, Bz。
rawData = readmatrix(inputFile, 'FileType', 'text', 'NumHeaderLines', 9);
rawData = rawData(:, 1:6);
rawData = rawData(all(isfinite(rawData), 2), :);

x = rawData(:, 1);
y = rawData(:, 2);
z = rawData(:, 3);
Bx = rawData(:, 4);
By = rawData(:, 5);
Bz = rawData(:, 6);

xValues = unique(x, 'sorted')';
yValues = unique(y, 'sorted')';
zValues = unique(z, 'sorted')';
dxValues = diff(xValues);
dyValues = diff(yValues);

fprintf('\n[1] 数据与网格检查\n');
fprintf('数据点数：%d\n', numel(x));
fprintf('x 采样点数：%d；y 采样点数：%d；z 采样点数：%d\n', ...
    numel(xValues), numel(yValues), numel(zValues));
fprintf('x 范围：[%.12g, %.12g] mm；实际 dx 中位数：%.12g mm\n', ...
    min(xValues), max(xValues), median(dxValues));
fprintf('y 范围：[%.12g, %.12g] mm；实际 dy 中位数：%.12g mm\n', ...
    min(yValues), max(yValues), median(dyValues));
fprintf('z = %.12g mm\n', zValues(1));

if numel(x) ~= 10000 || numel(xValues) ~= 100 || numel(yValues) ~= 100 || numel(zValues) ~= 1
    error(['当前脚本要求 10000 个点、100×100 单一 z 平面；实际为 %d 个点、' ...
        '%d 个 x 值、%d 个 y 值、%d 个 z 值。'], ...
        numel(x), numel(xValues), numel(yValues), numel(zValues));
end

%% 3. 根据真实坐标建立 100×100 Ground Truth 网格
% 使用坐标索引建立矩阵，避免简单 reshape 造成 x/y 方向错位。
[X_original, Y_original] = meshgrid(xValues, yValues);
[~, xIndex] = ismember(x, xValues);
[~, yIndex] = ismember(y, yValues);
originalSize = [numel(yValues), numel(xValues)];

Bx_original = accumarray([yIndex, xIndex], Bx, originalSize, @mean, NaN);
By_original = accumarray([yIndex, xIndex], By, originalSize, @mean, NaN);
Bz_original = accumarray([yIndex, xIndex], Bz, originalSize, @mean, NaN);
if any(~isfinite([Bx_original(:); By_original(:); Bz_original(:)]))
    error('原始数据未覆盖完整规则 100×100 网格。');
end

% 原始 |B| 与 Ct_2D：Ct_2D 是平面内六个梯度元素的平方和。
Bmag_original = sqrt(Bx_original.^2 + By_original.^2 + Bz_original.^2);
[Gxx_original, Gxy_original] = gradient(Bx_original, xValues, yValues);
[Gyx_original, Gyy_original] = gradient(By_original, xValues, yValues);
[Gzx_original, Gzy_original] = gradient(Bz_original, xValues, yValues);
Ct_original = Gxx_original.^2 + Gxy_original.^2 + ...
    Gyx_original.^2 + Gyy_original.^2 + ...
    Gzx_original.^2 + Gzy_original.^2;

%% 4. 全部 100×100 点作为 RBF 中心，进行坐标归一化和 epsilon 扫描
% 此处不降采样：原始 100×100 的 Bx、By、Bz 全部用于 RBF 训练。
normalizer = createNormalizer(xValues, yValues);
[xNormalized, yNormalized] = normalizeAxes(xValues, yValues, normalizer);
nearestDistance = median([median(diff(xNormalized)), median(diff(yNormalized))]);
epsilonBase = 1 / max(nearestDistance, eps);
epsilonList = epsilonBase * epsilonMultipliers;

fprintf('\n[2] 全数据 RBF 参数\n');
fprintf('RBF 中心数：100×100 = %d\n', numel(X_original));
fprintf('归一化网格间距中位数：%.12g\n', nearestDistance);
fprintf('epsilon 候选值：%s\n', num2str(epsilonList, ' %.8g'));

epsilonTable = table(epsilonList(:), nan(numel(epsilonList), 1), ...
    nan(numel(epsilonList), 1), nan(numel(epsilonList), 1), ...
    nan(numel(epsilonList), 1), nan(numel(epsilonList), 1), ...
    'VariableNames', {'epsilon', 'RMSE_Bx', 'RMSE_By', 'RMSE_Bz', ...
    'RMSE_Bmag', 'RMSE_Ct2D'});

for k = 1:numel(epsilonList)
    candidate = fitSeparableGaussianRBF3(xNormalized, yNormalized, ...
        Bx_original, By_original, Bz_original, epsilonList(k), normalizer);
    [candidateB, candidateDx, candidateDy] = evaluateOnGrid(candidate, xValues, yValues);
    candidateBmag = sqrt(candidateB(:, :, 1).^2 + candidateB(:, :, 2).^2 + ...
        candidateB(:, :, 3).^2);
    candidateCt = candidateDx(:, :, 1).^2 + candidateDy(:, :, 1).^2 + ...
        candidateDx(:, :, 2).^2 + candidateDy(:, :, 2).^2 + ...
        candidateDx(:, :, 3).^2 + candidateDy(:, :, 3).^2;

    epsilonTable.RMSE_Bx(k) = calculateMetrics(Bx_original, candidateB(:, :, 1)).RMSE;
    epsilonTable.RMSE_By(k) = calculateMetrics(By_original, candidateB(:, :, 2)).RMSE;
    epsilonTable.RMSE_Bz(k) = calculateMetrics(Bz_original, candidateB(:, :, 3)).RMSE;
    epsilonTable.RMSE_Bmag(k) = calculateMetrics(Bmag_original, candidateBmag).RMSE;
    epsilonTable.RMSE_Ct2D(k) = calculateMetrics(Ct_original, candidateCt).RMSE;
end

% |B| 和 Ct_2D 量纲不同，因此各自归一化后再构成综合评分。
epsilonTable.Score = epsilonTable.RMSE_Bmag / ...
    max(max(epsilonTable.RMSE_Bmag), eps) + ...
    epsilonTable.RMSE_Ct2D / max(max(epsilonTable.RMSE_Ct2D), eps);
[~, bestIndex] = min(epsilonTable.Score);
bestEpsilon = epsilonTable.epsilon(bestIndex);

fprintf('\n[3] epsilon 扫描结果\n');
disp(epsilonTable);
fprintf('最优 epsilon = %.12g（归一化坐标）\n', bestEpsilon);

%% 5. 用全部 100×100 中心拟合最优 Gaussian RBF
model = fitSeparableGaussianRBF3(xNormalized, yNormalized, ...
    Bx_original, By_original, Bz_original, bestEpsilon, normalizer);

% 在原始 100×100 坐标重新计算一次，用于与 Ground Truth 严格逐点评价。
[B_reference, dBdx_reference, dBdy_reference] = evaluateOnGrid(model, xValues, yValues);
Bx_reference = B_reference(:, :, 1);
By_reference = B_reference(:, :, 2);
Bz_reference = B_reference(:, :, 3);
Bmag_reference = sqrt(Bx_reference.^2 + By_reference.^2 + Bz_reference.^2);
Ct_reference = dBdx_reference(:, :, 1).^2 + dBdy_reference(:, :, 1).^2 + ...
    dBdx_reference(:, :, 2).^2 + dBdy_reference(:, :, 2).^2 + ...
    dBdx_reference(:, :, 3).^2 + dBdy_reference(:, :, 3).^2;

%% 6. 在可选 1000×1000 或 2000×2000 网格上执行超分辨率重建
xHigh = linspace(min(xValues), max(xValues), reconstructionResolution);
yHigh = linspace(min(yValues), max(yValues), reconstructionResolution);

fprintf('\n[4] 开始 %d×%d 超分辨率重建，请耐心等待……\n', ...
    reconstructionResolution, reconstructionResolution);
[B_high, dBdx_high, dBdy_high] = evaluateOnGrid(model, xHigh, yHigh);

Bx_rbf = B_high(:, :, 1);
By_rbf = B_high(:, :, 2);
Bz_rbf = B_high(:, :, 3);
Bmag_rbf = sqrt(Bx_rbf.^2 + By_rbf.^2 + Bz_rbf.^2);
Ct_rbf = dBdx_high(:, :, 1).^2 + dBdy_high(:, :, 1).^2 + ...
    dBdx_high(:, :, 2).^2 + dBdy_high(:, :, 2).^2 + ...
    dBdx_high(:, :, 3).^2 + dBdy_high(:, :, 3).^2;

% 高分辨率导数中间量不再用于后续计算，释放内存以便绘图和保存。
clear dBdx_high dBdy_high B_high

%% 7. 在原始 100×100 对应坐标上计算重建指标
% 1000×1000/2000×2000 网格没有独立真值；因此所有定量误差均在原始
% 100×100 Ground Truth 坐标上评估，保证比较公平且一一对应。
metricBx = calculateMetrics(Bx_original, Bx_reference);
metricBy = calculateMetrics(By_original, By_reference);
metricBz = calculateMetrics(Bz_original, Bz_reference);
metricBmag = calculateMetrics(Bmag_original, Bmag_reference);
metricCt = calculateMetrics(Ct_original, Ct_reference);

metricTable = table(["Bx"; "By"; "Bz"; "|B|"; "Ct_2D"], ...
    [metricBx.RMSE; metricBy.RMSE; metricBz.RMSE; metricBmag.RMSE; metricCt.RMSE], ...
    [metricBx.MAE; metricBy.MAE; metricBz.MAE; metricBmag.MAE; metricCt.MAE], ...
    [metricBx.MaxError; metricBy.MaxError; metricBz.MaxError; metricBmag.MaxError; metricCt.MaxError], ...
    [metricBx.R2; metricBy.R2; metricBz.R2; metricBmag.R2; metricCt.R2], ...
    'VariableNames', {'Variable', 'RMSE', 'MAE', 'MaxError', 'R2'});

fprintf('\n[5] 原始 100×100 坐标上的重建指标\n');
disp(metricTable);

%% 8. 原始与高分辨率 RBF 的峰值位置和幅值比较
peakBmag_original = findPeak(Bmag_original, xValues, yValues);
peakBmag_rbf = findPeak(Bmag_rbf, xHigh, yHigh);
peakCt_original = findPeak(Ct_original, xValues, yValues);
peakCt_rbf = findPeak(Ct_rbf, xHigh, yHigh);

fprintf('\n[6] 峰值比较\n');
printPeakComparison('|B|', peakBmag_original, peakBmag_rbf);
printPeakComparison('Ct_2D', peakCt_original, peakCt_rbf);

%% 9. 图 1：原始 100×100 与高分辨率 RBF 的核心成像对比
% 同类物理量共享 xlim、ylim、caxis 和 colormap，便于客观观察细节变化。
bmagLimits = finiteLimits([Bmag_original(:); Bmag_rbf(:)]);
ctLimits = finiteLimits([Ct_original(:); Ct_rbf(:)]);

figCore = figure('Color', 'w', 'Position', [60, 80, 1380, 920], ...
    'Name', '原始与超分辨率 RBF 核心成像');
layout = tiledlayout(figCore, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
colormap(figCore, parula(256));
plotImageTile(layout, xValues, yValues, Bmag_original, ...
    '原始磁场模值 |B|（100×100）', '|B| (T)', bmagLimits);
plotImageTile(layout, xHigh, yHigh, Bmag_rbf, ...
    sprintf('RBF 重建磁场模值 |B|（%d×%d）', reconstructionResolution, reconstructionResolution), ...
    '|B| (T)', bmagLimits);
plotImageTile(layout, xValues, yValues, Ct_original, ...
    '原始二维磁梯度平方和 Ct_2D（100×100）', 'Ct_2D (T^2/mm^2)', ctLimits);
plotImageTile(layout, xHigh, yHigh, Ct_rbf, ...
    sprintf('RBF 重建 Ct_2D（%d×%d）', reconstructionResolution, reconstructionResolution), ...
    'Ct_2D (T^2/mm^2)', ctLimits);
title(layout, sprintf('全 100×100 点训练；z = %.12g mm；epsilon = %.8g', ...
    zValues(1), bestEpsilon));
saveFigure(figCore, resultFolder, '01_核心对比_Bmag_Ct2D', saveFigures, figureResolution);

%% 10. 图 2：中心 x/y 剖面线，比较原始与超分辨率结果
[~, originalCenterRow] = min(abs(yValues));
[~, originalCenterColumn] = min(abs(xValues));
[~, highCenterRow] = min(abs(yHigh));
[~, highCenterColumn] = min(abs(xHigh));

figProfile = figure('Color', 'w', 'Position', [80, 100, 1330, 820], ...
    'Name', '原始与超分辨率 RBF 中心剖面');
profileLayout = tiledlayout(figProfile, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile(profileLayout);
plot(xValues, Bmag_original(originalCenterRow, :), 'k-', 'LineWidth', 1.5); hold on;
plot(xHigh, Bmag_rbf(highCenterRow, :), 'r-', 'LineWidth', 1.2); grid on;
xlabel('x (mm)'); ylabel('|B| (T)'); title('|B| 的 x 方向中心剖面');
legend('原始 100×100', sprintf('RBF %d×%d', reconstructionResolution, reconstructionResolution), 'Location', 'best');

nexttile(profileLayout);
plot(xValues, Ct_original(originalCenterRow, :), 'k-', 'LineWidth', 1.5); hold on;
plot(xHigh, Ct_rbf(highCenterRow, :), 'r-', 'LineWidth', 1.2); grid on;
xlabel('x (mm)'); ylabel('Ct_2D (T^2/mm^2)'); title('Ct_2D 的 x 方向中心剖面');
legend('原始 100×100', sprintf('RBF %d×%d', reconstructionResolution, reconstructionResolution), 'Location', 'best');

nexttile(profileLayout);
plot(yValues, Bmag_original(:, originalCenterColumn), 'k-', 'LineWidth', 1.5); hold on;
plot(yHigh, Bmag_rbf(:, highCenterColumn), 'r-', 'LineWidth', 1.2); grid on;
xlabel('y (mm)'); ylabel('|B| (T)'); title('|B| 的 y 方向中心剖面');
legend('原始 100×100', sprintf('RBF %d×%d', reconstructionResolution, reconstructionResolution), 'Location', 'best');

nexttile(profileLayout);
plot(yValues, Ct_original(:, originalCenterColumn), 'k-', 'LineWidth', 1.5); hold on;
plot(yHigh, Ct_rbf(:, highCenterColumn), 'r-', 'LineWidth', 1.2); grid on;
xlabel('y (mm)'); ylabel('Ct_2D (T^2/mm^2)'); title('Ct_2D 的 y 方向中心剖面');
legend('原始 100×100', sprintf('RBF %d×%d', reconstructionResolution, reconstructionResolution), 'Location', 'best');
saveFigure(figProfile, resultFolder, '02_中心剖面线对比', saveFigures, figureResolution);

%% 11. 图 3：epsilon 对 |B| 与 Ct_2D 的原始网格误差影响
figEpsilon = figure('Color', 'w', 'Position', [170, 210, 1120, 460], ...
    'Name', 'epsilon 参数扫描');
epsilonLayout = tiledlayout(figEpsilon, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile(epsilonLayout);
plot(epsilonTable.epsilon, epsilonTable.RMSE_Bmag, 'o-', ...
    'LineWidth', 1.5, 'MarkerFaceColor', [0.15, 0.45, 0.75]); grid on;
xlabel('epsilon（归一化坐标）'); ylabel('|B| RMSE (T)'); title('epsilon - RMSE(|B|)');
nexttile(epsilonLayout);
plot(epsilonTable.epsilon, epsilonTable.RMSE_Ct2D, 'o-', ...
    'LineWidth', 1.5, 'MarkerFaceColor', [0.80, 0.30, 0.15]); grid on;
xlabel('epsilon（归一化坐标）'); ylabel('Ct_2D RMSE (T^2/mm^2)'); title('epsilon - RMSE(Ct_2D)');
saveFigure(figEpsilon, resultFolder, '03_epsilon_RMSE扫描', saveFigures, figureResolution);

%% 12. 保存高分辨率数据、原始网格指标和实验摘要
results = struct();
results.inputFile = inputFile;
results.trainingResolution = [100, 100];
results.reconstructionResolution = [reconstructionResolution, reconstructionResolution];
results.zValue_mm = zValues(1);
results.xOriginal_mm = xValues;
results.yOriginal_mm = yValues;
results.xHigh_mm = xHigh;
results.yHigh_mm = yHigh;
results.dxOriginal_mm = median(dxValues);
results.dyOriginal_mm = median(dyValues);
results.bestEpsilon = bestEpsilon;
results.epsilonTable = epsilonTable;
results.metricTableOnOriginalGrid = metricTable;
results.Bmag_original = Bmag_original;
results.Ct2D_original = Ct_original;
results.Bx_reference = Bx_reference;
results.By_reference = By_reference;
results.Bz_reference = Bz_reference;
results.Bmag_reference = Bmag_reference;
results.Ct2D_reference = Ct_reference;
results.Bx_rbf_high = Bx_rbf;
results.By_rbf_high = By_rbf;
results.Bz_rbf_high = Bz_rbf;
results.Bmag_rbf_high = Bmag_rbf;
results.Ct2D_rbf_high = Ct_rbf;
results.peakBmag_original = peakBmag_original;
results.peakBmag_rbf = peakBmag_rbf;
results.peakCt2D_original = peakCt_original;
results.peakCt2D_rbf = peakCt_rbf;

save(fullfile(resultFolder, 'RBF_100_100_超分辨率结果.mat'), 'results', '-v7.3');
writetable(metricTable, fullfile(resultFolder, 'RBF_100_100_原始网格误差指标.csv'), ...
    'Encoding', 'UTF-8');
writetable(epsilonTable, fullfile(resultFolder, 'RBF_100_100_epsilon扫描.csv'), ...
    'Encoding', 'UTF-8');
writeSummary(fullfile(resultFolder, 'RBF_100_100_超分辨率摘要.txt'), results);

fprintf('\n完成：100×100 全数据训练 → %d×%d RBF 超分辨率重建。\n', ...
    reconstructionResolution, reconstructionResolution);
fprintf('结果保存目录：%s\n', resultFolder);

%% 局部函数
function normalizer = createNormalizer(xValues, yValues)
% 为 x/y 分别建立归一化参数，使 RBF epsilon 与 mm 绝对坐标尺度解耦。
    normalizer.xMin = min(xValues);
    normalizer.xScale = max(xValues) - min(xValues);
    normalizer.yMin = min(yValues);
    normalizer.yScale = max(yValues) - min(yValues);
    if normalizer.xScale == 0 || normalizer.yScale == 0
        error('x 或 y 坐标范围为零，无法建立二维 RBF。');
    end
end

function [xNormalized, yNormalized] = normalizeAxes(xValues, yValues, normalizer)
    xNormalized = (double(xValues) - normalizer.xMin) / normalizer.xScale;
    yNormalized = (double(yValues) - normalizer.yMin) / normalizer.yScale;
end

function model = fitSeparableGaussianRBF3(xn, yn, bx, by, bz, epsilon, normalizer)
% 在规则网格上拟合可分离的二维 Gaussian RBF。
% 对任一分量 V，有 V = Ky * W * Kx'，故 W = Ky\V/Kx'。
% W 的三层分别存储 Bx、By、Bz 的独立 RBF 权重。
    Kx = exp(-(epsilon^2) * (xn(:) - xn(:)').^2);
    Ky = exp(-(epsilon^2) * (yn(:) - yn(:)').^2);
    regularization = max(1e-12, 1e-10 * max(norm(Kx, inf), norm(Ky, inf)));
    Kx = Kx + regularization * eye(size(Kx));
    Ky = Ky + regularization * eye(size(Ky));

    model.weights = zeros(numel(yn), numel(xn), 3);
    model.weights(:, :, 1) = Ky \ bx / Kx';
    model.weights(:, :, 2) = Ky \ by / Kx';
    model.weights(:, :, 3) = Ky \ bz / Kx';
    model.xCenters = xn(:)';
    model.yCenters = yn(:)';
    model.epsilon = epsilon;
    model.normalizer = normalizer;
end

function [values, dfdx, dfdy] = evaluateOnGrid(model, xQuery, yQuery)
% 在任意规则 x-y 网格上计算三分量 RBF 值与解析 x/y 导数。
% 输出数组尺寸为 [numel(yQuery), numel(xQuery), 3]。
    [xq, yq] = normalizeAxes(xQuery, yQuery, model.normalizer);
    epsilon = model.epsilon;
    deltaX = xq(:) - model.xCenters;
    deltaY = yq(:) - model.yCenters;
    Kx = exp(-(epsilon^2) * deltaX.^2);
    Ky = exp(-(epsilon^2) * deltaY.^2);
    dKx = (-2 * epsilon^2 * deltaX .* Kx) / model.normalizer.xScale;
    dKy = (-2 * epsilon^2 * deltaY .* Ky) / model.normalizer.yScale;

    outputSize = [numel(yQuery), numel(xQuery), 3];
    values = zeros(outputSize);
    dfdx = zeros(outputSize);
    dfdy = zeros(outputSize);
    for component = 1:3
        weights = model.weights(:, :, component);
        values(:, :, component) = Ky * weights * Kx';
        dfdx(:, :, component) = Ky * weights * dKx';
        dfdy(:, :, component) = dKy * weights * Kx';
    end
end

function metric = calculateMetrics(trueValue, predictedValue)
% 计算原始 100×100 坐标上的 RMSE、MAE、最大绝对误差与 R²。
    trueValue = double(trueValue(:));
    predictedValue = double(predictedValue(:));
    valid = isfinite(trueValue) & isfinite(predictedValue);
    trueValue = trueValue(valid);
    predictedValue = predictedValue(valid);
    residual = predictedValue - trueValue;
    metric.RMSE = sqrt(mean(residual.^2));
    metric.MAE = mean(abs(residual));
    metric.MaxError = max(abs(residual));
    totalVariance = sum((trueValue - mean(trueValue)).^2);
    if totalVariance <= eps
        metric.R2 = NaN;
    else
        metric.R2 = 1 - sum(residual.^2) / totalVariance;
    end
end

function peak = findPeak(imageData, xValues, yValues)
% 返回图像最大值及其在真实 x-y 坐标中的位置。
    [peak.value, index] = max(imageData(:));
    [row, column] = ind2sub(size(imageData), index);
    peak.x = xValues(column);
    peak.y = yValues(row);
end

function printPeakComparison(name, originalPeak, rbfPeak)
% 输出原始与 RBF 的峰值变化率和异常中心位置偏移。
    changeRate = 100 * (rbfPeak.value - originalPeak.value) / ...
        max(abs(originalPeak.value), eps);
    positionShift = hypot(rbfPeak.x - originalPeak.x, rbfPeak.y - originalPeak.y);
    fprintf('%s 原始峰值：%.10g，位置：(%.10g, %.10g) mm\n', ...
        name, originalPeak.value, originalPeak.x, originalPeak.y);
    fprintf('%s RBF 峰值： %.10g，位置：(%.10g, %.10g) mm\n', ...
        name, rbfPeak.value, rbfPeak.x, rbfPeak.y);
    fprintf('%s 峰值变化率：%.6f%%；位置偏移：%.10g mm\n', ...
        name, changeRate, positionShift);
end

function limits = finiteLimits(values)
% 为原始/RBF 同类成像建立统一的有限 colorbar 范围。
    values = values(isfinite(values));
    limits = [min(values), max(values)];
    if limits(1) == limits(2)
        delta = max(abs(limits(1)) * 0.01, 1);
        limits = limits + [-delta, delta];
    end
end

function plotImageTile(layout, xValues, yValues, imageData, titleText, colorLabel, limits)
% 绘制单个成像图，并强制设置统一 x/y 范围和 colorbar 范围。
    nexttile(layout);
    imagesc(xValues, yValues, imageData);
    set(gca, 'YDir', 'normal');
    axis image;
    xlim([min(xValues), max(xValues)]);
    ylim([min(yValues), max(yValues)]);
    caxis(limits);
    xlabel('x (mm)');
    ylabel('y (mm)');
    title(titleText, 'Interpreter', 'none');
    colorbarHandle = colorbar;
    colorbarHandle.Label.String = colorLabel;
end

function saveFigure(fig, folder, fileStem, shouldSave, resolution)
% 按设定 DPI 导出 PNG；不影响 MATLAB 图窗的交互显示。
    if shouldSave
        exportgraphics(fig, fullfile(folder, [fileStem '.png']), 'Resolution', resolution);
    end
end

function writeSummary(filePath, results)
% 将训练/重建尺度、最优 epsilon 与原始网格误差写入文本摘要。
    fileId = fopen(filePath, 'w');
    if fileId < 0
        warning('无法写入摘要：%s', filePath);
        return;
    end
    cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
    fprintf(fileId, '100×100 全数据 Gaussian RBF 超分辨率重建摘要\n');
    fprintf(fileId, '输入文件：%s\n', results.inputFile);
    fprintf(fileId, '训练网格：%d×%d；重建网格：%d×%d\n', ...
        results.trainingResolution(1), results.trainingResolution(2), ...
        results.reconstructionResolution(1), results.reconstructionResolution(2));
    fprintf(fileId, 'z：%.12g mm；最优 epsilon：%.12g\n\n', ...
        results.zValue_mm, results.bestEpsilon);
    for k = 1:height(results.metricTableOnOriginalGrid)
        fprintf(fileId, '%s: RMSE=%.12g, MAE=%.12g, MaxError=%.12g, R2=%.12g\n', ...
            results.metricTableOnOriginalGrid.Variable(k), ...
            results.metricTableOnOriginalGrid.RMSE(k), ...
            results.metricTableOnOriginalGrid.MAE(k), ...
            results.metricTableOnOriginalGrid.MaxError(k), ...
            results.metricTableOnOriginalGrid.R2(k));
    end
end
