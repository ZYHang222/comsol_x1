%% RBF_10_10_to_1000：10×10 或 100×100 磁场数据的 Gaussian RBF 超分辨率重建
% 手动指定 COMSOL 文本源文件，可使用全部 10×10 或 100×100 个点训练，
% 并重建为 1000×1000 网格。|B| 由 Bx、By、Bz 计算；Ct_2D 只使用 x/y 导数。
% 输入为单一 z 平面时，Ct_2D 不代表完整三维磁梯度张量 Ct。

%% 1. 参数设置与源文件路径
clear;
clc;
close all;

reconstructionResolution = 1000; % 可改为 2000，需更多内存。
allowedResolutions = [1000, 2000];
epsilonMultipliers = [0.50, 0.75, 1.00, 1.50, 2.00];
saveFigures = true;
figureResolution = 300;

% 手动修改此处的源文件路径：可填写 10×10 或 100×100 的 COMSOL .txt 文件。
inputFile = 'D:\YAN\大论文\x1\数据\10-10.txt';

if ~ismember(reconstructionResolution, allowedResolutions)
    error('reconstructionResolution 只能设置为 1000 或 2000。');
end

scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder)
    scriptFolder = pwd;
end
if ~isfile(inputFile)
    error('找不到 inputFile 指定的源文件：%s', inputFile);
end
[~, baseName, ~] = fileparts(inputFile);
resultFolder = fullfile(scriptFolder, sprintf('RBF_results_%s_%dx%d', ...
    baseName, reconstructionResolution, reconstructionResolution));
if ~isfolder(resultFolder)
    mkdir(resultFolder);
end

fprintf('============================================================\n');
fprintf('Gaussian RBF 超分辨率重建（支持 10×10 或 100×100 训练数据）\n');
fprintf('输入文件：%s\n', inputFile);
fprintf('重建网格：%d×%d\n', reconstructionResolution, reconstructionResolution);
fprintf('结果目录：%s\n', resultFolder);
fprintf('============================================================\n');

%% 2. 读取数据并验证为 10×10 或 100×100 单 z 平面
% 当前 COMSOL .txt 固定为前 9 行元数据/变量名，第 10 行起为六列数值。
% 显式指定头行数，避免 readmatrix 自动推断范围导致磁场列读取异常。
rawData = readmatrix(inputFile, 'FileType', 'text', 'NumHeaderLines', 9);
if size(rawData, 2) < 6
    error('所选文件的数值列少于 6 列，无法读取 x、y、z、Bx、By、Bz。');
end
rawData = rawData(:, 1:6);
rawData = rawData(all(isfinite(rawData), 2), :);
if isempty(rawData)
    error('未在所选文件中读取到六列有效数值数据。');
end

x = rawData(:, 1);
y = rawData(:, 2);
z = rawData(:, 3);
Bx = rawData(:, 4);
By = rawData(:, 5);
Bz = rawData(:, 6);
xValues = unique(x, 'sorted')';
yValues = unique(y, 'sorted')';
zValues = unique(z, 'sorted')';
trainingGridSize = [numel(xValues), numel(yValues)];
trainingGridText = sprintf('%d×%d', trainingGridSize(1), trainingGridSize(2));

fprintf('\n[1] 数据与网格检查\n');
fprintf('数据点数：%d；x 点数：%d；y 点数：%d；z 点数：%d\n', ...
    numel(x), numel(xValues), numel(yValues), numel(zValues));
isSupportedGrid = isequal(trainingGridSize, [10, 10]) || isequal(trainingGridSize, [100, 100]);
if numel(x) ~= prod(trainingGridSize) || ~isSupportedGrid || numel(zValues) ~= 1
    error(['当前脚本仅支持完整 10×10 或 100×100 单一 z 平面；实际为 %d 个点、' ...
        '%d 个 x 值、%d 个 y 值、%d 个 z 值。'], ...
        numel(x), numel(xValues), numel(yValues), numel(zValues));
end
if ~isRegularAxis(xValues) || ~isRegularAxis(yValues)
    error('x 或 y 坐标不是等间距规则网格，不能使用本规则网格 RBF 脚本。');
end
fprintf('x 范围：[%.12g, %.12g] mm；dx：%.12g mm\n', ...
    min(xValues), max(xValues), median(diff(xValues)));
fprintf('y 范围：[%.12g, %.12g] mm；dy：%.12g mm；z：%.12g mm\n', ...
    min(yValues), max(yValues), median(diff(yValues)), zValues(1));
fprintf('已识别训练网格：%s\n', trainingGridText);

%% 3. 按真实坐标建立原始训练网格
% 使用坐标索引而不是 reshape，避免源文件行顺序不同造成 x/y 方向错位。
[~, xIndex] = ismember(x, xValues);
[~, yIndex] = ismember(y, yValues);
originalSize = [numel(yValues), numel(xValues)];
BxOriginal = accumarray([yIndex, xIndex], Bx, originalSize, @mean, NaN);
ByOriginal = accumarray([yIndex, xIndex], By, originalSize, @mean, NaN);
BzOriginal = accumarray([yIndex, xIndex], Bz, originalSize, @mean, NaN);
if any(~isfinite([BxOriginal(:); ByOriginal(:); BzOriginal(:)]))
    error('原始数据未覆盖完整训练网格，或存在重复/缺失坐标。');
end

BmagOriginal = sqrt(BxOriginal.^2 + ByOriginal.^2 + BzOriginal.^2);
assertFiniteImage(BmagOriginal, '原始 |B|');
[dBxDxOriginal, dBxDyOriginal] = gradient(BxOriginal, xValues, yValues);
[dByDxOriginal, dByDyOriginal] = gradient(ByOriginal, xValues, yValues);
[dBzDxOriginal, dBzDyOriginal] = gradient(BzOriginal, xValues, yValues);
CtOriginal = dBxDxOriginal.^2 + dBxDyOriginal.^2 + ...
    dByDxOriginal.^2 + dByDyOriginal.^2 + ...
    dBzDxOriginal.^2 + dBzDyOriginal.^2;
assertFiniteImage(CtOriginal, '原始 Ct_2D');
fprintf('|B| 原始范围：[%.12g, %.12g] T\n', min(BmagOriginal, [], 'all'), max(BmagOriginal, [], 'all'));

%% 4. 通过独立验证节点选择 epsilon，再使用全部数据训练
% exp(-epsilon^2*((x-xi)^2+(y-yi)^2)) 可分解为 Kx(x,xi)*Ky(y,yi)。
normalizer = createNormalizer(xValues, yValues);
[xNormalized, yNormalized] = normalizeAxes(xValues, yValues, normalizer);
nearestDistance = median([median(diff(xNormalized)), median(diff(yNormalized))]);
epsilonList = epsilonMultipliers / max(nearestDistance, eps);

% 不能用训练节点选择 epsilon，因为插值 RBF 会在训练点近似严格通过。
% 因此保留每隔一个节点组成的规则子网格训练，并在其余节点进行验证。
xTrainIndex = unique([1:2:numel(xValues), numel(xValues)]);
yTrainIndex = unique([1:2:numel(yValues), numel(yValues)]);
validationMask = true(originalSize);
validationMask(yTrainIndex, xTrainIndex) = false;
if ~any(validationMask, 'all')
    error('无法从当前网格构造 epsilon 的独立验证节点。');
end

epsilonTable = table('Size', [numel(epsilonList), 3], ...
    'VariableTypes', {'double', 'double', 'double'}, ...
    'VariableNames', {'epsilon', 'RMSE_BxByBz_Validation', 'RMSE_Bmag_Validation'});
epsilonTable.epsilon = epsilonList(:);

for index = 1:numel(epsilonList)
    candidateEpsilon = epsilonList(index);
    candidateCoefficients = fitSeparableRBF(xNormalized(xTrainIndex), yNormalized(yTrainIndex), ...
        BxOriginal(yTrainIndex, xTrainIndex), ByOriginal(yTrainIndex, xTrainIndex), ...
        BzOriginal(yTrainIndex, xTrainIndex), candidateEpsilon);
    [candidateField, ~, ~] = evaluateSeparableRBF( ...
        xNormalized(xTrainIndex), yNormalized(yTrainIndex), xNormalized, yNormalized, ...
        candidateCoefficients, candidateEpsilon);
    candidateBmag = sqrt(sum(candidateField.^2, 3));
    componentDifference = candidateField - cat(3, BxOriginal, ByOriginal, BzOriginal);
    componentValidationMask = repmat(validationMask, 1, 1, 3);
    epsilonTable.RMSE_BxByBz_Validation(index) = ...
        sqrt(mean(componentDifference(componentValidationMask).^2));
    epsilonTable.RMSE_Bmag_Validation(index) = ...
        calculateMetrics(BmagOriginal(validationMask), candidateBmag(validationMask)).RMSE;
end

% 分量误差和模值误差按幅值范围归一化后共同决定最优 epsilon。
componentScale = max(range([BxOriginal(:); ByOriginal(:); BzOriginal(:)]), eps);
score = epsilonTable.RMSE_BxByBz_Validation / componentScale + ...
    epsilonTable.RMSE_Bmag_Validation / max(range(BmagOriginal(:)), eps);
[~, bestIndex] = min(score);
bestEpsilon = epsilonTable.epsilon(bestIndex);
if ~isfinite(bestEpsilon) || bestEpsilon <= 0
    error('自动选择的 epsilon 无效（%.12g）。请检查 epsilon 候选值和验证指标。', bestEpsilon);
end
coefficients = fitSeparableRBF(xNormalized, yNormalized, ...
    BxOriginal, ByOriginal, BzOriginal, bestEpsilon);
fprintf('\n[2] epsilon 自动选择\n');
disp(epsilonTable);
fprintf('最优 epsilon：%.12g\n', bestEpsilon);

%% 5. 计算原始训练节点上的重建指标
[fieldReference, dFieldDxReference, dFieldDyReference] = evaluateSeparableRBF( ...
    xNormalized, yNormalized, xNormalized, yNormalized, coefficients, bestEpsilon);
[BxReference, ByReference, BzReference, BmagReference, CtReference] = ...
    calculateDerivedFields(fieldReference, dFieldDxReference, dFieldDyReference, normalizer);
metricTable = buildMetricTable(BxOriginal, ByOriginal, BzOriginal, BmagOriginal, CtOriginal, ...
    BxReference, ByReference, BzReference, BmagReference, CtReference);
fprintf('\n[3] 原始 %s 坐标上的重建指标\n', trainingGridText);
disp(metricTable);

%% 6. 重建高分辨率规则网格
xHigh = linspace(min(xValues), max(xValues), reconstructionResolution);
yHigh = linspace(min(yValues), max(yValues), reconstructionResolution);
[xHighNormalized, yHighNormalized] = normalizeAxes(xHigh, yHigh, normalizer);
fprintf('\n[4] 正在重建 %d×%d 网格...\n', reconstructionResolution, reconstructionResolution);
% 先仅计算三分量场值，使 |B| 图可以尽早显示；Ct_2D 的导数另行计算。
fieldHigh = evaluateFieldOnGrid( ...
    xNormalized, yNormalized, xHighNormalized, yHighNormalized, coefficients, bestEpsilon);
BxRbf = fieldHigh(:, :, 1);
ByRbf = fieldHigh(:, :, 2);
BzRbf = fieldHigh(:, :, 3);
BmagRbf = sqrt(BxRbf.^2 + ByRbf.^2 + BzRbf.^2);
clear fieldHigh
assertFiniteImage(BmagRbf, 'RBF 重建 |B|');
fprintf('|B| 重建范围：[%.12g, %.12g] T\n', min(BmagRbf, [], 'all'), max(BmagRbf, [], 'all'));

%% 7. 计算 Ct_2D
bmagLimits = finiteLimits([BmagOriginal(:); BmagRbf(:)]);
% 以下计算六个 x/y 导数并构造 Ct_2D。
fprintf('正在计算 Ct_2D 所需的 x/y 导数...\n');
[dFieldDxHigh, dFieldDyHigh] = evaluateDerivativesOnGrid( ...
    xNormalized, yNormalized, xHighNormalized, yHighNormalized, coefficients, bestEpsilon);
CtRbf = calculateCt2D(dFieldDxHigh, dFieldDyHigh, normalizer);
clear dFieldDxHigh dFieldDyHigh
assertFiniteImage(CtRbf, 'RBF 重建 Ct_2D');

%% 8. 输出 |B| 与 Ct_2D 核心对比和剖面图（不生成误差图或 epsilon 扫描图）
ctLimits = finiteLimits([CtOriginal(:); CtRbf(:)]);
figCore = figure('Color', 'w', 'Position', [60, 80, 1380, 920], ...
    'Name', [trainingGridText ' 原始与 RBF 超分辨率核心成像']);
layout = tiledlayout(figCore, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
colormap(figCore, parula(256));
plotImageTile(layout, xValues, yValues, BmagOriginal, ...
    ['原始磁场模值 |B|（' trainingGridText '）'], '|B| (T)', bmagLimits);
plotImageTile(layout, xHigh, yHigh, BmagRbf, ...
    sprintf('RBF 重建磁场模值 |B|（%d×%d）', reconstructionResolution, reconstructionResolution), '|B| (T)', bmagLimits);
plotImageTile(layout, xValues, yValues, CtOriginal, ...
    ['原始二维磁梯度平方和 Ct_2D（' trainingGridText '）'], 'Ct_2D (T^2/mm^2)', ctLimits);
plotImageTile(layout, xHigh, yHigh, CtRbf, ...
    sprintf('RBF 重建 Ct_2D（%d×%d）', reconstructionResolution, reconstructionResolution), 'Ct_2D (T^2/mm^2)', ctLimits);
title(layout, sprintf('全 %s 点训练；z = %.12g mm；epsilon = %.8g', ...
    trainingGridText, zValues(1), bestEpsilon));
saveFigure(figCore, resultFolder, '01_核心对比_Bmag_Ct2D', saveFigures, figureResolution);
drawnow;

[~, originalCenterRow] = min(abs(yValues));
[~, originalCenterColumn] = min(abs(xValues));

% 原始网格未必恰好有 x=0 或 y=0。剖面必须在同一物理坐标线上比较，
% 因而直接在原始最近中心坐标处重新计算 RBF 剖面，而不从高分辨率
% 网格中选取另一个“接近零”的行/列。
[xProfileNormalized, yProfileXNormalized] = normalizeAxes( ...
    xHigh, yValues(originalCenterRow), normalizer);
[fieldProfileX, dFieldDxProfileX, dFieldDyProfileX] = evaluateSeparableRBF( ...
    xNormalized, yNormalized, xProfileNormalized, yProfileXNormalized, coefficients, bestEpsilon);
[~, ~, ~, bmagProfileX, ctProfileX] = calculateDerivedFields( ...
    fieldProfileX, dFieldDxProfileX, dFieldDyProfileX, normalizer);

[xProfileYNormalized, yProfileNormalized] = normalizeAxes( ...
    xValues(originalCenterColumn), yHigh, normalizer);
[fieldProfileY, dFieldDxProfileY, dFieldDyProfileY] = evaluateSeparableRBF( ...
    xNormalized, yNormalized, xProfileYNormalized, yProfileNormalized, coefficients, bestEpsilon);
[~, ~, ~, bmagProfileY, ctProfileY] = calculateDerivedFields( ...
    fieldProfileY, dFieldDxProfileY, dFieldDyProfileY, normalizer);

bmagProfileX = reshape(bmagProfileX, 1, []);
ctProfileX = reshape(ctProfileX, 1, []);
bmagProfileY = reshape(bmagProfileY, [], 1);
ctProfileY = reshape(ctProfileY, [], 1);
figProfile = figure('Color', 'w', 'Position', [80, 100, 1330, 820], ...
    'Name', [trainingGridText ' 原始与 RBF 超分辨率中心剖面']);
profileLayout = tiledlayout(figProfile, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
plotProfileTile(profileLayout, xValues, BmagOriginal(originalCenterRow, :), xHigh, bmagProfileX, ...
    'x (mm)', '|B| (T)', '|B| 的 x 方向中心剖面', trainingGridText, reconstructionResolution);
plotProfileTile(profileLayout, xValues, CtOriginal(originalCenterRow, :), xHigh, ctProfileX, ...
    'x (mm)', 'Ct_2D (T^2/mm^2)', 'Ct_2D 的 x 方向中心剖面', trainingGridText, reconstructionResolution);
plotProfileTile(profileLayout, yValues, BmagOriginal(:, originalCenterColumn), yHigh, bmagProfileY, ...
    'y (mm)', '|B| (T)', '|B| 的 y 方向中心剖面', trainingGridText, reconstructionResolution);
plotProfileTile(profileLayout, yValues, CtOriginal(:, originalCenterColumn), yHigh, ctProfileY, ...
    'y (mm)', 'Ct_2D (T^2/mm^2)', 'Ct_2D 的 y 方向中心剖面', trainingGridText, reconstructionResolution);
saveFigure(figProfile, resultFolder, '02_中心剖面线对比', saveFigures, figureResolution);
drawnow;

%% 9. 保存 MAT、CSV 和文字摘要
results = struct();
results.inputFile = inputFile;
results.trainingResolution = trainingGridSize;
results.reconstructionResolution = [reconstructionResolution, reconstructionResolution];
results.zValue_mm = zValues(1);
results.xOriginal_mm = xValues;
results.yOriginal_mm = yValues;
results.xHigh_mm = xHigh;
results.yHigh_mm = yHigh;
results.bestEpsilon = bestEpsilon;
results.epsilonTable = epsilonTable;
results.metricTableOnOriginalGrid = metricTable;
results.Bmag_original = BmagOriginal;
results.Ct2D_original = CtOriginal;
results.Bx_rbf = BxRbf;
results.By_rbf = ByRbf;
results.Bz_rbf = BzRbf;
results.Bmag_rbf = BmagRbf;
results.Ct2D_rbf = CtRbf;
save(fullfile(resultFolder, 'RBF_超分辨率结果.mat'), 'results', '-v7.3');
writetable(metricTable, fullfile(resultFolder, 'RBF_原始网格误差指标.csv'), 'Encoding', 'UTF-8');
writetable(epsilonTable, fullfile(resultFolder, 'RBF_epsilon数值表.csv'), 'Encoding', 'UTF-8');
writeSummary(fullfile(resultFolder, 'RBF_结果摘要.txt'), results);
fprintf('\n完成：已生成 %d×%d RBF 重建结果。\n结果目录：%s\n', ...
    reconstructionResolution, reconstructionResolution, resultFolder);

%% 局部函数
function isRegular = isRegularAxis(values)
% 功能：检查一维坐标轴是否为等间距规则采样。
% 输入：values 为按升序排列的 x 或 y 坐标向量。
% 输出：isRegular 为逻辑值；true 表示相邻间隔在容差内一致。
    spacings = diff(values);
    referenceSpacing = median(spacings);
    tolerance = max(abs(referenceSpacing) * 1e-9, 1e-12);
    isRegular = all(abs(spacings - referenceSpacing) <= tolerance);
end

function normalizer = createNormalizer(xValues, yValues)
% 功能：建立坐标归一化参数，使 RBF 的 epsilon 不受 mm 数值尺度影响。
% 输入：xValues、yValues 为原始物理坐标向量，单位为 mm。
% 输出：normalizer 保存每一方向的最小值和坐标跨度。
    normalizer.xMin = min(xValues);
    normalizer.yMin = min(yValues);
    normalizer.xScale = max(xValues) - min(xValues);
    normalizer.yScale = max(yValues) - min(yValues);
    if normalizer.xScale == 0 || normalizer.yScale == 0
        error('x 或 y 坐标范围为零，无法建立二维 RBF。');
    end
end

function [xNormalized, yNormalized] = normalizeAxes(xValues, yValues, normalizer)
% 功能：使用 createNormalizer 的参数将坐标映射至 [0, 1]。
% 输入：xValues、yValues 可为训练节点、查询节点或单一剖面坐标。
% 输出：xNormalized、yNormalized 为无量纲坐标，供 Gaussian 核计算。
    xNormalized = (xValues - normalizer.xMin) / normalizer.xScale;
    yNormalized = (yValues - normalizer.yMin) / normalizer.yScale;
end

function coefficients = fitSeparableRBF(xNodes, yNodes, bxGrid, byGrid, bzGrid, epsilon)
% 功能：分别拟合 Bx、By、Bz 的二维 Gaussian RBF 系数矩阵。
% 输入：xNodes、yNodes 为归一化训练节点；三个 Grid 的行对应 y、列对应 x；
%       epsilon 为归一化坐标下的 Gaussian 核形状参数。
% 输出：coefficients(:,:,1:3) 分别为 Bx、By、Bz 的 RBF 系数矩阵。
% 原理：二维核 exp[-e^2((x-xi)^2+(y-yi)^2)] 可写成 Ky*Kx 的乘积，
%       因而对每个分量求解 Ky*A*Kx' = B，无需构造 n^2×n^2 大矩阵。
    kx = gaussianKernel(xNodes, xNodes, epsilon);
    ky = gaussianKernel(yNodes, yNodes, epsilon);
    regularization = 1e-12;
    kx = kx + regularization * eye(size(kx));
    ky = ky + regularization * eye(size(ky));
    coefficients = zeros(numel(yNodes), numel(xNodes), 3);
    coefficients(:, :, 1) = ky \ bxGrid / kx.';
    coefficients(:, :, 2) = ky \ byGrid / kx.';
    coefficients(:, :, 3) = ky \ bzGrid / kx.';
end

function kernel = gaussianKernel(queryPoints, centerPoints, epsilon)
% 功能：建立一个方向上的 Gaussian 核矩阵。
% 输入：queryPoints 为待估计坐标；centerPoints 为 RBF 中心；epsilon 为形状参数。
% 输出：kernel(i,j)=exp[-epsilon^2*(queryPoints(i)-centerPoints(j))^2]。
    distance = queryPoints(:) - centerPoints(:).';
    kernel = exp(-(epsilon * distance).^2);
end

function derivativeKernel = gaussianKernelDerivative(queryPoints, centerPoints, epsilon)
% 功能：计算 Gaussian 核对查询坐标的一阶导数。
% 输入：与 gaussianKernel 相同，均为归一化坐标。
% 输出：derivativeKernel(i,j)=d phi(queryPoints(i),centerPoints(j))/d queryPoints。
% 说明：该导数仍针对归一化坐标，后续由 calculateCt2D 换算为对 mm 的导数。
    distance = queryPoints(:) - centerPoints(:).';
    kernel = exp(-(epsilon * distance).^2);
    derivativeKernel = -2 * epsilon^2 * distance .* kernel;
end

function [field, dFieldDxNormalized, dFieldDyNormalized] = evaluateSeparableRBF( ...
        xNodes, yNodes, xQuery, yQuery, coefficients, epsilon)
% 功能：在规则 x-y 查询网格上同时计算三个磁场分量与一阶导数。
% 输入：xNodes、yNodes 是训练中心；xQuery、yQuery 是查询坐标；
%       coefficients 是 fitSeparableRBF 的结果。
% 输出：field(:,:,1:3) 为 Bx、By、Bz；dFieldDxNormalized 和
%       dFieldDyNormalized 是对归一化 x/y 坐标的导数。
    kx = gaussianKernel(xQuery, xNodes, epsilon);
    ky = gaussianKernel(yQuery, yNodes, epsilon);
    dkx = gaussianKernelDerivative(xQuery, xNodes, epsilon);
    dky = gaussianKernelDerivative(yQuery, yNodes, epsilon);
    field = zeros(numel(yQuery), numel(xQuery), 3);
    dFieldDxNormalized = zeros(numel(yQuery), numel(xQuery), 3);
    dFieldDyNormalized = zeros(numel(yQuery), numel(xQuery), 3);
    for component = 1:3
        coefficient = coefficients(:, :, component);
        field(:, :, component) = ky * coefficient * kx.';
        dFieldDxNormalized(:, :, component) = ky * coefficient * dkx.';
        dFieldDyNormalized(:, :, component) = dky * coefficient * kx.';
    end
end

function field = evaluateFieldOnGrid(xNodes, yNodes, xQuery, yQuery, coefficients, epsilon)
% 功能：仅计算高分辨率网格上的 Bx、By、Bz，不计算导数以节约内存。
% 输入：与 evaluateSeparableRBF 相同。
% 输出：field 的尺寸为 length(yQuery)×length(xQuery)×3。
% 用途：由 field 立即计算 |B|，避免 Ct_2D 的导数计算延迟模值成像。
    kx = gaussianKernel(xQuery, xNodes, epsilon);
    ky = gaussianKernel(yQuery, yNodes, epsilon);
    field = zeros(numel(yQuery), numel(xQuery), 3);
    for component = 1:3
        field(:, :, component) = ky * coefficients(:, :, component) * kx.';
    end
end

function [dFieldDxNormalized, dFieldDyNormalized] = evaluateDerivativesOnGrid( ...
        xNodes, yNodes, xQuery, yQuery, coefficients, epsilon)
% 功能：仅计算 RBF 的归一化 x/y 导数，避免重复计算三分量场值。
% 输入：与 evaluateSeparableRBF 相同。
% 输出：两个三维数组，第三维依次对应 Bx、By、Bz 的导数。
% 用途：将输出交给 calculateCt2D 计算二维平面磁梯度平方和。
    kx = gaussianKernel(xQuery, xNodes, epsilon);
    ky = gaussianKernel(yQuery, yNodes, epsilon);
    dkx = gaussianKernelDerivative(xQuery, xNodes, epsilon);
    dky = gaussianKernelDerivative(yQuery, yNodes, epsilon);
    dFieldDxNormalized = zeros(numel(yQuery), numel(xQuery), 3);
    dFieldDyNormalized = zeros(numel(yQuery), numel(xQuery), 3);
    for component = 1:3
        coefficient = coefficients(:, :, component);
        dFieldDxNormalized(:, :, component) = ky * coefficient * dkx.';
        dFieldDyNormalized(:, :, component) = dky * coefficient * kx.';
    end
end

function [bx, by, bz, bmag, ct2D] = calculateDerivedFields(field, dFieldDxNormalized, dFieldDyNormalized, normalizer)
% 功能：由三个重建分量及其导数计算派生物理量 |B| 与 Ct_2D。
% 输入：field 的第三维为 Bx、By、Bz；导数针对归一化坐标；
%       normalizer 提供归一化到 mm 坐标的尺度换算。
% 输出：bx、by、bz 为各分量；bmag 为磁场模值；ct2D 为平面内梯度平方和。
    bx = field(:, :, 1);
    by = field(:, :, 2);
    bz = field(:, :, 3);
    bmag = sqrt(bx.^2 + by.^2 + bz.^2);
    ct2D = calculateCt2D(dFieldDxNormalized, dFieldDyNormalized, normalizer);
end

function ct2D = calculateCt2D(dFieldDxNormalized, dFieldDyNormalized, normalizer)
% 功能：计算单一 z 平面可观测的二维磁梯度平方和 Ct_2D。
% 输入：归一化坐标下的 dB/dx、dB/dy 与坐标尺度 normalizer。
% 输出：ct2D=(dBx/dx)^2+(dBx/dy)^2+(dBy/dx)^2+(dBy/dy)^2+
%       (dBz/dx)^2+(dBz/dy)^2，单位为 T^2/mm^2。
% 注意：没有 z 方向导数，因此该量不是完整三维 Ct。
    dFieldDx = dFieldDxNormalized / normalizer.xScale;
    dFieldDy = dFieldDyNormalized / normalizer.yScale;
    ct2D = dFieldDx(:, :, 1).^2 + dFieldDy(:, :, 1).^2 + ...
        dFieldDx(:, :, 2).^2 + dFieldDy(:, :, 2).^2 + ...
        dFieldDx(:, :, 3).^2 + dFieldDy(:, :, 3).^2;
end

function metricTable = buildMetricTable(bxTrue, byTrue, bzTrue, bmagTrue, ctTrue, bxRbf, byRbf, bzRbf, bmagRbf, ctRbf)
% 功能：汇总原始节点上 Bx、By、Bz、|B|、Ct_2D 的误差指标。
% 输入：True 为原始网格参考值；Rbf 为在相同坐标上重新估计的 RBF 值。
% 输出：metricTable 含每个物理量的 RMSE 和 MAE，供命令窗口、CSV 和摘要使用。
    metrics = [calculateMetrics(bxTrue, bxRbf); calculateMetrics(byTrue, byRbf); ...
        calculateMetrics(bzTrue, bzRbf); calculateMetrics(bmagTrue, bmagRbf); ...
        calculateMetrics(ctTrue, ctRbf)];
    metricTable = table(["Bx"; "By"; "Bz"; "|B|"; "Ct_2D"], ...
        [metrics.RMSE]', [metrics.MAE]', 'VariableNames', {'Variable', 'RMSE', 'MAE'});
end

function metric = calculateMetrics(trueValue, predictedValue)
% 功能：计算同尺寸参考值与预测值之间的 RMSE、MAE。
% 输入：trueValue 为参考数据；predictedValue 为 RBF 数据。
% 输出：metric.RMSE、metric.MAE；自动忽略 NaN/Inf 节点。
    valid = isfinite(trueValue) & isfinite(predictedValue);
    residual = predictedValue(valid) - trueValue(valid);
    metric.RMSE = sqrt(mean(residual.^2));
    metric.MAE = mean(abs(residual));
end

function limits = finiteLimits(values)
% 功能：为原始图和重建图建立相同的有效 colorbar 范围。
% 输入：同类物理量拼接后的数组，例如 [BmagOriginal(:); BmagRbf(:)]。
% 输出：limits=[最小值,最大值]；常量场时自动扩展范围避免 caxis 报错。
    values = values(isfinite(values));
    limits = [min(values), max(values)];
    if limits(1) == limits(2)
        delta = max(abs(limits(1)) * 0.01, 1);
        limits = limits + [-delta, delta];
    end
end

function assertFiniteImage(imageData, fieldName)
% 功能：在调用 imagesc 前检查图像数组是否完整有效。
% 输入：imageData 为待绘制二维数组；fieldName 为错误信息中显示的物理量名称。
% 输出：无；若发现空数组、NaN 或 Inf，则停止并报告问题位置。
    if isempty(imageData) || any(~isfinite(imageData), 'all')
        error('%s 含有 NaN 或 Inf，无法绘图。请检查输入数据或 RBF 参数。', fieldName);
    end
end

function plotImageTile(layout, xValues, yValues, imageData, titleText, colorLabel, limits)
% 功能：在 tiledlayout 的下一个子图中绘制二维空间成像及 colorbar。
% 输入：xValues、yValues 是物理坐标；imageData 的行对应 y、列对应 x；
%       titleText、colorLabel 为文字；limits 强制原始/重建图使用相同色轴。
% 输出：无，直接写入当前图窗的下一个 tile。
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

function plotProfileTile(layout, originalAxis, originalValue, highAxis, highValue, xLabelText, yLabelText, titleText, trainingGridText, resolution)
% 功能：在 tiledlayout 中比较同一物理坐标线上的原始与 RBF 剖面。
% 输入：originalAxis/Value 是原始网格剖面；highAxis/Value 是高分辨率 RBF 剖面；
%       trainingGridText 和 resolution 用于图例标注。
% 输出：无，直接生成一幅剖面对比子图。
    nexttile(layout);
    plot(originalAxis, originalValue, 'k-', 'LineWidth', 1.5); hold on;
    plot(highAxis, highValue, 'r-', 'LineWidth', 1.2); grid on;
    xlabel(xLabelText);
    ylabel(yLabelText);
    title(titleText);
    legend(['原始 ' trainingGridText], sprintf('RBF %d×%d', resolution, resolution), 'Location', 'best');
end

function saveFigure(fig, folder, fileStem, shouldSave, resolution)
% 功能：刷新 MATLAB 图窗，并按指定 DPI 选择性导出 PNG 文件。
% 输入：fig 为图窗句柄；folder/fileStem 决定保存位置和文件名；
%       shouldSave 控制是否导出；resolution 为 PNG 分辨率（DPI）。
% 输出：无，图窗始终保留；shouldSave=true 时额外写入图片文件。
    drawnow;
    if shouldSave
        exportgraphics(fig, fullfile(folder, [fileStem '.png']), 'Resolution', resolution);
    end
end

function writeSummary(filePath, results)
% 功能：将输入文件、训练/重建分辨率、epsilon 和误差指标写成文本摘要。
% 输入：filePath 为输出 .txt 路径；results 为主程序构建的结果结构体。
% 输出：无；无法创建文件时给出 warning，不中断已完成的重建流程。
    fileId = fopen(filePath, 'w');
    if fileId < 0
        warning('无法写入摘要：%s', filePath);
        return;
    end
    cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
    fprintf(fileId, 'Gaussian RBF 超分辨率重建摘要\n');
    fprintf(fileId, '输入文件：%s\n', results.inputFile);
    fprintf(fileId, '训练网格：%d×%d；重建网格：%d×%d\n', ...
        results.trainingResolution(1), results.trainingResolution(2), ...
        results.reconstructionResolution(1), results.reconstructionResolution(2));
    fprintf(fileId, 'z：%.12g mm；最优 epsilon：%.12g\n\n', ...
        results.zValue_mm, results.bestEpsilon);
    for index = 1:height(results.metricTableOnOriginalGrid)
        fprintf(fileId, '%s: RMSE=%.12g, MAE=%.12g\n', ...
            results.metricTableOnOriginalGrid.Variable(index), ...
            results.metricTableOnOriginalGrid.RMSE(index), ...
            results.metricTableOnOriginalGrid.MAE(index));
    end
end
