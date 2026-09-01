%% RBF_10_10_to_1000：10×10 或 100×100 磁场数据的 Gaussian RBF 超分辨率重建
% 手动指定 COMSOL 文本源文件，可使用全部 10×10 或 100×100 个点训练，
% 并重建为 100×100、500×500、1000×1000 或 2000×2000 网格。|B| 由 Bx、By、Bz 计算；
% Ct_2D 只使用 x/y 导数。
% 输入为单一 z 平面时，Ct_2D 不代表完整三维磁梯度张量 Ct。
%
% 脚本执行顺序：
%   1) 公共参数、原始数据读取和训练网格建立；
%   2) 模块一：Gaussian RBF 重建；
%   3) 模块二：Separable Global Ordinary Kriging 重建；
%   4) 模块三：Smoothing Thin Plate Spline 重建；
%   5) 读取 Ground Truth，并将其重采样到实际重建网格后评价三个方法；
%   6) 统一成像、中心剖面、峰值比较以及 MAT/CSV/TXT 保存。

%% ==================== 公共部分：参数与文件路径 ====================
clear;
clc;
close all;

reconstructionResolution =100; % 可改为 100、500、1000 或 2000，需相应内存。
allowedResolutions = [50,100, 500, 1000, 2000];
epsilonMultipliers = [0.50, 0.75, 1.00, 1.50, 2.00];
saveFigures = true;
figureResolution = 300;

% 手动修改此处的源文件路径：可填写 10×10 或 100×100 的 COMSOL .txt 文件。
inputFile = 'D:\YAN\大论文\x1\数据\10-10.txt';

if ~ismember(reconstructionResolution, allowedResolutions)
    error('reconstructionResolution 只能设置为 100、500、1000 或 2000。');
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

%% 公共部分 1：读取并验证原始磁场数据
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

%% 公共部分 2：建立原始训练网格并计算基准量
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

%% ==================== 模块一：Gaussian RBF 重建 ====================
% RBF 输入：BxOriginal、ByOriginal、BzOriginal。
% RBF 输出：BxRbf、ByRbf、BzRbf、BmagRbf、CtRbf。
% RBF-1：通过独立验证节点选择 epsilon，并使用全部数据训练
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

% RBF-2：计算原始训练节点上的 RBF 重建指标
[fieldReference, dFieldDxReference, dFieldDyReference] = evaluateSeparableRBF( ...
    xNormalized, yNormalized, xNormalized, yNormalized, coefficients, bestEpsilon);
[BxReference, ByReference, BzReference, BmagReference, CtReference] = ...
    calculateDerivedFields(fieldReference, dFieldDxReference, dFieldDyReference, normalizer);
metricTable = buildMetricTable(BxOriginal, ByOriginal, BzOriginal, BmagOriginal, CtOriginal, ...
    BxReference, ByReference, BzReference, BmagReference, CtReference);
fprintf('\n[3] 原始 %s 坐标上的重建指标\n', trainingGridText);
disp(metricTable);

% RBF-3：在目标高分辨率网格上计算三分量场
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

% RBF-4：由 x/y 导数计算 Ct_2D
% 以下计算六个 x/y 导数并构造 Ct_2D。
fprintf('正在计算 Ct_2D 所需的 x/y 导数...\n');
[dFieldDxHigh, dFieldDyHigh] = evaluateDerivativesOnGrid( ...
    xNormalized, yNormalized, xHighNormalized, yHighNormalized, coefficients, bestEpsilon);
CtRbf = calculateCt2D(dFieldDxHigh, dFieldDyHigh, normalizer);
clear dFieldDxHigh dFieldDyHigh
assertFiniteImage(CtRbf, 'RBF 重建 Ct_2D');

% RBF-5：记录公共中心剖面位置
% 不单独生成 RBF Figure 1、2；原始数据和三种方法统一在后续 Figure 1、2 对比。
[~, originalCenterRow] = min(abs(yValues));
[~, originalCenterColumn] = min(abs(xValues));

% RBF-6：保存 RBF 独立结果
% 同时导出统一点表 TXT，列顺序固定为 x、y、z、Bx、By、Bz。
% Kriging_PhysicsConstraint_Optimization.m 可直接读取其中的 Kriging TXT。
rbfPointTableFile = fullfile(resultFolder, 'RBF_重建数据.txt');
writeReconstructionPointTable(rbfPointTableFile, 'Gaussian RBF', ...
    xHigh, yHigh, zValues(1), BxRbf, ByRbf, BzRbf);
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
results.pointTableFile = rbfPointTableFile;
save(fullfile(resultFolder, 'RBF_超分辨率结果.mat'), 'results', '-v7.3');
writetable(metricTable, fullfile(resultFolder, 'RBF_原始网格误差指标.csv'), 'Encoding', 'UTF-8');
writetable(epsilonTable, fullfile(resultFolder, 'RBF_epsilon数值表.csv'), 'Encoding', 'UTF-8');
writeSummary(fullfile(resultFolder, 'RBF_结果摘要.txt'), results);
fprintf('\n完成：已生成 %d×%d RBF 重建结果。\n结果目录：%s\n', ...
    reconstructionResolution, reconstructionResolution, resultFolder);
fprintf('RBF 三分量点表 TXT：%s\n', rbfPointTableFile);

%% ==================== 模块二：Ordinary Kriging 重建 ====================
% Kriging 输入：BxOriginal、ByOriginal、BzOriginal、xHigh、yHigh。
% Kriging 输出：BxKriging、ByKriging、BzKriging、BmagKriging、CtKriging。
% 本模块独立使用 Kriging 参数，不读取或覆盖 RBF 的系数和结果。

% Kriging-1：设置 Ordinary Kriging 协方差参数
fprintf('\n============================================================\n');
fprintf('新增模块：Separable Global Ordinary Kriging 三分量重建\n');
fprintf('训练网格：%s；目标网格：%d×%d\n', ...
    trainingGridText, reconstructionResolution, reconstructionResolution);
fprintf('============================================================\n');

% 此实现对规则网格使用可分离 Gaussian 协方差的全局 Ordinary Kriging。
% 它一次性使用全部原始采样点，避免旧版“四个局部模型再双线性融合”改变峰形。
% 通过 x/y 两个小型特征分解求解，不会构造 10000×10000 的全局协方差矩阵。
krigingImplementation = 'separable-global-ordinary';
krigingVariogramModel = 'gaussian';
krigingSill = 1.0;                % Gaussian 协方差的信号方差。
krigingNugget = 1e-10;            % 相对块金/数值稳定项；越接近零越接近严格通过采样点。
krigingRangeMultiplier = 0.75;    % 推荐 0.5~1.5；过大将使 Gaussian 协方差矩阵病态。
krigingRange = krigingRangeMultiplier * ...
    max(median(diff(xValues)), median(diff(yValues))); % 单位：mm。

krigingConfig = createKrigingConfig(krigingImplementation, krigingVariogramModel, ...
    krigingNugget, krigingSill, krigingRange, numel(xValues), numel(yValues));
fprintf('实现：%s；协方差：%s；range：%.12g mm；nugget：%.3g\n', ...
    krigingConfig.implementation, krigingConfig.variogramModel, ...
    krigingConfig.range, krigingConfig.nugget);

% 三个磁场分量使用相同空间权重，但以各自观测值独立预测，绝不对 |B| 或 Ct_2D
% 直接进行 Kriging。输出尺寸与原 RBF 的 BxRbf/ByRbf/BzRbf 完全相同。
fprintf('正在执行 %d×%d Ordinary Kriging，请耐心等待...\n', ...
    reconstructionResolution, reconstructionResolution);
[BxKriging, ByKriging, BzKriging] = separableGlobalOrdinaryKrigingGrid( ...
    xValues, yValues, BxOriginal, ByOriginal, BzOriginal, ...
    xHigh, yHigh, krigingConfig);

assertFiniteImage(BxKriging, 'Kriging 重建 Bx');
assertFiniteImage(ByKriging, 'Kriging 重建 By');
assertFiniteImage(BzKriging, 'Kriging 重建 Bz');
BmagKriging = sqrt(BxKriging.^2 + ByKriging.^2 + BzKriging.^2);
assertFiniteImage(BmagKriging, 'Kriging 重建 |B|');

% CtKriging 的定义严格沿用原 RBF 主程序：仅使用单一 z 平面可求得的
% 六个 x/y 偏导数，不人为补充任何 z 方向导数。
[dBxKrigingDx, dBxKrigingDy] = gradient(BxKriging, xHigh, yHigh);
[dByKrigingDx, dByKrigingDy] = gradient(ByKriging, xHigh, yHigh);
[dBzKrigingDx, dBzKrigingDy] = gradient(BzKriging, xHigh, yHigh);
CtKriging = dBxKrigingDx.^2 + dBxKrigingDy.^2 + ...
    dByKrigingDx.^2 + dByKrigingDy.^2 + ...
    dBzKrigingDx.^2 + dBzKrigingDy.^2;
clear dBxKrigingDx dBxKrigingDy dByKrigingDx dByKrigingDy dBzKrigingDx dBzKrigingDy
assertFiniteImage(CtKriging, 'Kriging 重建 Ct_2D');

% Kriging-2：记录共同中心剖面位置
% Figure 1、2 由三个方法统一绘制，避免每个方法重复创建图窗。
[~, krigingCenterRow] = min(abs(yHigh - yValues(originalCenterRow)));
[~, krigingCenterColumn] = min(abs(xHigh - xValues(originalCenterColumn)));

% Kriging-3：保存 Kriging 独立结果
% 当前输入文件只提供 10×10 或 100×100 采样数据，不含真实 1000×1000 场。
% 因此以下表格比较各算法输出的峰值，不将 RBF 或 Kriging 伪作 Ground Truth，
% 也不输出任何虚假的高分辨率 RMSE、MAE 或 R²。
comparisonTable = buildMethodComparisonTable(BmagRbf, CtRbf, BmagKriging, CtKriging, xHigh, yHigh);
fprintf('\n[新增] RBF 与 Kriging 峰值/位置比较（非 Ground Truth 误差）\n');
disp(comparisonTable);
fprintf(['说明：当前输入未包含真实 %d×%d Ground Truth；新增模块仅进行 RBF 与 ' ...
    'Kriging 的成像、剖面和峰值位置比较，不计算伪误差。\n'], ...
    reconstructionResolution, reconstructionResolution);

krigingResults = struct();
krigingPointTableFile = fullfile(resultFolder, 'Kriging_重建数据.txt');
writeReconstructionPointTable(krigingPointTableFile, 'Separable Global Ordinary Kriging', ...
    xHigh, yHigh, zValues(1), BxKriging, ByKriging, BzKriging);
krigingResults.inputFile = inputFile;
krigingResults.trainingResolution = trainingGridSize;
krigingResults.reconstructionResolution = [reconstructionResolution, reconstructionResolution];
krigingResults.xHigh_mm = xHigh;
krigingResults.yHigh_mm = yHigh;
krigingResults.implementation = krigingConfig.implementation;
krigingResults.variogramModel = krigingConfig.variogramModel;
krigingResults.nugget = krigingConfig.nugget;
krigingResults.sill = krigingConfig.sill;
krigingResults.rangeMultiplier = krigingRangeMultiplier;
krigingResults.range_mm = krigingConfig.range;
krigingResults.BxKriging = BxKriging;
krigingResults.ByKriging = ByKriging;
krigingResults.BzKriging = BzKriging;
krigingResults.BmagKriging = BmagKriging;
krigingResults.CtKriging = CtKriging;
krigingResults.zValue_mm = zValues(1);
krigingResults.pointTableFile = krigingPointTableFile;
krigingResults.comparisonTable = comparisonTable;
save(fullfile(resultFolder, 'Kriging_超分辨率结果.mat'), 'krigingResults', '-v7.3');
writetable(comparisonTable, fullfile(resultFolder, 'RBF与Kriging峰值位置比较.csv'), 'Encoding', 'UTF-8');
writeKrigingSummary(fullfile(resultFolder, 'Kriging_结果摘要.txt'), krigingResults);
fprintf('Kriging 结果已保存至：%s\n', resultFolder);
fprintf('Kriging 三分量点表 TXT：%s\n', krigingPointTableFile);

%% ==================== 模块三：Smoothing Thin Plate Spline 重建 ====================
% TPS 输入：BxOriginal、ByOriginal、BzOriginal、xHigh、yHigh。
% TPS 输出：BxTps、ByTps、BzTps、BmagTps、CtTps。
% TPS 与 RBF、Kriging 相互独立，直接对原始三个磁场分量拟合。

% TPS-1：设置平滑参数
% 取值范围为 0 到 1：越接近 1 越接近严格插值；越小则平滑越强。
% 可根据 Ground Truth 误差评价结果手动调整；0.99 是保留局部异常的初始设置。
tpsSmoothingParameter = 0.99;
% tpaps 是全局薄板样条。100×100 的 10000 个节点会导致全局方程过大并长时间卡顿。
% 因此每个方向最多均匀选取该数量的控制节点；10×10 输入会自动保留全部节点。
% 可按机器性能调大到 30；不建议直接设为 100。
tpsMaxControlPointsPerAxis = 50;

% TPS-2：拟合三个分量并计算 |B|、Ct_2D
fprintf('\n============================================================\n');
fprintf('新增模块：Smoothing Thin Plate Spline 三分量重建\n');
fprintf('训练网格：%s；目标网格：%d×%d；TPS 平滑参数=%.12g；每方向最多 %d 个控制节点\n', ...
    trainingGridText, reconstructionResolution, reconstructionResolution, tpsSmoothingParameter, ...
    tpsMaxControlPointsPerAxis);
fprintf('============================================================\n');

fprintf('正在执行 %d×%d Smoothing Thin Plate Spline，请耐心等待...\n', ...
    reconstructionResolution, reconstructionResolution);
[BxTps, ByTps, BzTps, tpsModel] = smoothingThinPlateSplineGrid( ...
    xValues, yValues, BxOriginal, ByOriginal, BzOriginal, ...
    xHigh, yHigh, tpsSmoothingParameter, tpsMaxControlPointsPerAxis);
assertFiniteImage(BxTps, 'TPS 重建 Bx');
assertFiniteImage(ByTps, 'TPS 重建 By');
assertFiniteImage(BzTps, 'TPS 重建 Bz');
BmagTps = sqrt(BxTps.^2 + ByTps.^2 + BzTps.^2);
Z_TPS = BmagTps; % 供方法比较使用的 TPS 磁场模值变量，不覆盖 RBF/Kriging 结果。
assertFiniteImage(BmagTps, 'TPS 重建 |B|');
[dBxTpsDx, dBxTpsDy] = gradient(BxTps, xHigh, yHigh);
[dByTpsDx, dByTpsDy] = gradient(ByTps, xHigh, yHigh);
[dBzTpsDx, dBzTpsDy] = gradient(BzTps, xHigh, yHigh);
CtTps = dBxTpsDx.^2 + dBxTpsDy.^2 + dByTpsDx.^2 + dByTpsDy.^2 + ...
    dBzTpsDx.^2 + dBzTpsDy.^2;
clear dBxTpsDx dBxTpsDy dByTpsDx dByTpsDy dBzTpsDx dBzTpsDy
assertFiniteImage(CtTps, 'TPS 重建 Ct_2D');

%% ==================== 公共部分：读取 Ground Truth ====================
% Ground Truth 只读取 COMSOL 的原始 1000×1000 数据，不由任何插值方法生成。
% Ground Truth 直接来自 COMSOL 的 1000×1000 导出文件，不使用任何重建方法生成。
groundTruthFile = 'D:\YAN\大论文\x1\数据\1000-1000.txt';
if ~isfile(groundTruthFile)
    error('找不到 Ground Truth 文件：%s', groundTruthFile);
end
fprintf('\n正在直接读取 Ground Truth：%s\n', groundTruthFile);
groundTruth = readComsolGroundTruth1000(groundTruthFile);
groundTruth.Bmag = sqrt(groundTruth.Bx.^2 + groundTruth.By.^2 + groundTruth.Bz.^2);
[gtBxDx, gtBxDy] = gradient(groundTruth.Bx, groundTruth.x, groundTruth.y);
[gtByDx, gtByDy] = gradient(groundTruth.By, groundTruth.x, groundTruth.y);
[gtBzDx, gtBzDy] = gradient(groundTruth.Bz, groundTruth.x, groundTruth.y);
groundTruth.Ct2D = gtBxDx.^2 + gtBxDy.^2 + gtByDx.^2 + gtByDy.^2 + ...
    gtBzDx.^2 + gtBzDy.^2;
clear gtBxDx gtBxDy gtByDx gtByDy gtBzDx gtBzDy
assertFiniteImage(groundTruth.Bmag, 'Ground Truth |B|');
assertFiniteImage(groundTruth.Ct2D, 'Ground Truth Ct_2D');

% Figure 1 与后续误差评价统一使用重采样到当前重建网格的 Ground Truth。
% 只改变用于显示和逐点比较的真值副本，不修改 COMSOL 原始 1000×1000 数据。
groundTruthOnReconstructionGrid = resampleGroundTruthToReconstructionGrid( ...
    groundTruth, xHigh, yHigh);
groundTruthDisplayText = sprintf('重采样至 %d×%d', numel(yHigh), numel(xHigh));

%% ==================== 公共部分：三方法统一成像、剖面与峰值比较 ====================
% 这一部分只负责展示和比较，不再进行新的 RBF、Kriging 或 TPS 插值。
% 三种方法的输出均保持独立，分别使用 BmagRbf/BmagKriging/BmagTps 和
% CtRbf/CtKriging/CtTps 进入同一组图像、剖面和峰值表。

% Figure 1：原始输入、重采样 Ground Truth 与三种独立重建在同一网格和色轴下对比。
bmagThreeMethodLimits = finiteLimits([BmagOriginal(:); groundTruthOnReconstructionGrid.Bmag(:); ...
    BmagRbf(:); BmagKriging(:); BmagTps(:)]);
% TPS 分量场平滑且可微，五列 Ct_2D 使用包含 TPS 的同一色轴范围直接比较。
ctThreeMethodLimits = finiteLimits([CtOriginal(:); groundTruthOnReconstructionGrid.Ct2D(:); ...
    CtRbf(:); CtKriging(:); CtTps(:)]);
figThreeMethod = figure(1);
set(figThreeMethod, 'Color', 'w', 'Position', [30, 70, 1980, 850], ...
    'Name', [trainingGridText ' 原始、Ground Truth、RBF、Kriging 与 Smoothing TPS 成像对比']);
clf(figThreeMethod);
threeMethodLayout = tiledlayout(figThreeMethod, 2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
colormap(figThreeMethod, parula(256));
plotImageTile(threeMethodLayout, xValues, yValues, BmagOriginal, ['原始输入 |B|（' trainingGridText '）'], '|B| (T)', bmagThreeMethodLimits);
plotImageTile(threeMethodLayout, xHigh, yHigh, groundTruthOnReconstructionGrid.Bmag, ...
    ['Ground Truth |B|（' groundTruthDisplayText '）'], '|B| (T)', bmagThreeMethodLimits);
plotImageTile(threeMethodLayout, xHigh, yHigh, BmagRbf, 'RBF Reconstruction |B|', '|B| (T)', bmagThreeMethodLimits);
plotImageTile(threeMethodLayout, xHigh, yHigh, BmagKriging, 'Global Ordinary Kriging Reconstruction |B|', '|B| (T)', bmagThreeMethodLimits);
plotImageTile(threeMethodLayout, xHigh, yHigh, BmagTps, 'Smoothing TPS Reconstruction |B|', '|B| (T)', bmagThreeMethodLimits);
plotImageTile(threeMethodLayout, xValues, yValues, CtOriginal, ['原始输入 Ct_2D（' trainingGridText '）'], 'Ct_2D (T^2/mm^2)', ctThreeMethodLimits);
plotImageTile(threeMethodLayout, xHigh, yHigh, groundTruthOnReconstructionGrid.Ct2D, ...
    ['Ground Truth Ct_2D（' groundTruthDisplayText '）'], 'Ct_2D (T^2/mm^2)', ctThreeMethodLimits);
plotImageTile(threeMethodLayout, xHigh, yHigh, CtRbf, 'RBF Reconstruction Ct_2D', 'Ct_2D (T^2/mm^2)', ctThreeMethodLimits);
plotImageTile(threeMethodLayout, xHigh, yHigh, CtKriging, 'Global Ordinary Kriging Reconstruction Ct_2D', 'Ct_2D (T^2/mm^2)', ctThreeMethodLimits);
plotImageTile(threeMethodLayout, xHigh, yHigh, CtTps, 'Smoothing TPS Reconstruction Ct_2D', 'Ct_2D (T^2/mm^2)', ctThreeMethodLimits);
title(threeMethodLayout, sprintf('%s 原始输入、Ground Truth 与三种独立重建；TPS 平滑参数=%.12g', ...
    trainingGridText, tpsSmoothingParameter));
saveFigure(figThreeMethod, resultFolder, '01_RBF_Kriging_TPS成像对比', saveFigures, figureResolution);

% Figure 2：黑色圆点线是原始采样曲线，其余三条曲线来自同一高分辨率目标网格。
figTpsProfile = figure(2);
set(figTpsProfile, 'Color', 'w', 'Position', [90, 105, 1330, 820], ...
    'Name', [trainingGridText ' 原始、RBF、Kriging 与 TPS 中心剖面对比']);
clf(figTpsProfile);
tpsProfileLayout = tiledlayout(figTpsProfile, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
plotFourMethodProfileTile(tpsProfileLayout, xValues, BmagOriginal(originalCenterRow, :), xHigh, BmagRbf(krigingCenterRow, :), BmagKriging(krigingCenterRow, :), BmagTps(krigingCenterRow, :), 'x (mm)', '|B| (T)', sprintf('|B| 的 x 方向剖面，y = %.12g mm', yValues(originalCenterRow)), trainingGridText, []);
plotFourMethodProfileTile(tpsProfileLayout, xValues, CtOriginal(originalCenterRow, :), xHigh, CtRbf(krigingCenterRow, :), CtKriging(krigingCenterRow, :), CtTps(krigingCenterRow, :), 'x (mm)', 'Ct_2D (T^2/mm^2)', sprintf('Ct_2D 的 x 方向剖面，y = %.12g mm', yValues(originalCenterRow)), trainingGridText, ctThreeMethodLimits);
plotFourMethodProfileTile(tpsProfileLayout, yValues, BmagOriginal(:, originalCenterColumn), yHigh, BmagRbf(:, krigingCenterColumn), BmagKriging(:, krigingCenterColumn), BmagTps(:, krigingCenterColumn), 'y (mm)', '|B| (T)', sprintf('|B| 的 y 方向剖面，x = %.12g mm', xValues(originalCenterColumn)), trainingGridText, []);
plotFourMethodProfileTile(tpsProfileLayout, yValues, CtOriginal(:, originalCenterColumn), yHigh, CtRbf(:, krigingCenterColumn), CtKriging(:, krigingCenterColumn), CtTps(:, krigingCenterColumn), 'y (mm)', 'Ct_2D (T^2/mm^2)', sprintf('Ct_2D 的 y 方向剖面，x = %.12g mm', xValues(originalCenterColumn)), trainingGridText, ctThreeMethodLimits);
saveFigure(figTpsProfile, resultFolder, '02_RBF_Kriging_TPS中心剖面对比', saveFigures, figureResolution);

tpsComparisonTable = buildThreeMethodComparisonTable(BmagRbf, CtRbf, BmagKriging, CtKriging, BmagTps, CtTps, xHigh, yHigh);
fprintf('\n[新增] RBF、Kriging 与 Smoothing TPS 峰值/位置比较（非 Ground Truth 误差）\n');
disp(tpsComparisonTable);
tpsResults = struct();
tpsResults.inputFile = inputFile;
tpsResults.trainingResolution = trainingGridSize;
tpsResults.reconstructionResolution = [reconstructionResolution, reconstructionResolution];
tpsResults.xHigh_mm = xHigh;
tpsResults.yHigh_mm = yHigh;
tpsResults.smoothingParameter = tpsSmoothingParameter;
tpsResults.maxControlPointsPerAxis = tpsMaxControlPointsPerAxis;
tpsResults.controlGridSize = tpsModel.controlGridSize;
tpsResults.controlPointCount = tpsModel.controlPointCount;
tpsResults.BxTps = BxTps;
tpsResults.ByTps = ByTps;
tpsResults.BzTps = BzTps;
tpsResults.BmagTps = BmagTps;
tpsResults.Z_TPS = Z_TPS;
tpsResults.CtTps = CtTps;
tpsPointTableFile = fullfile(resultFolder, 'TPS_重建数据.txt');
writeReconstructionPointTable(tpsPointTableFile, 'Smoothing Thin Plate Spline', ...
    xHigh, yHigh, zValues(1), BxTps, ByTps, BzTps);
tpsResults.zValue_mm = zValues(1);
tpsResults.pointTableFile = tpsPointTableFile;
tpsResults.comparisonTable = tpsComparisonTable;
save(fullfile(resultFolder, 'TPS_超分辨率结果.mat'), 'tpsResults', '-v7.3');
writetable(tpsComparisonTable, fullfile(resultFolder, 'RBF_Kriging_TPS峰值位置比较.csv'), 'Encoding', 'UTF-8');
writeTpsSummary(fullfile(resultFolder, 'TPS_结果摘要.txt'), tpsResults);
fprintf('TPS 结果已保存至：%s\n', resultFolder);
fprintf('TPS 三分量点表 TXT：%s\n', tpsPointTableFile);

%% ==================== 公共部分：按重建网格进行 Ground Truth 误差评价 ====================
% RBF、Kriging 和 TPS 重建结果始终保持用户指定的 reconstructionResolution，
% 不会为了误差评价重新计算或重采样这些重建结果。仅将原始 COMSOL Ground Truth
% 从 1000×1000 重采样至 xHigh/yHigh，并在重建网格上计算逐点误差。

fprintf('\n============================================================\n');
fprintf('Ground Truth 误差评价：复用已读取的 %s\n', groundTruthFile);
fprintf('============================================================\n');

%% Ground Truth-1：复用 Figure 1 已建立的重采样 Ground Truth
evaluationGroundTruth = groundTruthOnReconstructionGrid;
groundTruthEvaluationOperation = sprintf(['已将原始 Ground Truth 从 %d×%d 重采样至重建网格 %d×%d；' ...
    'RBF、Kriging 和 TPS 重建结果未进行重采样。'], ...
    numel(groundTruth.y), numel(groundTruth.x), numel(yHigh), numel(xHigh));
fprintf('%s\n', groundTruthEvaluationOperation);

% 三种重建结果直接参与评价，不再为误差评价进行二次 RBF/Kriging/TPS 求值。
BxRbfEval = BxRbf; ByRbfEval = ByRbf; BzRbfEval = BzRbf; CtRbfEval = CtRbf;
BxKrigingEval = BxKriging; ByKrigingEval = ByKriging; BzKrigingEval = BzKriging; CtKrigingEval = CtKriging;
BxTpsEval = BxTps; ByTpsEval = ByTps; BzTpsEval = BzTps; CtTpsEval = CtTps;

%% Ground Truth-2：统一尺寸检查、误差场和误差指标计算
% 尺寸、坐标和 NaN/Inf 检查通过后，才允许在重建网格逐点计算误差。
validateGroundTruthComparison(evaluationGroundTruth, xHigh, yHigh, ...
    BxRbfEval, ByRbfEval, BzRbfEval, CtRbfEval, ...
    BxKrigingEval, ByKrigingEval, BzKrigingEval, CtKrigingEval, ...
    BxTpsEval, ByTpsEval, BzTpsEval, CtTpsEval);
BmagRbfEval = sqrt(BxRbfEval.^2 + ByRbfEval.^2 + BzRbfEval.^2);
BmagKrigingEval = sqrt(BxKrigingEval.^2 + ByKrigingEval.^2 + BzKrigingEval.^2);
BmagTpsEval = sqrt(BxTpsEval.^2 + ByTpsEval.^2 + BzTpsEval.^2);
Z_TPS_Eval = BmagTpsEval;
assertFiniteImage(BmagRbfEval, 'RBF 重建网格 |B|');
assertFiniteImage(BmagKrigingEval, 'Kriging 重建网格 |B|');
assertFiniteImage(BmagTpsEval, 'TPS 重建网格 |B|');

% 误差场定义：E = Z_reconstructed - Z_groundtruth。此处七图和主汇总表的
% Z 取磁场模值 |B|；同时输出 Bx、By、Bz、Ct_2D 的完整定量误差表。
errorBmagRbf = BmagRbfEval - evaluationGroundTruth.Bmag;
errorBmagKriging = BmagKrigingEval - evaluationGroundTruth.Bmag;
errorBmagTps = BmagTpsEval - evaluationGroundTruth.Bmag;
% 全部物理量误差场写入 MAT，便于后续对 Bx、By、Bz、Ct_2D 继续单独成像。
% 当前七图展示最常用的磁场模值 |B|，包含三种方法各自的误差图。
errorFieldsRbf.Bx = BxRbfEval - evaluationGroundTruth.Bx;
errorFieldsRbf.By = ByRbfEval - evaluationGroundTruth.By;
errorFieldsRbf.Bz = BzRbfEval - evaluationGroundTruth.Bz;
errorFieldsRbf.Bmag = errorBmagRbf;
errorFieldsRbf.Ct2D = CtRbfEval - evaluationGroundTruth.Ct2D;
errorFieldsKriging.Bx = BxKrigingEval - evaluationGroundTruth.Bx;
errorFieldsKriging.By = ByKrigingEval - evaluationGroundTruth.By;
errorFieldsKriging.Bz = BzKrigingEval - evaluationGroundTruth.Bz;
errorFieldsKriging.Bmag = errorBmagKriging;
errorFieldsKriging.Ct2D = CtKrigingEval - evaluationGroundTruth.Ct2D;
errorFieldsTps.Bx = BxTpsEval - evaluationGroundTruth.Bx;
errorFieldsTps.By = ByTpsEval - evaluationGroundTruth.By;
errorFieldsTps.Bz = BzTpsEval - evaluationGroundTruth.Bz;
errorFieldsTps.Bmag = errorBmagTps;
errorFieldsTps.Ct2D = CtTpsEval - evaluationGroundTruth.Ct2D;
metricsBmagRbf = calculateGroundTruthMetrics(evaluationGroundTruth.Bmag, BmagRbfEval);
metricsBmagKriging = calculateGroundTruthMetrics(evaluationGroundTruth.Bmag, BmagKrigingEval);
metricsBmagTps = calculateGroundTruthMetrics(evaluationGroundTruth.Bmag, BmagTpsEval);
errorSummaryTable = buildGroundTruthSummaryTable(evaluationGroundTruth.Bmag, BmagRbfEval, BmagKrigingEval, BmagTpsEval, ...
    xHigh, yHigh);
detailedErrorTable = buildDetailedGroundTruthMetrics(evaluationGroundTruth, ...
    BxRbfEval, ByRbfEval, BzRbfEval, BmagRbfEval, CtRbfEval, ...
    BxKrigingEval, ByKrigingEval, BzKrigingEval, BmagKrigingEval, CtKrigingEval, ...
    BxTpsEval, ByTpsEval, BzTpsEval, BmagTpsEval, CtTpsEval);
peakComparisonTable = buildGroundTruthPeakTable(evaluationGroundTruth.Bmag, BmagRbfEval, BmagKrigingEval, BmagTpsEval, ...
    xHigh, yHigh);

fprintf('\n[Ground Truth] |B| 的 %d×%d 重建网格误差汇总\n', reconstructionResolution, reconstructionResolution);
disp(errorSummaryTable);
fprintf('[Ground Truth] 三分量、|B| 与 Ct_2D 的完整误差表\n');
disp(detailedErrorTable);
fprintf('[Ground Truth] |B| 峰值与物理位置（单位：T，mm）\n');
disp(peakComparisonTable);

%% Ground Truth-3：Ground Truth、三种重建与误差场图（Z = |B|）
% 所有图均在 xHigh/yHigh 重建坐标上绘制；Ground Truth 已重采样到该坐标。
bmagGroundTruthLimits = finiteLimits([evaluationGroundTruth.Bmag(:); BmagRbfEval(:); BmagKrigingEval(:); BmagTpsEval(:)]);
errorLimit = max(abs([errorBmagRbf(:); errorBmagKriging(:); errorBmagTps(:)]));
if errorLimit == 0
    errorLimit = 1;
end
figGroundTruthError = figure(3);
set(figGroundTruthError, 'Color', 'w', 'Position', [35, 50, 1780, 880], ...
    'Name', [trainingGridText ' Ground Truth 误差评价']);
clf(figGroundTruthError);
groundTruthLayout = tiledlayout(figGroundTruthError, 2, 4, ...
    'TileSpacing', 'compact', 'Padding', 'compact');
colormap(figGroundTruthError, parula(256));
plotImageTile(groundTruthLayout, xHigh, yHigh, evaluationGroundTruth.Bmag, ...
    'Ground Truth |B|（重采样至重建网格）', '|B| (T)', bmagGroundTruthLimits);
plotImageTile(groundTruthLayout, xHigh, yHigh, BmagRbfEval, ...
    'RBF Reconstruction |B|', '|B| (T)', bmagGroundTruthLimits);
plotSignedErrorTile(groundTruthLayout, xHigh, yHigh, errorBmagRbf, ...
    'RBF Error：RBF - Ground Truth', 'Error |B| (T)', errorLimit);
plotImageTile(groundTruthLayout, xHigh, yHigh, BmagKrigingEval, ...
    'Kriging Reconstruction |B|', '|B| (T)', bmagGroundTruthLimits);
plotSignedErrorTile(groundTruthLayout, xHigh, yHigh, errorBmagKriging, ...
    'Kriging Error：Kriging - Ground Truth', 'Error |B| (T)', errorLimit);
plotImageTile(groundTruthLayout, xHigh, yHigh, BmagTpsEval, ...
    'Smoothing TPS Reconstruction |B|', '|B| (T)', bmagGroundTruthLimits);
plotSignedErrorTile(groundTruthLayout, xHigh, yHigh, errorBmagTps, ...
    'TPS Error：TPS - Ground Truth', 'Error |B| (T)', errorLimit);
title(groundTruthLayout, sprintf('%s 输入；%d×%d 重建网格上的三种方法逐点误差评价', ...
    trainingGridText, reconstructionResolution, reconstructionResolution));
saveFigure(figGroundTruthError, resultFolder, '03_GroundTruth_RBF_Kriging_TPS误差评价', saveFigures, figureResolution);

% Ground Truth 与三种方法均在同一重建网格，直接观察中心剖面误差来源。
gtCenterRow = krigingCenterRow;
gtCenterColumn = krigingCenterColumn;
figGroundTruthProfile = figure(4);
set(figGroundTruthProfile, 'Color', 'w', 'Position', [80, 120, 1380, 620], ...
    'Name', [trainingGridText ' Ground Truth 四方法中心剖面对比']);
clf(figGroundTruthProfile);
groundTruthProfileLayout = tiledlayout(figGroundTruthProfile, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
plotGroundTruthProfileTile(groundTruthProfileLayout, xHigh, evaluationGroundTruth.Bmag(gtCenterRow, :), ...
    BmagRbfEval(gtCenterRow, :), BmagKrigingEval(gtCenterRow, :), BmagTpsEval(gtCenterRow, :), ...
    'x (mm)', '|B| (T)', sprintf('|B| 的 x 方向剖面，y = %.12g mm', yHigh(gtCenterRow)));
plotGroundTruthProfileTile(groundTruthProfileLayout, yHigh, evaluationGroundTruth.Bmag(:, gtCenterColumn), ...
    BmagRbfEval(:, gtCenterColumn), BmagKrigingEval(:, gtCenterColumn), BmagTpsEval(:, gtCenterColumn), ...
    'y (mm)', '|B| (T)', sprintf('|B| 的 y 方向剖面，x = %.12g mm', xHigh(gtCenterColumn)));
saveFigure(figGroundTruthProfile, resultFolder, '04_GroundTruth_RBF_Kriging_TPS中心剖面对比', saveFigures, figureResolution);

%% Ground Truth-4：保存误差评价结果
groundTruthResults = struct();
groundTruthResults.file = groundTruthFile;
groundTruthResults.evaluationOperation = groundTruthEvaluationOperation;
groundTruthResults.sourceGroundTruthX_mm = groundTruth.x;
groundTruthResults.sourceGroundTruthY_mm = groundTruth.y;
groundTruthResults.x_mm = xHigh;
groundTruthResults.y_mm = yHigh;
groundTruthResults.BmagGroundTruth = evaluationGroundTruth.Bmag;
groundTruthResults.Ct2DGroundTruth = evaluationGroundTruth.Ct2D;
groundTruthResults.BmagRbf = BmagRbfEval;
groundTruthResults.Ct2DRbf = CtRbfEval;
groundTruthResults.BmagKriging = BmagKrigingEval;
groundTruthResults.Ct2DKriging = CtKrigingEval;
groundTruthResults.BmagTps = BmagTpsEval;
groundTruthResults.Z_TPS = Z_TPS_Eval;
groundTruthResults.Ct2DTps = CtTpsEval;
groundTruthResults.errorBmagRbf = errorBmagRbf;
groundTruthResults.errorBmagKriging = errorBmagKriging;
groundTruthResults.errorBmagTps = errorBmagTps;
groundTruthResults.errorFieldsRbf = errorFieldsRbf;
groundTruthResults.errorFieldsKriging = errorFieldsKriging;
groundTruthResults.errorFieldsTps = errorFieldsTps;
groundTruthResults.errorSummaryTable = errorSummaryTable;
groundTruthResults.detailedErrorTable = detailedErrorTable;
groundTruthResults.peakComparisonTable = peakComparisonTable;
groundTruthResults.metricsBmagRbf = metricsBmagRbf;
groundTruthResults.metricsBmagKriging = metricsBmagKriging;
groundTruthResults.metricsBmagTps = metricsBmagTps;
save(fullfile(resultFolder, 'GroundTruth_误差评价结果.mat'), 'groundTruthResults', '-v7.3');
writetable(errorSummaryTable, fullfile(resultFolder, 'GroundTruth_Bmag误差汇总.csv'), 'Encoding', 'UTF-8');
writetable(detailedErrorTable, fullfile(resultFolder, 'GroundTruth_完整误差指标.csv'), 'Encoding', 'UTF-8');
writetable(peakComparisonTable, fullfile(resultFolder, 'GroundTruth_Bmag峰值比较.csv'), 'Encoding', 'UTF-8');
writeGroundTruthSummary(fullfile(resultFolder, 'GroundTruth_误差评价摘要.txt'), ...
    groundTruthFile, errorSummaryTable, detailedErrorTable, peakComparisonTable);
fprintf('Ground Truth 误差评价结果已保存至：%s\n', resultFolder);

%% ==================== 局部函数 ====================
% 以下函数只提供公共检查、三个重建模块和结果输出所需的底层功能。
% 函数本身不在脚本主体中执行，脚本主体负责组织 RBF、Kriging 和 TPS 的流程。

%% 局部函数组 1：公共坐标、数据和绘图工具
function isRegular = isRegularAxis(values)
% 功能：检查一维坐标轴是否为等间距规则采样。
% 输入：values 为按升序排列的 x 或 y 坐标向量。
% 输出：isRegular 为逻辑值；true 表示相邻间隔在容差内一致。
    spacings = diff(values);
    referenceSpacing = median(spacings);
    tolerance = max(abs(referenceSpacing) * 1e-9, 1e-12);
    isRegular = all(abs(spacings - referenceSpacing) <= tolerance);
end

function writeReconstructionPointTable(filePath, methodName, xValues, yValues, zValue, bx, by, bz)
% 功能：将一个重建方法的三分量场导出为可复用的六列 TXT 点表。
% 格式：前 9 行为说明头；第 10 行起依次为 x(mm)、y(mm)、z(mm)、Bx(T)、By(T)、Bz(T)。
% 说明：矩阵行对应 y，列对应 x；按行逐块写入，避免 1000×1000 或 2000×2000
%       重建时额外构造完整六列大矩阵而占用过多内存。
    xValues = xValues(:)';
    yValues = yValues(:)';
    expectedSize = [numel(yValues), numel(xValues)];
    if ~isequal(size(bx), expectedSize) || ~isequal(size(by), expectedSize) || ...
            ~isequal(size(bz), expectedSize)
        error('%s 的三分量尺寸必须全部为 %d×%d，无法导出 TXT。', ...
            methodName, expectedSize(1), expectedSize(2));
    end
    assertFiniteImage(bx, [methodName ' TXT Bx']);
    assertFiniteImage(by, [methodName ' TXT By']);
    assertFiniteImage(bz, [methodName ' TXT Bz']);

    fileId = fopen(filePath, 'w');
    if fileId < 0
        error('无法创建重建 TXT 文件：%s', filePath);
    end
    cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
    fprintf(fileId, '%% Magnetic field reconstruction point table\n');
    fprintf(fileId, '%% Method: %s\n', methodName);
    fprintf(fileId, '%% Grid size: %d x %d (rows=y, columns=x)\n', expectedSize(1), expectedSize(2));
    fprintf(fileId, '%% Length unit: mm\n');
    fprintf(fileId, '%% Magnetic field unit: T\n');
    fprintf(fileId, '%% z plane: %.12g mm\n', zValue);
    fprintf(fileId, '%% Data order: y-major rows, x increasing within each row\n');
    fprintf(fileId, '%% Columns use tab separators\n');
    fprintf(fileId, '%% x(mm)\ty(mm)\tz(mm)\tBx(T)\tBy(T)\tBz(T)\n');

    zRow = repmat(zValue, 1, numel(xValues));
    for rowIndex = 1:numel(yValues)
        yRow = repmat(yValues(rowIndex), 1, numel(xValues));
        rowData = [xValues; yRow; zRow; bx(rowIndex, :); by(rowIndex, :); bz(rowIndex, :)];
        fprintf(fileId, '%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n', rowData);
    end
end

function [isRegular, nominalSpacing, maxCoordinateDeviation, tolerance] = ...
    isRegularComsolGroundTruthAxis(values)
% 功能：检查 COMSOL Ground Truth 坐标轴是否为规则网格，并容忍文本导出的坐标量化误差。
% 说明：COMSOL 可能将等间距坐标仅输出到小数点后三位，使相邻步长显示为 0.198/0.199 mm；
%       因此此处比较每个坐标与同一端点定义的理想均匀轴的偏差，而不直接比较相邻步长。
% 输入：values 为按升序排列的一维坐标向量，单位为 mm。
% 输出：isRegular 表示是否为可用于逐点比较的规则网格；其余输出用于报错诊断。
    values = values(:)';
    nominalSpacing = (values(end) - values(1)) / (numel(values) - 1);
    idealValues = values(1) + (0:numel(values)-1) * nominalSpacing;

    % 0.00051 mm 对应坐标按 0.001 mm 四舍五入时的最大误差，并留出极小浮点余量。
    tolerance = max(5.1e-4, abs(nominalSpacing) * 1e-10);
    maxCoordinateDeviation = max(abs(values - idealValues));
    isRegular = all(isfinite(values)) && all(diff(values) > 0) && ...
        nominalSpacing > 0 && maxCoordinateDeviation <= tolerance;
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

%% 局部函数组 2：Gaussian RBF 核、拟合与求值
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
    cleanup = onCleanup(@() fclose(fileId)); 
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

%% 局部函数组 3：规则网格 Ordinary Kriging 配置与主计算
function config = createKrigingConfig(implementation, variogramModel, nugget, sill, rangeValue, xCount, yCount)
% 功能：验证并保存规则网格 Ordinary Kriging 的协方差参数。
% 输入：当前实现使用可分离 Gaussian 协方差，因此只接受 gaussian 模型。
%       nugget 是相对信号方差的块金/数值稳定项；range 单位为 mm。
% 输出：config 为 separableGlobalOrdinaryKrigingGrid 使用的参数结构体。
    implementation = lower(string(implementation));
    if implementation ~= "separable-global-ordinary"
        error('当前 Kriging 实现必须为 separable-global-ordinary。');
    end
    variogramModel = lower(string(variogramModel));
    if variogramModel ~= "gaussian"
        error(['separable-global-ordinary 依赖 Gaussian 可分离协方差；' ...
            '请将 krigingVariogramModel 设置为 gaussian。']);
    end
    if xCount < 2 || yCount < 2
        error('Ordinary Kriging 至少需要 2×2 原始规则采样点。');
    end
    if ~isfinite(nugget) || nugget < 0 || ~isfinite(sill) || sill <= 0 || ...
            ~isfinite(rangeValue) || rangeValue <= 0
        error('Kriging 的 nugget、sill、range 参数必须为有限且有效的数值。');
    end
    config.implementation = char(implementation);
    config.variogramModel = char(variogramModel);
    config.nugget = nugget;
    config.sill = sill;
    config.range = rangeValue;
end

function [bxOutput, byOutput, bzOutput] = separableGlobalOrdinaryKrigingGrid( ...
        xNodes, yNodes, bxInput, byInput, bzInput, xQuery, yQuery, config)
% 功能：在规则目标网格执行严格的全局 Ordinary Kriging。
% 输入：xNodes/yNodes 是原始规则采样轴；三分量矩阵的行对应 y、列对应 x。
% 输出：bxOutput、byOutput、bzOutput 均为 length(yQuery)×length(xQuery)。
% 算法：Gaussian 协方差 C(h)=sill*exp(-(hx^2+hy^2)/range^2) 可分离为 Cx*Cy。
%       因此利用两个一维协方差矩阵的特征分解完成 C^(-1) 运算，等价于全局
%       Ordinary Kriging，但避免显式构造 10000×10000 协方差矩阵。
    expectedSize = [numel(yNodes), numel(xNodes)];
    if ~isequal(size(bxInput), expectedSize) || ~isequal(size(byInput), expectedSize) || ...
            ~isequal(size(bzInput), expectedSize)
        error('Kriging 输入分量矩阵尺寸必须为 length(yNodes)×length(xNodes)。');
    end

    xNodes = xNodes(:)'; yNodes = yNodes(:)';
    xQuery = xQuery(:)'; yQuery = yQuery(:)';
    covarianceX = exp(-((xNodes - xNodes.').^2) / config.range^2);
    covarianceY = exp(-((yNodes - yNodes.').^2) / config.range^2);
    covarianceX = 0.5 * (covarianceX + covarianceX.');
    covarianceY = 0.5 * (covarianceY + covarianceY.');
    [eigenvectorX, eigenvalueX] = eig(covarianceX, 'vector');
    [eigenvectorY, eigenvalueY] = eig(covarianceY, 'vector');
    inverseEigenvalues = 1 ./ (config.sill * (eigenvalueY * eigenvalueX.') + ...
        config.nugget * config.sill);
    if any(~isfinite(inverseEigenvalues), 'all') || any(inverseEigenvalues(:) <= 0)
        error('Kriging 协方差矩阵不可逆；请增大 krigingNugget 或调整 krigingRange。');
    end

    % Ordinary Kriging 将未知常数均值作为拉格朗日项，而不是假设均值为零。
    inverseCovarianceOne = solveSeparableKrigingSystem(ones(expectedSize), ...
        eigenvectorY, eigenvectorX, inverseEigenvalues);
    meanDenominator = sum(inverseCovarianceOne, 'all');
    if ~isfinite(meanDenominator) || abs(meanDenominator) <= eps
        error('Ordinary Kriging 常数均值项退化；请检查协方差参数。');
    end
    covarianceXQuery = exp(-((xQuery(:) - xNodes).^2) / config.range^2);
    covarianceYQuery = exp(-((yQuery(:) - yNodes).^2) / config.range^2);
    bxOutput = predictSeparableOrdinaryKrigingComponent(bxInput, covarianceXQuery, ...
        covarianceYQuery, eigenvectorY, eigenvectorX, inverseEigenvalues, ...
        inverseCovarianceOne, meanDenominator, config.sill);
    byOutput = predictSeparableOrdinaryKrigingComponent(byInput, covarianceXQuery, ...
        covarianceYQuery, eigenvectorY, eigenvectorX, inverseEigenvalues, ...
        inverseCovarianceOne, meanDenominator, config.sill);
    bzOutput = predictSeparableOrdinaryKrigingComponent(bzInput, covarianceXQuery, ...
        covarianceYQuery, eigenvectorY, eigenvectorX, inverseEigenvalues, ...
        inverseCovarianceOne, meanDenominator, config.sill);
end

function solution = solveSeparableKrigingSystem(values, eigenvectorY, eigenvectorX, inverseEigenvalues)
% 功能：计算 C^(-1)*values，其中 C 是规则网格上的可分离 Gaussian 协方差矩阵。
% 说明：该函数只在输入节点尺寸上工作，100×100 数据也只需处理 100×100 矩阵。
    spectralValues = eigenvectorY.' * values * eigenvectorX;
    solution = eigenvectorY * (spectralValues .* inverseEigenvalues) * eigenvectorX.';
end

function output = predictSeparableOrdinaryKrigingComponent(values, covarianceXQuery, ...
        covarianceYQuery, eigenvectorY, eigenvectorX, inverseEigenvalues, ...
        inverseCovarianceOne, meanDenominator, sill)
% 功能：用 Ordinary Kriging 的常数未知均值公式预测单个磁场分量。
% 公式：beta=(1'*C^(-1)*z)/(1'*C^(-1)*1)，zHat=beta+c'*C^(-1)*(z-beta)。
    inverseCovarianceValues = solveSeparableKrigingSystem(values, ...
        eigenvectorY, eigenvectorX, inverseEigenvalues);
    estimatedMean = sum(inverseCovarianceValues, 'all') / meanDenominator;
    alpha = solveSeparableKrigingSystem(values - estimatedMean, ...
        eigenvectorY, eigenvectorX, inverseEigenvalues);
    output = estimatedMean + sill * covarianceYQuery * alpha * covarianceXQuery.';
end

%% 局部函数组 4：Smoothing Thin Plate Spline 主计算
function [bxOutput, byOutput, bzOutput, tpsModel] = smoothingThinPlateSplineGrid( ...
        xNodes, yNodes, bxInput, byInput, bzInput, xQuery, yQuery, smoothingParameter, maxControlPointsPerAxis)
% 功能：拟合一次二维 Smoothing Thin Plate Spline，并在规则目标网格上求值。
% 输入：xNodes/yNodes 和三个 Input 为原始低分辨率采样；xQuery/yQuery 为目标物理
%       坐标；smoothingParameter 取值在 0~1。maxControlPointsPerAxis 限制每个方向
%       的均匀控制节点数，避免 100×100 输入的全局 TPS 方程规模过大而卡死。
% 输出：三个场的尺寸均为 length(yQuery)×length(xQuery)；tpsModel 保存已拟合样条，
%       可在 Ground Truth 网格复用，不需要再次拟合。
% 算法：对均匀控制节点上的 Bx、By、Bz 分别执行 tpaps，再由 fnval 求值。
%       坐标先归一化到 [0,1]，降低薄板样条拟合的数值尺度敏感性。
    expectedSize = [numel(yNodes), numel(xNodes)];
    if ~isequal(size(bxInput), expectedSize) || ~isequal(size(byInput), expectedSize) || ...
            ~isequal(size(bzInput), expectedSize)
        error('TPS 输入分量矩阵尺寸必须为 length(yNodes)×length(xNodes)。');
    end
    if ~isscalar(smoothingParameter) || ~isfinite(smoothingParameter) || ...
            smoothingParameter < 0 || smoothingParameter > 1
        error('TPS 平滑参数必须是 0 到 1 之间的有限数。');
    end
    if ~isscalar(maxControlPointsPerAxis) || ~isfinite(maxControlPointsPerAxis) || ...
            maxControlPointsPerAxis < 3 || mod(maxControlPointsPerAxis, 1) ~= 0
        error('TPS 每方向最大控制节点数必须是不小于 3 的整数。');
    end
    if exist('tpaps', 'file') ~= 2 || exist('fnval', 'file') ~= 2
        error(['未找到 tpaps 或 fnval。Smoothing Thin Plate Spline 需要 Curve Fitting ' ...
            'Toolbox；请安装/启用该工具箱后重新运行。']);
    end

    xControlIndex = selectUniformTpsControlIndices(numel(xNodes), maxControlPointsPerAxis);
    yControlIndex = selectUniformTpsControlIndices(numel(yNodes), maxControlPointsPerAxis);
    xControl = xNodes(xControlIndex);
    yControl = yNodes(yControlIndex);
    xMinimum = min(xNodes);
    yMinimum = min(yNodes);
    xScale = max(xNodes) - xMinimum;
    yScale = max(yNodes) - yMinimum;
    if xScale <= 0 || yScale <= 0
        error('TPS 的 x 或 y 坐标范围为零，无法建立二维薄板样条。');
    end
    xControlNormalized = (xControl - xMinimum) / xScale;
    yControlNormalized = (yControl - yMinimum) / yScale;
    [xControlMesh, yControlMesh] = meshgrid(xControlNormalized, yControlNormalized);
    controlSites = [xControlMesh(:).'; yControlMesh(:).'];
    bxControl = bxInput(yControlIndex, xControlIndex);
    byControl = byInput(yControlIndex, xControlIndex);
    bzControl = bzInput(yControlIndex, xControlIndex);
    fprintf('  TPS 控制网格：%d×%d（共 %d 点）\n', ...
        numel(xControlIndex), numel(yControlIndex), numel(controlSites) / 2);
    fprintf('  TPS 正在拟合 Bx...\n'); drawnow;
    tpsModel.bxSpline = tpaps(controlSites, bxControl(:).', smoothingParameter);
    fprintf('  TPS 正在拟合 By...\n'); drawnow;
    tpsModel.bySpline = tpaps(controlSites, byControl(:).', smoothingParameter);
    fprintf('  TPS 正在拟合 Bz...\n'); drawnow;
    tpsModel.bzSpline = tpaps(controlSites, bzControl(:).', smoothingParameter);
    tpsModel.xMinimum = xMinimum;
    tpsModel.yMinimum = yMinimum;
    tpsModel.xScale = xScale;
    tpsModel.yScale = yScale;
    tpsModel.controlGridSize = [numel(yControlIndex), numel(xControlIndex)];
    tpsModel.controlPointCount = numel(xControlIndex) * numel(yControlIndex);
    tpsModel.smoothingParameter = smoothingParameter;
    [bxOutput, byOutput, bzOutput] = evaluateSmoothingThinPlateSplineGrid(tpsModel, xQuery, yQuery);
end

function [bxOutput, byOutput, bzOutput] = evaluateSmoothingThinPlateSplineGrid(tpsModel, xQuery, yQuery)
% 功能：在任意规则物理网格上复用已经拟合的 TPS 模型。
% 输入：tpsModel 由 smoothingThinPlateSplineGrid 建立；xQuery/yQuery 为目标物理坐标。
% 输出：三个输出均为 length(yQuery)×length(xQuery)，行对应 y、列对应 x。
    xQueryNormalized = (xQuery - tpsModel.xMinimum) / tpsModel.xScale;
    yQueryNormalized = (yQuery - tpsModel.yMinimum) / tpsModel.yScale;
    [xQueryMesh, yQueryMesh] = meshgrid(xQueryNormalized, yQueryNormalized);
    querySites = [xQueryMesh(:).'; yQueryMesh(:).'];
    fprintf('  TPS 正在对 %d×%d 目标网格计算 Bx...\n', numel(yQuery), numel(xQuery)); drawnow;
    bxOutput = reshape(fnval(tpsModel.bxSpline, querySites), numel(yQuery), numel(xQuery));
    fprintf('  TPS 正在对 %d×%d 目标网格计算 By...\n', numel(yQuery), numel(xQuery)); drawnow;
    byOutput = reshape(fnval(tpsModel.bySpline, querySites), numel(yQuery), numel(xQuery));
    fprintf('  TPS 正在对 %d×%d 目标网格计算 Bz...\n', numel(yQuery), numel(xQuery)); drawnow;
    bzOutput = reshape(fnval(tpsModel.bzSpline, querySites), numel(yQuery), numel(xQuery));
end

function controlIndex = selectUniformTpsControlIndices(nodeCount, maximumCount)
% 功能：在规则节点中均匀选择 TPS 控制节点，并始终保留两个边界。
% 输入：nodeCount 为原始坐标数；maximumCount 为每方向允许的最大控制节点数。
% 输出：controlIndex 是严格递增的整数索引；原始节点较少时直接返回全部索引。
    controlCount = min(nodeCount, maximumCount);
    controlIndex = unique(round(linspace(1, nodeCount, controlCount)));
    if numel(controlIndex) ~= controlCount
        error('无法构造所需数量的 TPS 均匀控制节点。');
    end
end

%% 局部函数组 6：三方法公共剖面、峰值和结果摘要
function plotFourMethodProfileTile(layout, originalAxis, originalValues, highAxis, rbfValues, krigingValues, tpsValues, xLabelText, yLabelText, titleText, trainingGridText, yLimits)
% 功能：在同一中心剖面中绘制原始离散采样以及 RBF、Kriging、TPS 三种重建曲线。
% 输入：originalAxis/originalValues 是原始 10×10 或 100×100 采样曲线；highAxis
%       及其三条 values 是共同高分辨率网格上的重建曲线；yLimits 为空时自动缩放，
%       非空时使用指定纵轴范围，保证同类物理量的剖面可直接比较。
% 输出：无，原始数据为黑色圆点线，三种方法为不同颜色的连续线。
    nexttile(layout);
    plot(originalAxis, originalValues, 'ko-', 'LineWidth', 1.1, ...
        'MarkerSize', 4, 'MarkerFaceColor', 'k'); hold on;
    plot(highAxis, rbfValues, 'r-', 'LineWidth', 1.2);
    plot(highAxis, krigingValues, 'b-', 'LineWidth', 1.2);
    plot(highAxis, tpsValues, 'Color', [0.10, 0.55, 0.20], 'LineWidth', 1.2); grid on;
    xlabel(xLabelText);
    ylabel(yLabelText);
    title(titleText);
    if ~isempty(yLimits)
        ylim(yLimits);
    end
    legend(['原始输入 ' trainingGridText], 'RBF', 'Global Ordinary Kriging', 'Smoothing TPS', 'Location', 'best');
end

function plotGroundTruthProfileTile(layout, axisValues, groundTruthValues, rbfValues, krigingValues, tpsValues, xLabelText, yLabelText, titleText)
% 功能：在 Ground Truth 物理网格上绘制真值、RBF、Kriging、TPS 四条中心剖面。
% 输入：所有 values 必须同长度且对应 axisValues 的完全相同物理坐标。
% 输出：无，直接向 tiledlayout 添加四方法比较曲线。
    nexttile(layout);
    plot(axisValues, groundTruthValues, 'k-', 'LineWidth', 1.5); hold on;
    plot(axisValues, rbfValues, 'r-', 'LineWidth', 1.1);
    plot(axisValues, krigingValues, 'b-', 'LineWidth', 1.1);
    plot(axisValues, tpsValues, 'Color', [0.10, 0.55, 0.20], 'LineWidth', 1.1); grid on;
    xlabel(xLabelText);
    ylabel(yLabelText);
    title(titleText);
    legend('Ground Truth', 'RBF', 'Global Ordinary Kriging', 'Smoothing TPS', 'Location', 'best');
end

function comparisonTable = buildThreeMethodComparisonTable(bmagRbf, ctRbf, bmagKriging, ctKriging, bmagTps, ctTps, xValues, yValues)
% 功能：汇总 RBF、Kriging、TPS 的 |B|、Ct_2D 峰值及对应物理位置，非真值误差表。
% 输入：六个共同目标网格物理量和对应 x/y 坐标。
% 输出：每行对应一个物理量，三种方法各有峰值与位置三列。
    rbfBmagPeak = findFieldPeak(bmagRbf, xValues, yValues);
    krigingBmagPeak = findFieldPeak(bmagKriging, xValues, yValues);
    tpsBmagPeak = findFieldPeak(bmagTps, xValues, yValues);
    rbfCtPeak = findFieldPeak(ctRbf, xValues, yValues);
    krigingCtPeak = findFieldPeak(ctKriging, xValues, yValues);
    tpsCtPeak = findFieldPeak(ctTps, xValues, yValues);
    comparisonTable = table(["|B|"; "Ct_2D"], ...
        [rbfBmagPeak.value; rbfCtPeak.value], [rbfBmagPeak.x; rbfCtPeak.x], [rbfBmagPeak.y; rbfCtPeak.y], ...
        [krigingBmagPeak.value; krigingCtPeak.value], [krigingBmagPeak.x; krigingCtPeak.x], [krigingBmagPeak.y; krigingCtPeak.y], ...
        [tpsBmagPeak.value; tpsCtPeak.value], [tpsBmagPeak.x; tpsCtPeak.x], [tpsBmagPeak.y; tpsCtPeak.y], ...
        'VariableNames', {'PhysicalQuantity', 'RBFPeak', 'RBFPeakX_mm', 'RBFPeakY_mm', ...
        'KrigingPeak', 'KrigingPeakX_mm', 'KrigingPeakY_mm', 'TPSPeak', 'TPSPeakX_mm', 'TPSPeakY_mm'});
end

function comparisonTable = buildMethodComparisonTable(bmagRbf, ctRbf, bmagKriging, ctKriging, xValues, yValues)
% 功能：汇总 RBF 与 Kriging 的 |B|、Ct_2D 峰值及峰值物理位置。
% 输入：四个 1000×1000 或 2000×2000 图像及其共同 x/y 坐标。
% 输出：comparisonTable 仅是方法间特征比较，不是以任一方法为真值的误差表。
    rbfBmagPeak = findFieldPeak(bmagRbf, xValues, yValues);
    krigingBmagPeak = findFieldPeak(bmagKriging, xValues, yValues);
    rbfCtPeak = findFieldPeak(ctRbf, xValues, yValues);
    krigingCtPeak = findFieldPeak(ctKriging, xValues, yValues);
    comparisonTable = table(["|B|"; "Ct_2D"], ...
        [rbfBmagPeak.value; rbfCtPeak.value], ...
        [rbfBmagPeak.x; rbfCtPeak.x], ...
        [rbfBmagPeak.y; rbfCtPeak.y], ...
        [krigingBmagPeak.value; krigingCtPeak.value], ...
        [krigingBmagPeak.x; krigingCtPeak.x], ...
        [krigingBmagPeak.y; krigingCtPeak.y], ...
        'VariableNames', {'PhysicalQuantity', 'RBFPeak', 'RBFPeakX_mm', 'RBFPeakY_mm', ...
        'KrigingPeak', 'KrigingPeakX_mm', 'KrigingPeakY_mm'});
end

function peak = findFieldPeak(field, xValues, yValues)
% 功能：在二维物理量图中定位最大值和其对应的 x/y 坐标。
% 输入：field 的行对应 y、列对应 x；xValues/yValues 为图像坐标。
% 输出：peak.value、peak.x、peak.y。
    [peak.value, linearIndex] = max(field, [], 'all');
    [rowIndex, columnIndex] = ind2sub(size(field), linearIndex);
    peak.x = xValues(columnIndex);
    peak.y = yValues(rowIndex);
end

function writeKrigingSummary(filePath, krigingResults)
% 功能：写入 Kriging 参数、输出尺寸及 RBF/Kriging 特征比较说明。
% 输入：filePath 为摘要文件路径；krigingResults 为新增模块的结果结构体。
% 输出：无；文件无法创建时 warning，但不影响已完成的计算和 MAT 保存。
    fileId = fopen(filePath, 'w');
    if fileId < 0
        warning('无法写入 Kriging 摘要：%s', filePath);
        return;
    end
    cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
    fprintf(fileId, 'Separable Global Ordinary Kriging 三分量超分辨率重建摘要\n');
    fprintf(fileId, '输入文件：%s\n', krigingResults.inputFile);
    fprintf(fileId, '训练网格：%d×%d；重建网格：%d×%d\n', ...
        krigingResults.trainingResolution(1), krigingResults.trainingResolution(2), ...
        krigingResults.reconstructionResolution(1), krigingResults.reconstructionResolution(2));
    fprintf(fileId, ['实现：%s；协方差：%s；nugget：%.12g；sill：%.12g；' ...
        'rangeMultiplier：%.12g；range：%.12g mm\n\n'], ...
        krigingResults.implementation, krigingResults.variogramModel, krigingResults.nugget, ...
        krigingResults.sill, krigingResults.rangeMultiplier, krigingResults.range_mm);
    fprintf(fileId, '当前输入不含真实高分辨率 Ground Truth，以下是 RBF 与 Kriging 特征比较，不是误差评价。\n');
    for index = 1:height(krigingResults.comparisonTable)
        row = krigingResults.comparisonTable(index, :);
        fprintf(fileId, '%s: RBF 峰值=%.12g，位置=(%.12g, %.12g) mm；', ...
            row.PhysicalQuantity, row.RBFPeak, row.RBFPeakX_mm, row.RBFPeakY_mm);
        fprintf(fileId, 'Kriging 峰值=%.12g，位置=(%.12g, %.12g) mm\n', ...
            row.KrigingPeak, row.KrigingPeakX_mm, row.KrigingPeakY_mm);
    end
end

function writeTpsSummary(filePath, tpsResults)
% 功能：写入 Smoothing TPS 参数、输出尺寸与三方法峰值比较说明。
% 输入：filePath 是文本输出路径；tpsResults 是 TPS 主模块保存的结果结构体。
% 输出：无；无法创建文件时 warning，不影响已完成的插值计算。
    fileId = fopen(filePath, 'w');
    if fileId < 0
        warning('无法写入 TPS 摘要：%s', filePath);
        return;
    end
    cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
    fprintf(fileId, 'Smoothing Thin Plate Spline 三分量超分辨率重建摘要\n');
    fprintf(fileId, '输入文件：%s\n', tpsResults.inputFile);
    fprintf(fileId, '训练网格：%d×%d；重建网格：%d×%d\n', ...
        tpsResults.trainingResolution(1), tpsResults.trainingResolution(2), ...
        tpsResults.reconstructionResolution(1), tpsResults.reconstructionResolution(2));
    fprintf(fileId, 'TPS 平滑参数=%.12g（0 表示强平滑，1 表示接近严格插值）\n\n', ...
        tpsResults.smoothingParameter);
    fprintf(fileId, 'TPS 控制网格：%d×%d（共 %d 点；每方向上限=%d）。\n', ...
        tpsResults.controlGridSize(1), tpsResults.controlGridSize(2), ...
        tpsResults.controlPointCount, tpsResults.maxControlPointsPerAxis);
    fprintf(fileId, 'TPS 对原始低分辨率 Bx、By、Bz 分量分别拟合二维平滑薄板样条。\n');
    fprintf(fileId, '模型只拟合一次，并只在主重建网格求值；Ground Truth 将重采样至该网格用于评价。\n');
    fprintf(fileId, 'TPS 不使用 RBF 或 Kriging 重建结果。\n');
    fprintf(fileId, '下表仅比较 RBF、Kriging、TPS 的输出峰值和位置，不是 Ground Truth 误差。\n');
    for index = 1:height(tpsResults.comparisonTable)
        row = tpsResults.comparisonTable(index, :);
        fprintf(fileId, '%s: RBF=%.12g，Kriging=%.12g，TPS=%.12g\n', ...
            row.PhysicalQuantity, row.RBFPeak, row.KrigingPeak, row.TPSPeak);
    end
end

%% 局部函数组 7：Ground Truth 读取、校验和误差评价
function groundTruth = readComsolGroundTruth1000(filePath)
% 功能：直接读取 COMSOL 导出的真实 1000×1000 磁场数据，不对其进行插值。
% 输入：filePath 为含 x、y、z、Bx、By、Bz 六列的 COMSOL 文本文件路径。
% 输出：groundTruth.x/y 为物理坐标（mm），Bx/By/Bz 均为 1000×1000 网格。
% 检查：数值行必须恰为 1000000，x/y 各 1000 个唯一坐标，且仅有一个 z 平面。
    rawData = readmatrix(filePath, 'FileType', 'text', 'NumHeaderLines', 9);
    if size(rawData, 2) < 6
        error('Ground Truth 数值列少于 6 列，无法读取 x、y、z、Bx、By、Bz。');
    end
    rawData = rawData(:, 1:6);
    invalidRows = any(~isfinite(rawData), 2);
    if any(invalidRows)
        error(['Ground Truth 数值数据含 %d 行 NaN/Inf。Ground Truth 的所有误差指标均会受影响，' ...
            '因此不允许继续计算。'], sum(invalidRows));
    end

    x = rawData(:, 1);
    y = rawData(:, 2);
    z = rawData(:, 3);
    bx = rawData(:, 4);
    by = rawData(:, 5);
    bz = rawData(:, 6);
    xValues = unique(x, 'sorted')';
    yValues = unique(y, 'sorted')';
    zValues = unique(z, 'sorted')';
    if numel(x) ~= 1000000 || numel(xValues) ~= 1000 || numel(yValues) ~= 1000 || numel(zValues) ~= 1
        error(['Ground Truth 必须是完整 1000×1000 单一 z 平面；实际为 %d 个点、' ...
            '%d 个 x 值、%d 个 y 值、%d 个 z 值。'], ...
            numel(x), numel(xValues), numel(yValues), numel(zValues));
    end
    [isRegularX, xSpacing, xDeviation, xTolerance] = isRegularComsolGroundTruthAxis(xValues);
    [isRegularY, ySpacing, yDeviation, yTolerance] = isRegularComsolGroundTruthAxis(yValues);
    if ~isRegularX || ~isRegularY
        error(['Ground Truth 的 x 或 y 坐标不是规则等间距网格，无法进行逐点比较。\n' ...
            'x：名义间隔=%.12g mm，最大坐标偏差=%.12g mm，允许偏差=%.12g mm。\n' ...
            'y：名义间隔=%.12g mm，最大坐标偏差=%.12g mm，允许偏差=%.12g mm。'], ...
            xSpacing, xDeviation, xTolerance, ySpacing, yDeviation, yTolerance);
    end
    [~, xIndex] = ismember(x, xValues);
    [~, yIndex] = ismember(y, yValues);
    gridSize = [1000, 1000];
    groundTruth.x = xValues;
    groundTruth.y = yValues;
    groundTruth.z_mm = zValues(1);
    groundTruth.Bx = accumarray([yIndex, xIndex], bx, gridSize, @mean, NaN);
    groundTruth.By = accumarray([yIndex, xIndex], by, gridSize, @mean, NaN);
    groundTruth.Bz = accumarray([yIndex, xIndex], bz, gridSize, @mean, NaN);
    % 每个网格位置都必须有值；重复坐标必然会导致其他位置缺失，从而被此检查捕获。
    assertGroundTruthFinite(groundTruth.Bx, 'Ground Truth Bx（同时检查缺失/重复坐标）');
    assertGroundTruthFinite(groundTruth.By, 'Ground Truth By（同时检查缺失/重复坐标）');
    assertGroundTruthFinite(groundTruth.Bz, 'Ground Truth Bz（同时检查缺失/重复坐标）');
    fprintf('Ground Truth 已读取：1000×1000；x=[%.12g, %.12g] mm；y=[%.12g, %.12g] mm；z=%.12g mm\n', ...
        xValues(1), xValues(end), yValues(1), yValues(end), zValues(1));
end

function isMatch = coordinateVectorsMatch(firstValues, secondValues)
% 功能：以数值容差比较两个物理坐标向量是否逐点一致。
% 输入：firstValues、secondValues 为升序 x 或 y 坐标向量，单位均为 mm。
% 输出：isMatch 为 true 表示长度和每个坐标点都一致。
    isMatch = isequal(size(firstValues), size(secondValues));
    if ~isMatch
        return;
    end
    tolerance = max(1e-9, 1e-10 * max([1; abs(firstValues(:)); abs(secondValues(:))]));
    isMatch = all(abs(firstValues(:) - secondValues(:)) <= tolerance);
end

function evaluationTruth = resampleGroundTruthToReconstructionGrid(sourceTruth, xTarget, yTarget)
% 功能：把原始 COMSOL Ground Truth 的三分量从 1000×1000 网格重采样到重建网格。
% 输入：sourceTruth 为原始真值结构体；xTarget/yTarget 为实际重建坐标。
% 输出：evaluationTruth 与重建结果同尺寸，包含 Bx、By、Bz、|B| 和 Ct_2D。
% 注意：该函数只改变 Ground Truth 评价数据，不改变任何 RBF、Kriging 或 TPS 结果。
    xTarget = xTarget(:)';
    yTarget = yTarget(:)';
    [xMesh, yMesh] = meshgrid(xTarget, yTarget);
    bxInterpolant = griddedInterpolant({sourceTruth.y, sourceTruth.x}, sourceTruth.Bx, ...
        'linear', 'nearest');
    byInterpolant = griddedInterpolant({sourceTruth.y, sourceTruth.x}, sourceTruth.By, ...
        'linear', 'nearest');
    bzInterpolant = griddedInterpolant({sourceTruth.y, sourceTruth.x}, sourceTruth.Bz, ...
        'linear', 'nearest');
    evaluationTruth.x = xTarget;
    evaluationTruth.y = yTarget;
    evaluationTruth.z_mm = sourceTruth.z_mm;
    evaluationTruth.Bx = bxInterpolant(yMesh, xMesh);
    evaluationTruth.By = byInterpolant(yMesh, xMesh);
    evaluationTruth.Bz = bzInterpolant(yMesh, xMesh);
    evaluationTruth.Bmag = sqrt(evaluationTruth.Bx.^2 + evaluationTruth.By.^2 + evaluationTruth.Bz.^2);
    [evaluationTruth.BxDx, evaluationTruth.BxDy] = gradient(evaluationTruth.Bx, xTarget, yTarget);
    [evaluationTruth.ByDx, evaluationTruth.ByDy] = gradient(evaluationTruth.By, xTarget, yTarget);
    [evaluationTruth.BzDx, evaluationTruth.BzDy] = gradient(evaluationTruth.Bz, xTarget, yTarget);
    evaluationTruth.Ct2D = evaluationTruth.BxDx.^2 + evaluationTruth.BxDy.^2 + ...
        evaluationTruth.ByDx.^2 + evaluationTruth.ByDy.^2 + ...
        evaluationTruth.BzDx.^2 + evaluationTruth.BzDy.^2;
    evaluationTruth = rmfield(evaluationTruth, ...
        {'BxDx', 'BxDy', 'ByDx', 'ByDy', 'BzDx', 'BzDy'});
    assertGroundTruthFinite(evaluationTruth.Bx, '重采样 Ground Truth Bx');
    assertGroundTruthFinite(evaluationTruth.By, '重采样 Ground Truth By');
    assertGroundTruthFinite(evaluationTruth.Bz, '重采样 Ground Truth Bz');
    assertGroundTruthFinite(evaluationTruth.Bmag, '重采样 Ground Truth |B|');
    assertGroundTruthFinite(evaluationTruth.Ct2D, '重采样 Ground Truth Ct_2D');
end

function validateGroundTruthComparison(groundTruth, evaluationX, evaluationY, ...
        bxRbf, byRbf, bzRbf, ctRbf, bxKriging, byKriging, bzKriging, ctKriging, ...
        bxTps, byTps, bzTps, ctTps)
% 功能：在误差运算前严格检查评价网格尺寸、坐标一致性和有限数状态。
% 输入：groundTruth 为直接读取的真值；evaluationX/Y 是重建评价网格的物理坐标；
%       其余输入是 RBF/Kriging/TPS 在该评价网格上的分量和 Ct_2D。
% 输出：无；任何不一致、NaN 或 Inf 均给出明确错误，绝不静默忽略。
    expectedSize = [numel(evaluationY), numel(evaluationX)];
    if ~isequal(size(groundTruth.Bx), expectedSize) || ~isequal(size(groundTruth.By), expectedSize) || ...
            ~isequal(size(groundTruth.Bz), expectedSize)
        error('评价用 Ground Truth 的 Bx、By、Bz 必须全部为 %d×%d。', ...
            expectedSize(1), expectedSize(2));
    end
    if ~coordinateVectorsMatch(evaluationX, groundTruth.x) || ...
            ~coordinateVectorsMatch(evaluationY, groundTruth.y)
        error(['重建评价坐标与 Ground Truth 坐标不一致，不能直接计算逐点误差。' ...
            '请在 Ground Truth 的 x/y 网格上重新求值。']);
    end
    fields = {bxRbf, byRbf, bzRbf, ctRbf, bxKriging, byKriging, bzKriging, ctKriging, ...
        bxTps, byTps, bzTps, ctTps};
    names = {'RBF Bx', 'RBF By', 'RBF Bz', 'RBF Ct_2D', ...
        'Kriging Bx', 'Kriging By', 'Kriging Bz', 'Kriging Ct_2D', ...
        'TPS Bx', 'TPS By', 'TPS Bz', 'TPS Ct_2D'};
    for index = 1:numel(fields)
        if ~isequal(size(fields{index}), expectedSize)
            error('%s 的尺寸为 %d×%d，必须为 %d×%d 才能进行 Ground Truth 评价。', ...
                names{index}, size(fields{index}, 1), size(fields{index}, 2), ...
                expectedSize(1), expectedSize(2));
        end
        assertGroundTruthFinite(fields{index}, names{index});
    end
end

function assertGroundTruthFinite(values, valueName)
% 功能：统计 NaN/Inf 数量，并阻止不完整数组参与 RMSE、MAE、R2、NRMSE 等计算。
% 输入：values 为待检查数组；valueName 为报错中显示的变量说明。
% 输出：无；数据完整时正常返回，存在非有限数时明确报错。
    nanCount = sum(isnan(values), 'all');
    infCount = sum(isinf(values), 'all');
    if nanCount > 0 || infCount > 0
        error(['%s 含有 %d 个 NaN 和 %d 个 Inf。它们会使误差场、RMSE、MAE、R2、' ...
            'NRMSE、最大绝对误差和峰值评价失效，因此不允许继续计算。'], ...
            valueName, nanCount, infCount);
    end
end

function metric = calculateGroundTruthMetrics(groundTruthValues, reconstructedValues)
% 功能：在完整 Ground Truth 网格上计算 RMSE、MAE、R2、NRMSE 和最大绝对误差。
% 输入：groundTruthValues 与 reconstructedValues 必须为同尺寸、同坐标的有限数组。
% 输出：metric 结构体。NRMSE 使用 Ground Truth 极差归一化；R2 使用 Ground Truth 方差。
    assertGroundTruthFinite(groundTruthValues, 'Ground Truth 评价数据');
    assertGroundTruthFinite(reconstructedValues, '重建评价数据');
    if ~isequal(size(groundTruthValues), size(reconstructedValues))
        error('Ground Truth 与重建结果尺寸不一致，不能计算误差指标。');
    end
    residual = reconstructedValues - groundTruthValues;
    metric.RMSE = sqrt(mean(residual.^2, 'all'));
    metric.MAE = mean(abs(residual), 'all');
    metric.MaxError = max(abs(residual), [], 'all');
    groundTruthMean = mean(groundTruthValues, 'all');
    totalVariance = sum((groundTruthValues - groundTruthMean).^2, 'all');
    groundTruthRange = max(groundTruthValues, [], 'all') - min(groundTruthValues, [], 'all');
    if totalVariance <= eps
        metric.R2 = NaN;
        warning('Ground Truth 方差为零，R2 无法定义。');
    else
        metric.R2 = 1 - sum(residual.^2, 'all') / totalVariance;
    end
    if groundTruthRange <= eps
        metric.NRMSE = NaN;
        warning('Ground Truth 极差为零，NRMSE 无法定义。');
    else
        metric.NRMSE = metric.RMSE / groundTruthRange;
    end
end

function summaryTable = buildGroundTruthSummaryTable(groundTruthBmag, rbfBmag, krigingBmag, tpsBmag, xValues, yValues)
% 功能：建立用户要求的 |B| 误差汇总表，逐行比较 RBF、Kriging 与 Smoothing TPS。
% 输入：四个同尺寸 |B| 数组和完全相同的重建网格 x/y 物理坐标（mm）。
% 输出：Method、RMSE、MAE、NRMSE、R2、MaxError、PeakValueError、PeakPositionError。
    rbfMetric = calculateGroundTruthMetrics(groundTruthBmag, rbfBmag);
    krigingMetric = calculateGroundTruthMetrics(groundTruthBmag, krigingBmag);
    tpsMetric = calculateGroundTruthMetrics(groundTruthBmag, tpsBmag);
    groundTruthPeak = findFieldPeak(groundTruthBmag, xValues, yValues);
    rbfPeak = findFieldPeak(rbfBmag, xValues, yValues);
    krigingPeak = findFieldPeak(krigingBmag, xValues, yValues);
    tpsPeak = findFieldPeak(tpsBmag, xValues, yValues);
    summaryTable = table(["RBF"; "Global Ordinary Kriging"; "Smoothing TPS"], ...
        [rbfMetric.RMSE; krigingMetric.RMSE; tpsMetric.RMSE], ...
        [rbfMetric.MAE; krigingMetric.MAE; tpsMetric.MAE], ...
        [rbfMetric.NRMSE; krigingMetric.NRMSE; tpsMetric.NRMSE], ...
        [rbfMetric.R2; krigingMetric.R2; tpsMetric.R2], ...
        [rbfMetric.MaxError; krigingMetric.MaxError; tpsMetric.MaxError], ...
        [abs(rbfPeak.value - groundTruthPeak.value); abs(krigingPeak.value - groundTruthPeak.value); abs(tpsPeak.value - groundTruthPeak.value)], ...
        [hypot(rbfPeak.x - groundTruthPeak.x, rbfPeak.y - groundTruthPeak.y); ...
         hypot(krigingPeak.x - groundTruthPeak.x, krigingPeak.y - groundTruthPeak.y); ...
         hypot(tpsPeak.x - groundTruthPeak.x, tpsPeak.y - groundTruthPeak.y)], ...
        'VariableNames', {'Method', 'RMSE', 'MAE', 'NRMSE', 'R2', 'MaxError', ...
        'PeakValueError', 'PeakPositionError_mm'});
end

function detailedTable = buildDetailedGroundTruthMetrics(groundTruth, ...
        bxRbf, byRbf, bzRbf, bmagRbf, ctRbf, bxKriging, byKriging, bzKriging, bmagKriging, ctKriging, ...
        bxTps, byTps, bzTps, bmagTps, ctTps)
% 功能：补充 Bx、By、Bz、|B|、Ct_2D 五种物理量的三种方法完整指标表。
% 输入：Ground Truth 分量与 RBF、Kriging、TPS 在同一重建物理网格的结果。
% 输出：每个物理量三行，分别给出 RBF、Global Ordinary Kriging、Smoothing TPS 的全部指标。
    quantityNames = ["Bx"; "By"; "Bz"; "|B|"; "Ct_2D"];
    truthValues = {groundTruth.Bx, groundTruth.By, groundTruth.Bz, groundTruth.Bmag, groundTruth.Ct2D};
    rbfValues = {bxRbf, byRbf, bzRbf, bmagRbf, ctRbf};
    krigingValues = {bxKriging, byKriging, bzKriging, bmagKriging, ctKriging};
    tpsValues = {bxTps, byTps, bzTps, bmagTps, ctTps};
    quantity = strings(15, 1);
    method = strings(15, 1);
    rmse = zeros(15, 1);
    mae = zeros(15, 1);
    nrmse = zeros(15, 1);
    r2 = zeros(15, 1);
    maxError = zeros(15, 1);
    row = 0;
    for index = 1:numel(quantityNames)
        rbfMetric = calculateGroundTruthMetrics(truthValues{index}, rbfValues{index});
        krigingMetric = calculateGroundTruthMetrics(truthValues{index}, krigingValues{index});
        tpsMetric = calculateGroundTruthMetrics(truthValues{index}, tpsValues{index});
        row = row + 1;
        quantity(row) = quantityNames(index); method(row) = "RBF";
        rmse(row) = rbfMetric.RMSE; mae(row) = rbfMetric.MAE; nrmse(row) = rbfMetric.NRMSE;
        r2(row) = rbfMetric.R2; maxError(row) = rbfMetric.MaxError;
        row = row + 1;
        quantity(row) = quantityNames(index); method(row) = "Global Ordinary Kriging";
        rmse(row) = krigingMetric.RMSE; mae(row) = krigingMetric.MAE; nrmse(row) = krigingMetric.NRMSE;
        r2(row) = krigingMetric.R2; maxError(row) = krigingMetric.MaxError;
        row = row + 1;
        quantity(row) = quantityNames(index); method(row) = "Smoothing TPS";
        rmse(row) = tpsMetric.RMSE; mae(row) = tpsMetric.MAE; nrmse(row) = tpsMetric.NRMSE;
        r2(row) = tpsMetric.R2; maxError(row) = tpsMetric.MaxError;
    end
    detailedTable = table(quantity, method, rmse, mae, nrmse, r2, maxError, ...
        'VariableNames', {'Quantity', 'Method', 'RMSE', 'MAE', 'NRMSE', 'R2', 'MaxError'});
end

function peakTable = buildGroundTruthPeakTable(groundTruthBmag, rbfBmag, krigingBmag, tpsBmag, xValues, yValues)
% 功能：列出 |B| 的 Ground Truth、RBF、Kriging、TPS 峰值及其统一物理坐标位置。
% 输入：四个同网格 |B| 数组；xValues/yValues 均为重建网格 mm 坐标。
% 输出：peakTable 的每一行对应一种来源，可直接核对峰值幅度和位置误差的来源。
    groundTruthPeak = findFieldPeak(groundTruthBmag, xValues, yValues);
    rbfPeak = findFieldPeak(rbfBmag, xValues, yValues);
    krigingPeak = findFieldPeak(krigingBmag, xValues, yValues);
    tpsPeak = findFieldPeak(tpsBmag, xValues, yValues);
    peakTable = table(["Ground Truth"; "RBF"; "Global Ordinary Kriging"; "Smoothing TPS"], ...
        [groundTruthPeak.value; rbfPeak.value; krigingPeak.value; tpsPeak.value], ...
        [groundTruthPeak.x; rbfPeak.x; krigingPeak.x; tpsPeak.x], ...
        [groundTruthPeak.y; rbfPeak.y; krigingPeak.y; tpsPeak.y], ...
        'VariableNames', {'Method', 'PeakValue_T', 'PeakX_mm', 'PeakY_mm'});
end

%% 局部函数组 8：Ground Truth 误差图与摘要输出
function plotSignedErrorTile(layout, xValues, yValues, errorData, titleText, colorLabel, limit)
% 功能：在 tiledlayout 中绘制以零为中心的二维 signed error 图。
% 输入：xValues/yValues 为 mm 物理坐标；errorData=重建-Ground Truth；limit 为
%       RBF 和 Kriging 共用的正负最大显示范围。
% 输出：无，直接在当前 figure 的下一子图中写入误差场和 colorbar。
    nexttile(layout);
    imagesc(xValues, yValues, errorData);
    set(gca, 'YDir', 'normal');
    axis image;
    xlim([min(xValues), max(xValues)]);
    ylim([min(yValues), max(yValues)]);
    caxis([-limit, limit]);
    colormap(gca, blueWhiteRedMap(256));
    xlabel('x (mm)');
    ylabel('y (mm)');
    title(titleText, 'Interpreter', 'none');
    colorbarHandle = colorbar;
    colorbarHandle.Label.String = colorLabel;
end

function map = blueWhiteRedMap(count)
% 功能：创建用于正负误差的蓝-白-红发散 colormap。
% 输入：count 为色阶数量。
% 输出：map 为 count×3 RGB 数组，蓝色代表负误差、白色代表零、红色代表正误差。
    blue = [0.10, 0.25, 0.80];
    white = [1.00, 1.00, 1.00];
    red = [0.80, 0.12, 0.12];
    lowerCount = floor(count / 2);
    upperCount = count - lowerCount;
    lowerMap = interp1([0; 1], [blue; white], linspace(0, 1, lowerCount)');
    upperMap = interp1([0; 1], [white; red], linspace(0, 1, upperCount)');
    map = [lowerMap; upperMap];
end

function writeGroundTruthSummary(filePath, groundTruthFile, summaryTable, detailedTable, peakTable)
% 功能：将 Ground Truth 文件路径、|B| 主表及完整物理量误差表写入文本摘要。
% 输入：filePath 为输出路径；groundTruthFile 为直接使用的 COMSOL 真值文件；
%       summaryTable、detailedTable、peakTable 为已计算的指标和峰值表。
% 输出：无；无法创建摘要文件时 warning，不影响已完成的误差计算。
    fileId = fopen(filePath, 'w');
    if fileId < 0
        warning('无法写入 Ground Truth 误差评价摘要：%s', filePath);
        return;
    end
    cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
    fprintf(fileId, 'Ground Truth 重采样到重建网格后的逐点误差评价摘要\n');
    fprintf(fileId, 'Ground Truth 文件：%s\n\n', groundTruthFile);
    fprintf(fileId, '|B| 主汇总表：\n');
    for index = 1:height(summaryTable)
        fprintf(fileId, ['%s: RMSE=%.12g, MAE=%.12g, NRMSE=%.12g, R2=%.12g, ' ...
            'MaxError=%.12g, PeakValueError=%.12g, PeakPositionError=%.12g mm\n'], ...
            summaryTable.Method(index), summaryTable.RMSE(index), summaryTable.MAE(index), ...
            summaryTable.NRMSE(index), summaryTable.R2(index), summaryTable.MaxError(index), ...
            summaryTable.PeakValueError(index), summaryTable.PeakPositionError_mm(index));
    end
    fprintf(fileId, '\n完整物理量误差表：\n');
    for index = 1:height(detailedTable)
        fprintf(fileId, '%s - %s: RMSE=%.12g, MAE=%.12g, NRMSE=%.12g, R2=%.12g, MaxError=%.12g\n', ...
            detailedTable.Quantity(index), detailedTable.Method(index), detailedTable.RMSE(index), ...
            detailedTable.MAE(index), detailedTable.NRMSE(index), detailedTable.R2(index), ...
            detailedTable.MaxError(index));
    end
    fprintf(fileId, '\n|B| 峰值与物理位置：\n');
    for index = 1:height(peakTable)
        fprintf(fileId, '%s: Peak=%.12g T，位置=(%.12g, %.12g) mm\n', ...
            peakTable.Method(index), peakTable.PeakValue_T(index), ...
            peakTable.PeakX_mm(index), peakTable.PeakY_mm(index));
    end
end
