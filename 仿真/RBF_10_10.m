%% RBF_10_10：RBF 插值前后 |B| 与 Ct_2D 的二维成像
% 本脚本针对一个实际测得的 x-y 平面，即单一物理 z 层的数据。
% 不直接对 |B| 或 Ct_2D 插值，而是严格遵循以下过程：
%   1. 分别对 Bx、By、Bz 进行 Gaussian RBF 重建。
%   2. 由三个重建分量计算磁场模值 |B|。
%   3. 由三个分量的 x/y 导数计算 Ct_2D。
%
% Ct_2D 被明确标记为二维可观测量。单一 z 平面不包含 dB/dz 信息，
% 因此本脚本不会将 Ct_2D 误称为完整三维 Ct。

%% 1. 参数设置
clear;
clc;
close all;

excelFile = 'D:\YAN\大论文\x1\数据\10.10.xlsx';
sheetIndex = 1;
sparseFraction = 0.25;      % 用于 RBF 训练的原始网格点比例。
imageResolution = 500;      % RBF 成像网格在每个方向上的像素数。
saveFigures = true;
figureResolution = 220;

if ~isfile(excelFile)
    [fileName, filePath] = uigetfile({'*.xlsx;*.xls', 'Excel files'}, ...
        'Select magnetic-field data');
    if isequal(fileName, 0)
        error('No input Excel file was selected.');
    end
    excelFile = fullfile(filePath, fileName);
end

scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder)
    scriptFolder = pwd;
end
[~, baseName, ~] = fileparts(excelFile);
resultFolder = fullfile(scriptFolder, ['RBF_results_' baseName]);
if ~isfolder(resultFolder)
    mkdir(resultFolder);
end

fprintf('============================================================\n');
fprintf('2D component-wise Gaussian RBF magnetic-field imaging\n');
fprintf('Input file: %s\n', excelFile);
fprintf('============================================================\n');

%% 2. 读取工作表并识别真实变量名所在行
sheetList = sheetnames(excelFile);
if sheetIndex < 1 || sheetIndex > numel(sheetList)
    error('sheetIndex=%d is outside the available worksheet range.', sheetIndex);
end
sheetName = sheetList(sheetIndex);
raw = readcell(excelFile, 'Sheet', sheetName);
[headerRow, headers, dataRows] = findHeaderAndDataRows(raw);

fprintf('\nWorksheet: %s\n', sheetName);
fprintf('Workbook size: %d rows x %d columns\n', size(raw, 1), size(raw, 2));
fprintf('Detected header row: %d; data rows: %d\n', headerRow, numel(dataRows));
fprintf('Non-empty headers:\n');
for c = 1:numel(headers)
    if isNonemptyString(headers(c))
        fprintf('  %-4s %s\n', excelColumnName(c), headers(c));
    end
end

%% 3. 解析 COMSOL 横向导出或常规逐点表格
data = readMagneticData(raw, headers, dataRows);
data.sourceFile = excelFile;
data.sheetName = sheetName;

validMask = all(isfinite([data.x, data.y, data.z, ...
    data.Bx, data.By, data.Bz]), 2);
if ~all(validMask)
    warning('Discarding %d sample(s) with non-finite values.', sum(~validMask));
    data = subsetData(data, validMask);
end
if numel(data.x) < 9
    error('At least nine finite samples are required for 2D imaging.');
end

% x-y 成像只能对应一个真实 z 层。若文件包含多个 z 层，不人为扩展为
% 虚拟平面，而是选择实际测得的中间 z 层。
[data, zSlice] = selectSingleZPlane(data);
fprintf('\nDetected format: %s\n', data.format);
fprintf('Imaging plane: z = %.12g %s\n', zSlice, data.coordinateUnit);
fprintf('Samples on imaging plane: %d\n', numel(data.x));

%% 4. 计算原始测量数据的 |B| 与 Ct_2D 图像
original = buildRegularGrid(data);
xGrid = original.x;
yGrid = original.y;
[XOriginal, YOriginal] = meshgrid(xGrid, yGrid);

% 先获得三个原始分量，再计算磁场模值 |B|。
BmagOriginal = sqrt(original.Bx.^2 + original.By.^2 + original.Bz.^2);

% gradient(F, x, y) 返回矩阵列方向的 dF/dx 与行方向的 dF/dy。
% 这里不使用 z 方向导数，因此 Ct_2D 不会被表述为三维张量不变量。
[dBxDx, dBxDy] = gradient(original.Bx, xGrid, yGrid);
[dByDx, dByDy] = gradient(original.By, xGrid, yGrid);
[dBzDx, dBzDy] = gradient(original.Bz, xGrid, yGrid);
CtOriginal = dBxDx.^2 + dBxDy.^2 + dByDx.^2 + dByDy.^2 + ...
    dBzDx.^2 + dBzDy.^2;

%% 5. 使用稀疏原始网格分别拟合 Bx、By、Bz
referenceXY = [XOriginal(:), YOriginal(:)];
referenceBx = original.Bx(:);
referenceBy = original.By(:);
referenceBz = original.Bz(:);

trainMask = selectSparseRegularGrid(xGrid, yGrid, sparseFraction);
trainXY = referenceXY(trainMask(:), :);
trainBx = referenceBx(trainMask(:));
trainBy = referenceBy(trainMask(:));
trainBz = referenceBz(trainMask(:));

normalizer = createXYNormalizer(referenceXY);
nearestDistance = medianNearestDistance(normalizeXY(trainXY, normalizer));
epsilonBase = 1 / max(nearestDistance, eps);
epsilonCandidates = epsilonBase * [0.50, 0.75, 1.00, 1.50, 2.00];

% 以原始网格上的 |B| 误差选择 epsilon。选中的同一个 epsilon 同时用于
% 三个分量，确保三个分量的重建尺度一致。
BmagReference = sqrt(referenceBx.^2 + referenceBy.^2 + referenceBz.^2);
epsilonTable = table(epsilonCandidates(:), nan(numel(epsilonCandidates), 1), ...
    nan(numel(epsilonCandidates), 1), nan(numel(epsilonCandidates), 1), ...
    'VariableNames', {'epsilon', 'RMSE_Bmag', 'MAE_Bmag', 'R2_Bmag'});

for k = 1:numel(epsilonCandidates)
    candidateBx = fitGaussianRBF(trainXY, trainBx, epsilonCandidates(k), normalizer);
    candidateBy = fitGaussianRBF(trainXY, trainBy, epsilonCandidates(k), normalizer);
    candidateBz = fitGaussianRBF(trainXY, trainBz, epsilonCandidates(k), normalizer);
    bx = evaluateGaussianRBF(candidateBx, referenceXY);
    by = evaluateGaussianRBF(candidateBy, referenceXY);
    bz = evaluateGaussianRBF(candidateBz, referenceXY);
    candidateBmag = sqrt(bx.^2 + by.^2 + bz.^2);
    metric = calculateMetrics(BmagReference, candidateBmag);
    epsilonTable.RMSE_Bmag(k) = metric.RMSE;
    epsilonTable.MAE_Bmag(k) = metric.MAE;
    epsilonTable.R2_Bmag(k) = metric.R2;
end
[~, bestIndex] = min(epsilonTable.RMSE_Bmag);
bestEpsilon = epsilonTable.epsilon(bestIndex);

fprintf('\nSparse training points: %d / %d\n', sum(trainMask(:)), numel(trainMask));
fprintf('RBF epsilon search:\n');
disp(epsilonTable);
fprintf('Selected epsilon: %.12g\n', bestEpsilon);

modelBx = fitGaussianRBF(trainXY, trainBx, bestEpsilon, normalizer);
modelBy = fitGaussianRBF(trainXY, trainBy, bestEpsilon, normalizer);
modelBz = fitGaussianRBF(trainXY, trainBz, bestEpsilon, normalizer);

%% 6. 计算高分辨率 RBF 图像及 RBF 解析导数
xQuery = linspace(min(xGrid), max(xGrid), imageResolution);
yQuery = linspace(min(yGrid), max(yGrid), imageResolution);
[XQuery, YQuery] = meshgrid(xQuery, yQuery);
queryXY = [XQuery(:), YQuery(:)];

[bxQuery, dBxDxQuery, dBxDyQuery] = evaluateGaussianRBF(modelBx, queryXY);
[byQuery, dByDxQuery, dByDyQuery] = evaluateGaussianRBF(modelBy, queryXY);
[bzQuery, dBzDxQuery, dBzDyQuery] = evaluateGaussianRBF(modelBz, queryXY);

BxRBF = reshape(bxQuery, size(XQuery));
ByRBF = reshape(byQuery, size(XQuery));
BzRBF = reshape(bzQuery, size(XQuery));
BmagRBF = sqrt(BxRBF.^2 + ByRBF.^2 + BzRBF.^2);
CtRBF = reshape(dBxDxQuery.^2 + dBxDyQuery.^2 + ...
    dByDxQuery.^2 + dByDyQuery.^2 + ...
    dBzDxQuery.^2 + dBzDyQuery.^2, size(XQuery));

% 误差图和指标必须使用相同坐标。此处在原始测量 x-y 网格上计算 RBF
% 预测值，再与原始数据逐点比较。
[bxRef, dBxDxRef, dBxDyRef] = evaluateGaussianRBF(modelBx, referenceXY);
[byRef, dByDxRef, dByDyRef] = evaluateGaussianRBF(modelBy, referenceXY);
[bzRef, dBzDxRef, dBzDyRef] = evaluateGaussianRBF(modelBz, referenceXY);
BmagRBFReference = sqrt(bxRef.^2 + byRef.^2 + bzRef.^2);
CtRBFReference = dBxDxRef.^2 + dBxDyRef.^2 + ...
    dByDxRef.^2 + dByDyRef.^2 + dBzDxRef.^2 + dBzDyRef.^2;

BmagRBFReferenceGrid = reshape(BmagRBFReference, size(BmagOriginal));
CtRBFReferenceGrid = reshape(CtRBFReference, size(CtOriginal));
BmagError = BmagRBFReferenceGrid - BmagOriginal;
CtError = CtRBFReferenceGrid - CtOriginal;
metricBmag = calculateMetrics(BmagOriginal(:), BmagRBFReference);
metricCt = calculateMetrics(CtOriginal(:), CtRBFReference);

%% 7. 输出峰值幅度与位置变化
bPeakOriginal = findImagePeak(BmagOriginal, xGrid, yGrid);
bPeakRBF = findImagePeak(BmagRBF, xQuery, yQuery);
ctPeakOriginal = findImagePeak(CtOriginal, xGrid, yGrid);
ctPeakRBF = findImagePeak(CtRBF, xQuery, yQuery);

fprintf('\n============================================================\n');
fprintf('Reconstruction comparison\n');
fprintf('|B| metrics on original grid: RMSE=%.8g, MAE=%.8g, R2=%.8g\n', ...
    metricBmag.RMSE, metricBmag.MAE, metricBmag.R2);
fprintf('Ct_2D metrics on original grid: RMSE=%.8g, MAE=%.8g, R2=%.8g\n', ...
    metricCt.RMSE, metricCt.MAE, metricCt.R2);
printPeakComparison('|B|', bPeakOriginal, bPeakRBF, data.coordinateUnit);
printPeakComparison('Ct_2D', ctPeakOriginal, ctPeakRBF, data.coordinateUnit);
fprintf('Ct_2D is derivative-based and is therefore usually more sensitive than |B|.\n');
fprintf('Use the signed error images to assess smoothing, peak shift and artifacts.\n');
fprintf('============================================================\n');

%% 8. 图 1：统一色标的 2×2 核心对比图
bmagLimits = finiteLimits([BmagOriginal(:); BmagRBF(:)]);
ctLimits = finiteLimits([CtOriginal(:); CtRBF(:)]);
fieldLabel = sprintf('|B| (%s)', data.fieldUnit);
ctLabel = sprintf('Ct_2D (%s^2/%s^2)', data.fieldUnit, data.coordinateUnit);

figCore = figure('Color', 'w', 'Position', [80, 80, 1300, 900], ...
    'Name', 'Original and RBF magnetic-field imaging');
layout = tiledlayout(figCore, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
colormap(figCore, parula(256));
plotImageTile(layout, xGrid, yGrid, BmagOriginal, ...
    'Original Magnetic Field Magnitude |B|', fieldLabel, bmagLimits);
plotImageTile(layout, xQuery, yQuery, BmagRBF, ...
    'RBF Reconstructed Magnetic Field Magnitude |B|', fieldLabel, bmagLimits);
plotImageTile(layout, xGrid, yGrid, CtOriginal, ...
    'Original 2D Magnetic Gradient Tensor Ct_2D', ctLabel, ctLimits);
plotImageTile(layout, xQuery, yQuery, CtRBF, ...
    'RBF Reconstructed 2D Magnetic Gradient Tensor Ct_2D', ctLabel, ctLimits);
title(layout, sprintf('z = %.12g %s; component-wise Gaussian RBF', ...
    zSlice, data.coordinateUnit));
saveFigureIfNeeded(figCore, resultFolder, '01_core_Bmag_Ct2D_comparison', ...
    saveFigures, figureResolution);

%% 9. 图 2 和图 3：以零为中心的有符号误差图
figBmagError = plotErrorImage(xGrid, yGrid, BmagError, ...
    '|B| Error: RBF - Original', sprintf('|B| error (%s)', data.fieldUnit));
saveFigureIfNeeded(figBmagError, resultFolder, '02_Bmag_error', ...
    saveFigures, figureResolution);

figCtError = plotErrorImage(xGrid, yGrid, CtError, ...
    'Ct_2D Error: RBF - Original', ...
    sprintf('Ct_2D error (%s^2/%s^2)', data.fieldUnit, data.coordinateUnit));
saveFigureIfNeeded(figCtError, resultFolder, '03_Ct2D_error', ...
    saveFigures, figureResolution);

%% 10. 保存数值结果与摘要文本
results = struct();
results.sourceFile = excelFile;
results.sheetName = sheetName;
results.headerRow = headerRow;
results.format = data.format;
results.zSlice = zSlice;
results.coordinateUnit = data.coordinateUnit;
results.fieldUnit = data.fieldUnit;
results.xOriginal = xGrid;
results.yOriginal = yGrid;
results.xRBF = xQuery;
results.yRBF = yQuery;
results.trainMask = trainMask;
results.bestEpsilon = bestEpsilon;
results.epsilonTable = epsilonTable;
results.BmagOriginal = BmagOriginal;
results.BmagRBF = BmagRBF;
results.BmagRBFReference = BmagRBFReferenceGrid;
results.BmagError = BmagError;
results.Ct2DOriginal = CtOriginal;
results.Ct2DRBF = CtRBF;
results.Ct2DRBFReference = CtRBFReferenceGrid;
results.Ct2DError = CtError;
results.metricBmag = metricBmag;
results.metricCt2D = metricCt;
results.peakBmagOriginal = bPeakOriginal;
results.peakBmagRBF = bPeakRBF;
results.peakCt2DOriginal = ctPeakOriginal;
results.peakCt2DRBF = ctPeakRBF;
save(fullfile(resultFolder, 'RBF_2D_imaging_results.mat'), 'results');
writeSummaryReport(fullfile(resultFolder, 'RBF_2D_imaging_summary.txt'), results);
fprintf('\nSaved figures and results to: %s\n', resultFolder);

%% 局部函数
function [headerRow, headers, dataRows] = findHeaderAndDataRows(raw)
% 将前部非空单元格最多的行视为表头。COMSOL 文件中表头行和数值行常常
% 并列最多，MATLAB 的 max 会返回较早出现的表头行。
    if isempty(raw)
        error('The selected worksheet is empty.');
    end
    countByRow = zeros(size(raw, 1), 1);
    for r = 1:size(raw, 1)
        countByRow(r) = sum(~cellfun(@isEmptyCellValue, raw(r, :)));
    end
    [largestCount, headerRow] = max(countByRow);
    if largestCount == 0
        error('No non-empty cells were found in the worksheet.');
    end
    headers = strings(1, size(raw, 2));
    for c = 1:size(raw, 2)
        if ~isEmptyCellValue(raw{headerRow, c})
            headers(c) = strtrim(string(raw{headerRow, c}));
        end
    end
    dataRows = [];
    for r = headerRow + 1:size(raw, 1)
        if any(~cellfun(@isEmptyCellValue, raw(r, :)))
            dataRows(end + 1) = r; %#ok<AGROW>
        end
    end
    if isempty(dataRows)
        error('No data rows were found below the detected header row.');
    end
end

function data = readMagneticData(raw, headers, dataRows)
% 输出统一的逐点向量结构，供后续成像步骤使用。
    data = struct('format', "unknown", 'x', [], 'y', [], 'z', [], ...
        'Bx', [], 'By', [], 'Bz', [], 'coordinateUnit', "as provided", ...
        'fieldUnit', "as provided");
    headerPresent = false(size(headers));
    for k = 1:numel(headers)
        headerPresent(k) = isNonemptyString(headers(k));
    end
    % COMSOL 横向导出将变量块沿列方向排列。只要检测到足够表头，就尝试
    % 横向格式解析，不再苛刻依赖“恰好一行非空数据”。
    if ~isempty(dataRows) && sum(headerPresent) >= 6
        data = tryReadComsolWide(raw, headers, dataRows(1), data);
    end
    if data.format == "unknown"
        data = tryReadComsolWideByLayout(raw, data);
    end
    if data.format == "unknown"
        data = readPointTable(raw, headers, dataRows, data);
    end
end

function data = tryReadComsolWideByLayout(raw, data)
% COMSOL 标准六个连续数值块的最终兜底解析器。某些非 ASCII 表头被 MATLAB
% 读取为缺失值时仍可使用。该路径限定为横向行（至少 12 个数值），避免将
% 普通六列逐点表误判为 COMSOL 横向格式。
    numericCountByRow = zeros(size(raw, 1), 1);
    numericMaskByRow = cell(size(raw, 1), 1);
    for r = 1:size(raw, 1)
        numericValues = cellToDouble(raw(r, :));
        numericMaskByRow{r} = isfinite(numericValues);
        numericCountByRow(r) = sum(numericMaskByRow{r});
    end
    [numericCount, dataRow] = max(numericCountByRow);
    if numericCount < 12 || mod(numericCount, 6) ~= 0
        return;
    end

    numericColumns = find(numericMaskByRow{dataRow});
    if any(diff(numericColumns) ~= 1)
        return;
    end
    blockLength = numericCount / 6;
    data.x = cellToDouble(raw(dataRow, numericColumns(1:blockLength)));
    data.y = cellToDouble(raw(dataRow, numericColumns(blockLength + 1:2 * blockLength)));
    data.z = cellToDouble(raw(dataRow, numericColumns(2 * blockLength + 1:3 * blockLength)));
    data.Bx = cellToDouble(raw(dataRow, numericColumns(3 * blockLength + 1:4 * blockLength)));
    data.By = cellToDouble(raw(dataRow, numericColumns(4 * blockLength + 1:5 * blockLength)));
    data.Bz = cellToDouble(raw(dataRow, numericColumns(5 * blockLength + 1:6 * blockLength)));
    data.format = "COMSOL-wide-layout-fallback";
    data.coordinateUnit = "mm";
    data.fieldUnit = "T";
    warning(['COMSOL headers could not be decoded. The widest numeric row was ' ...
        'mapped by block order: x, y, z, Bx, By, Bz.']);
end

function data = tryReadComsolWide(raw, headers, dataRow, data)
% COMSOL 横向导出将每个变量的所有采样值放在连续列块中。
    labels = lower(string(headers));
    xIndex = findWideCoordinates(labels, "x");
    yIndex = findWideCoordinates(labels, "y");
    zIndex = findWideCoordinates(labels, "z");
    [bxIndex, byIndex, bzIndex, fieldUnit] = findWideFields(labels);
    counts = [numel(xIndex), numel(yIndex), numel(zIndex), ...
        numel(bxIndex), numel(byIndex), numel(bzIndex)];
    if any(counts == 0) || numel(unique(counts)) ~= 1
        fprintf(['COMSOL-wide detection counts [x y z Bx By Bz] = ' ...
            '[%s]\n'], num2str(counts));
        return;
    end
    data.x = cellToDouble(raw(dataRow, xIndex));
    data.y = cellToDouble(raw(dataRow, yIndex));
    data.z = cellToDouble(raw(dataRow, zIndex));
    data.Bx = cellToDouble(raw(dataRow, bxIndex));
    data.By = cellToDouble(raw(dataRow, byIndex));
    data.Bz = cellToDouble(raw(dataRow, bzIndex));
    data.format = "COMSOL-wide";
    data.coordinateUnit = "mm";
    data.fieldUnit = fieldUnit;
end

function index = findWideCoordinates(labels, axisName)
% 坐标列以 x、y 或 z 开头，且列名中包含长度单位。
    index = [];
    for k = 1:numel(labels)
        label = labels(k);
        if ismissing(label)
            continue;
        end
        startsWithAxis = startsWith(strtrim(label), axisName);
        hasLengthUnit = contains(label, "(mm)") || ...
            contains(label, "(cm)") || contains(label, "(m)");
        if startsWithAxis && hasLengthUnit
            index(end + 1) = k; %#ok<AGROW>
        end
    end
end

function [bxIndex, byIndex, bzIndex, fieldUnit] = findWideFields(labels)
% 从带磁场单位的列名中提取最后一个 x/y/z 字符，避免依赖分量字母与单位
% 之间可能出现的中文或乱码文本。
    bxIndex = [];
    byIndex = [];
    bzIndex = [];
    fieldUnit = "T";
    for k = 1:numel(labels)
        label = labels(k);
        if ismissing(label)
            continue;
        end
        if ~isFieldUnitLabel(label)
            continue;
        end
        axisTokens = regexp(char(label), '[xyz]', 'match');
        if isempty(axisTokens)
            continue;
        end
        axisName = axisTokens{end};
        if axisName == 'x'
            bxIndex(end + 1) = k; %#ok<AGROW>
        elseif axisName == 'y'
            byIndex(end + 1) = k; %#ok<AGROW>
        else
            bzIndex(end + 1) = k; %#ok<AGROW>
        end
    end
    if isempty(bxIndex) || isempty(byIndex) || isempty(bzIndex)
        % 编码异常的 COMSOL 表头可能保留单位但丢失变量名。
        unitIndex = [];
        for k = 1:numel(labels)
            if ~ismissing(labels(k)) && isFieldUnitLabel(labels(k))
                unitIndex(end + 1) = k; %#ok<AGROW>
            end
        end
        if numel(unitIndex) >= 3 && mod(numel(unitIndex), 3) == 0
            blockLength = numel(unitIndex) / 3;
            bxIndex = unitIndex(1:blockLength);
            byIndex = unitIndex(blockLength + 1:2 * blockLength);
            bzIndex = unitIndex(2 * blockLength + 1:end);
            warning(['Wide-field axes could not be parsed. Consecutive unit blocks ' ...
                'were mapped to Bx, By and Bz; verify the printed headers.']);
        end
    end
    anyField = [bxIndex, byIndex, bzIndex];
    if ~isempty(anyField)
        fieldUnit = fieldUnitFromLabel(labels(anyField(1)));
    end
end

function tf = isFieldUnitLabel(label)
% COMSOL 磁场列通常以括号中的 T、nT、mT 或 uT 等单位标识。
    tf = contains(label, "(t)") || contains(label, "(nt)") || ...
        contains(label, "(mt)") || contains(label, "(ut)") || ...
        contains(label, "(μt)") || contains(label, "(µt)");
end

function unit = fieldUnitFromLabel(label)
    if contains(label, "(nt)")
        unit = "nT";
    elseif contains(label, "(mt)")
        unit = "mT";
    elseif contains(label, "(ut)") || contains(label, "(μt)") || contains(label, "(µt)")
        unit = "uT";
    else
        unit = "T";
    end
end

function data = readPointTable(raw, headers, dataRows, data)
% 常规表格格式：工作表的每一行对应一个坐标点及其磁场分量。
    xColumn = findPointColumn(headers, "coordinate", "x");
    yColumn = findPointColumn(headers, "coordinate", "y");
    zColumn = findPointColumn(headers, "coordinate", "z");
    bxColumn = findPointColumn(headers, "field", "x");
    byColumn = findPointColumn(headers, "field", "y");
    bzColumn = findPointColumn(headers, "field", "z");
    indices = [xColumn, yColumn, zColumn, bxColumn, byColumn, bzColumn];

    % 仅当表格是纯数值六列时，才允许按列顺序兜底映射。
    if any(indices == 0) && size(raw, 2) == 6
        numericColumns = false(1, 6);
        for c = 1:6
            numericColumns(c) = any(isfinite(cellToDouble(raw(dataRows, c))));
        end
        if all(numericColumns)
            warning(['Unrecognised six-column headers: using x, y, z, Bx, By, Bz ' ...
                'in that exact column order. Verify the workbook.']);
            indices = 1:6;
        end
    end
    if any(indices == 0)
        missing = ["x", "y", "z", "Bx", "By", "Bz"];
        error('Unable to identify point-table columns: %s.', ...
            strjoin(missing(indices == 0), ', '));
    end
    data.x = cellToDouble(raw(dataRows, indices(1)));
    data.y = cellToDouble(raw(dataRows, indices(2)));
    data.z = cellToDouble(raw(dataRows, indices(3)));
    data.Bx = cellToDouble(raw(dataRows, indices(4)));
    data.By = cellToDouble(raw(dataRows, indices(5)));
    data.Bz = cellToDouble(raw(dataRows, indices(6)));
    data.format = "point-table";
end

function index = findPointColumn(headers, kind, axisName)
% 防止单独的 x/y/z 坐标别名误匹配为 Bx/By/Bz 磁场列。
    index = 0;
    labels = lower(strtrim(string(headers)));
    normalized = normalizeLabels(labels);
    if kind == "coordinate"
        aliases = normalizeLabels([axisName, axisName + "mm", ...
            "coord" + axisName, "coordinate" + axisName, ...
            "position" + axisName, axisName + "coordinate"]);
        fieldTerms = ["field", "flux", "density", "magnetic", "tesla", ...
            "gauss", "磁场", "磁感应", "磁通"];
        for k = 1:numel(labels)
            if ismissing(labels(k))
                continue;
            end
            hasLengthUnit = any(contains(labels(k), ["mm", "cm", "meter", "metre"]));
            isField = any(contains(labels(k), fieldTerms));
            if (any(normalized(k) == aliases) || ...
                    (startsWith(normalized(k), axisName) && hasLengthUnit)) && ~isField
                index = k;
                return;
            end
        end
    else
        aliases = normalizeLabels(["b" + axisName, "b_" + axisName, ...
            "b " + axisName, "b(" + axisName + ")", ...
            "magneticfield" + axisName, "magneticfluxdensity" + axisName, ...
            "field" + axisName, "component" + axisName]);
        fieldTerms = ["field", "flux", "density", "magnetic", "tesla", ...
            "gauss", "component", "磁场", "磁感应", "磁通"];
        unitPattern = '[\(\[]\s*(t|nt|mt|ut|μt|µt|a/m|gauss)\s*[\)\]]';
        for k = 1:numel(labels)
            if ismissing(labels(k))
                continue;
            end
            hasAxis = contains(labels(k), axisName);
            hasFieldTerm = any(contains(labels(k), fieldTerms));
            hasFieldUnit = ~isempty(regexp(char(labels(k)), unitPattern, 'once'));
            if any(normalized(k) == aliases) || (hasAxis && (hasFieldTerm || hasFieldUnit))
                index = k;
                return;
            end
        end
    end
end

function labels = normalizeLabels(labels)
    labels = lower(strtrim(string(labels)));
    labels = regexprep(labels, '[\s_\-\(\)\[\]\{\}:,]+', '');
end

function data = subsetData(data, mask)
    data.x = data.x(mask);
    data.y = data.y(mask);
    data.z = data.z(mask);
    data.Bx = data.Bx(mask);
    data.By = data.By(mask);
    data.Bz = data.Bz(mask);
end

function [data, zSlice] = selectSingleZPlane(data)
% 只选择实际测得的 z 层，绝不虚构缺失的 z 方向信息。
    zLevels = unique(data.z(:), 'sorted');
    zSlice = zLevels(ceil(numel(zLevels) / 2));
    tolerance = max(1e-12, 1e-9 * max(1, max(abs(zLevels))));
    if numel(zLevels) > 1
        warning('Multiple z levels found. Using measured central plane z=%.12g %s.', ...
            zSlice, data.coordinateUnit);
    end
    data = subsetData(data, abs(data.z - zSlice) <= tolerance);
end

function grid = buildRegularGrid(data)
% 对重复采样点取均值，并要求最终 x-y 网格完整覆盖。
    grid.x = unique(data.x(:), 'sorted')';
    grid.y = unique(data.y(:), 'sorted')';
    if numel(grid.x) < 3 || numel(grid.y) < 3
        error('Ct_2D needs at least three unique x and y samples.');
    end
    [~, xIndex] = ismember(data.x, grid.x);
    [~, yIndex] = ismember(data.y, grid.y);
    gridSize = [numel(grid.y), numel(grid.x)];
    grid.Bx = accumarray([yIndex, xIndex], data.Bx, gridSize, @mean, NaN);
    grid.By = accumarray([yIndex, xIndex], data.By, gridSize, @mean, NaN);
    grid.Bz = accumarray([yIndex, xIndex], data.Bz, gridSize, @mean, NaN);
    if any(~isfinite([grid.Bx(:); grid.By(:); grid.Bz(:)]))
        error(['The selected z plane is not a complete regular x-y grid. ' ...
            'Use a scattered-grid derivative method or fill missing samples.']);
    end
end

function mask = selectSparseRegularGrid(xGrid, yGrid, fraction)
% 在整个测量范围内均匀选择 RBF 训练位置。
    total = numel(xGrid) * numel(yGrid);
    desired = min(total, max(9, round(total * fraction)));
    stride = max(1, ceil(sqrt(total / desired)));
    mask = false(numel(yGrid), numel(xGrid));
    mask(1:stride:end, 1:stride:end) = true;
    if sum(mask(:)) < 9
        mask(:, :) = true;
    end
end

function normalizer = createXYNormalizer(xy)
    normalizer.offset = min(xy, [], 1);
    normalizer.scale = max(xy, [], 1) - normalizer.offset;
    normalizer.scale(normalizer.scale == 0) = 1;
end

function points = normalizeXY(xy, normalizer)
    points = (double(xy) - normalizer.offset) ./ normalizer.scale;
end

function d = medianNearestDistance(points)
    distances = squaredDistanceMatrix(points, points);
    count = size(points, 1);
    distances(1:count + 1:end) = inf;
    nearest = sqrt(min(distances, [], 2));
    nearest = nearest(isfinite(nearest) & nearest > 0);
    if isempty(nearest)
        d = 1;
    else
        d = median(nearest);
    end
end

function model = fitGaussianRBF(trainXY, values, epsilon, normalizer)
% 在归一化坐标中拟合单个分量，并加入极小正则化以改善数值稳定性。
    points = normalizeXY(trainXY, normalizer);
    phi = exp(-(epsilon^2) * squaredDistanceMatrix(points, points));
    regularization = max(1e-12, 1e-10 * norm(phi, inf));
    model.points = points;
    model.weights = (phi + regularization * eye(size(phi))) \ double(values(:));
    model.epsilon = epsilon;
    model.normalizer = normalizer;
end

function [values, dfdx, dfdy] = evaluateGaussianRBF(model, queryXY)
% 在物理坐标单位下解析计算 f、df/dx 和 df/dy。
    query = normalizeXY(queryXY, model.normalizer);
    count = size(query, 1);
    values = zeros(count, 1);
    dfdx = zeros(count, 1);
    dfdy = zeros(count, 1);
    batchSize = 20000;
    for first = 1:batchSize:count
        last = min(first + batchSize - 1, count);
        points = query(first:last, :);
        dx = points(:, 1) - model.points(:, 1)';
        dy = points(:, 2) - model.points(:, 2)';
        phi = exp(-(model.epsilon^2) * (dx.^2 + dy.^2));
        values(first:last) = phi * model.weights;
        dfdx(first:last) = ((-2 * model.epsilon^2 * dx .* phi) * ...
            model.weights) / model.normalizer.scale(1);
        dfdy(first:last) = ((-2 * model.epsilon^2 * dy .* phi) * ...
            model.weights) / model.normalizer.scale(2);
    end
end

function d2 = squaredDistanceMatrix(a, b)
    d2 = sum(a.^2, 2) + sum(b.^2, 2)' - 2 * (a * b');
    d2 = max(d2, 0);
end

function metric = calculateMetrics(trueValues, predictedValues)
    trueValues = double(trueValues(:));
    predictedValues = double(predictedValues(:));
    valid = isfinite(trueValues) & isfinite(predictedValues);
    trueValues = trueValues(valid);
    predictedValues = predictedValues(valid);
    residual = predictedValues - trueValues;
    metric.RMSE = sqrt(mean(residual.^2));
    metric.MAE = mean(abs(residual));
    total = sum((trueValues - mean(trueValues)).^2);
    if total <= eps
        metric.R2 = NaN;
    else
        metric.R2 = 1 - sum(residual.^2) / total;
    end
end

function peak = findImagePeak(imageData, xValues, yValues)
    [peak.value, linearIndex] = max(imageData(:));
    [row, column] = ind2sub(size(imageData), linearIndex);
    peak.x = xValues(column);
    peak.y = yValues(row);
end

function printPeakComparison(name, originalPeak, rbfPeak, coordinateUnit)
    change = 100 * (rbfPeak.value - originalPeak.value) / max(abs(originalPeak.value), eps);
    shift = hypot(rbfPeak.x - originalPeak.x, rbfPeak.y - originalPeak.y);
    fprintf('%s original peak: %.8g at (%.8g, %.8g) %s\n', ...
        name, originalPeak.value, originalPeak.x, originalPeak.y, coordinateUnit);
    fprintf('%s RBF peak:      %.8g at (%.8g, %.8g) %s\n', ...
        name, rbfPeak.value, rbfPeak.x, rbfPeak.y, coordinateUnit);
    fprintf('%s peak change:   %.4f%%; position shift: %.8g %s\n', ...
        name, change, shift, coordinateUnit);
end

function limits = finiteLimits(values)
    values = values(isfinite(values));
    limits = [min(values), max(values)];
    if limits(1) == limits(2)
        delta = max(abs(limits(1)) * 0.01, 1);
        limits = limits + [-delta, delta];
    end
end

function plotImageTile(layout, xValues, yValues, imageData, titleText, label, limits)
    nexttile(layout);
    imagesc(xValues, yValues, imageData);
    set(gca, 'YDir', 'normal');
    axis image;
    xlim([min(xValues), max(xValues)]);
    ylim([min(yValues), max(yValues)]);
    caxis(limits);
    xlabel('x');
    ylabel('y');
    title(titleText, 'Interpreter', 'none');
    colorbarHandle = colorbar;
    colorbarHandle.Label.String = label;
end

function fig = plotErrorImage(xValues, yValues, imageData, titleText, label)
    fig = figure('Color', 'w', 'Position', [180, 180, 760, 620], ...
        'Name', titleText);
    imagesc(xValues, yValues, imageData);
    set(gca, 'YDir', 'normal');
    axis image;
    maxMagnitude = max(abs(imageData(isfinite(imageData))));
    if isempty(maxMagnitude) || maxMagnitude == 0
        maxMagnitude = 1;
    end
    caxis([-maxMagnitude, maxMagnitude]);
    colormap(fig, blueWhiteRed(256));
    xlabel('x');
    ylabel('y');
    title(titleText, 'Interpreter', 'none');
    colorbarHandle = colorbar;
    colorbarHandle.Label.String = label;
end

function map = blueWhiteRed(count)
    halfCount = ceil(count / 2);
    blue = [0.10, 0.28, 0.75];
    white = [1.00, 1.00, 1.00];
    red = [0.78, 0.12, 0.12];
    map = [interp1([0; 1], [blue; white], linspace(0, 1, halfCount)'); ...
           interp1([0; 1], [white; red], linspace(0, 1, count - halfCount)')];
end

function saveFigureIfNeeded(fig, folder, stem, shouldSave, resolution)
    if shouldSave
        exportgraphics(fig, fullfile(folder, [stem '.png']), ...
            'Resolution', resolution);
    end
end

function writeSummaryReport(filePath, results)
    fileId = fopen(filePath, 'w');
    if fileId < 0
        warning('Could not write summary report: %s', filePath);
        return;
    end
    cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
    fprintf(fileId, '2D component-wise Gaussian RBF magnetic-field imaging\n');
    fprintf(fileId, 'Source: %s\n', results.sourceFile);
    fprintf(fileId, 'Plane z: %.12g %s\n', results.zSlice, results.coordinateUnit);
    fprintf(fileId, 'Best epsilon: %.12g\n\n', results.bestEpsilon);
    fprintf(fileId, '|B| RMSE=%.12g, MAE=%.12g, R2=%.12g\n', ...
        results.metricBmag.RMSE, results.metricBmag.MAE, results.metricBmag.R2);
    fprintf(fileId, 'Ct_2D RMSE=%.12g, MAE=%.12g, R2=%.12g\n', ...
        results.metricCt2D.RMSE, results.metricCt2D.MAE, results.metricCt2D.R2);
end

function values = cellToDouble(cellValues)
    values = nan(numel(cellValues), 1);
    for k = 1:numel(cellValues)
        value = cellValues{k};
        if isnumeric(value) && isscalar(value)
            values(k) = double(value);
        elseif islogical(value) && isscalar(value)
            values(k) = double(value);
        elseif ischar(value) || (isstring(value) && isscalar(value))
            values(k) = str2double(strtrim(string(value)));
        end
    end
end

function tf = isEmptyCellValue(value)
    if isempty(value)
        tf = true;
    elseif isstring(value)
        tf = ismissing(value) || strlength(strtrim(value)) == 0;
    elseif ischar(value)
        tf = isempty(strtrim(value));
    else
        tf = false;
    end
end

function tf = isNonemptyString(value)
    tf = ~ismissing(value) && strlength(strtrim(value)) > 0;
end

function name = excelColumnName(index)
    name = '';
    while index > 0
        remainder = mod(index - 1, 26);
        name = [char('A' + remainder), name]; %#ok<AGROW>
        index = floor((index - 1) / 26);
    end
end
