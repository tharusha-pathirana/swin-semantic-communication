% Set argument values
receivedFile = 'combined_binary.bin';
imagePath = 'Datasets/Kodak/kodim23.png';      % Leave '' if you don’t want to save output


useCodebook = '';   % 'true' or 'false' or leave empty '' (Auto detect)
k = [];             % Optional, set to [] if not used
chunk = [];           % Optional, set to [] if not used
adaptive = '';   % 'true' or 'false' or leave empty ''
resH = [];           % Optional
resW = [];           % Optional
patch_size = [];     % 28 or 60, or [] if not used

pythonExe = '"C:\Python311\cv\Scripts\python.exe"';
script = 'receiver2.py';


% Build argument string
cmd = sprintf('%s %s --received_file "%s"', pythonExe, script, receivedFile);
if ~isempty(imagePath), cmd = sprintf('%s --image_path "%s"', cmd, imagePath); end
if ~isempty(useCodebook) && any(strcmpi(useCodebook, {'true', 'false'}))
    cmd = sprintf('%s --use_codebook %s', cmd, lower(useCodebook)); end
if ~isempty(k), cmd = sprintf('%s --k %d', cmd, k); end
if ~isempty(chunk), cmd = sprintf('%s --chunk_size %d', cmd, chunk); end
if ~isempty(adaptive) && any(strcmpi(adaptive, {'true','false'}))
    cmd = sprintf('%s --adaptive %s', cmd, lower(adaptive)); end
if ~isempty(patch_size), cmd = sprintf('%s --patch_size %d', cmd, patch_size); end


[~, output] = system(cmd);
disp(output);

% Parse Python output to determine actual used parameters
adaptiveLabel = 'True'; 
if contains(output, 'Adaptive Patching Enabled: False'), adaptiveLabel = 'False'; end
codebookDetected = contains(output, 'Codebook Enabled: True');

chunkMatch = regexp(output, 'Chunk Size:\s*(\d+)', 'tokens');
if ~isempty(chunkMatch), chunk = str2double(chunkMatch{1}{1}); else, chunk = 4; end
kMatch = regexp(output, 'Codebook k Size:\s*(\d+)', 'tokens');
if ~isempty(kMatch), k = str2double(kMatch{1}{1}); else, k = 512; end

if isempty(imagePath)
    reconImageName = 'default_image.png';
    imageName = 'default_image'; ext = '.png';
else
    [~, imageName, ext] = fileparts(imagePath);
    reconImageName = [imageName, ext];
end

% Build reconstructed image path
if codebookDetected
    localReconImage = sprintf('recon/%dd_%dk/adaptive=%s/reconstructed_%dd_%dk_%s', ...
        chunk, k, adaptiveLabel, chunk, k, reconImageName);
else
    localReconImage = sprintf('recon/without_codebook/adaptive=%s/reconstructed_%s', ...
        adaptiveLabel, reconImageName);
end





% Extract resolution
resMatch = regexp(output, 'Resolution read from file:\s*(\d+)x(\d+)', 'tokens');
if ~isempty(resMatch)
    resH = str2double(resMatch{1}{1});
    resW = str2double(resMatch{1}{2});
    resolutionStr = sprintf('%dx%d', resH, resW);
else
    resolutionStr = '';
end

% Extract metrics only if original image is known
if ~isempty(imagePath)
    psnrVal = extract_metric(output, 'PSNR:\s*([\d.]+)');
    msssimVal = extract_metric(output, 'MS-SSIM:\s*([\d.]+)');
    lpipsVal = extract_metric(output, 'LPIPS:\s*([\d.]+)');
    inputSize = extract_metric(output, 'Input Image Size:\s*([\d.]+)');
    outputSize = extract_metric(output, 'Output Image Size:\s*([\d.]+)');
    compression = extract_metric(output, 'Compression Ratio:\s*([\d.]+)');
    binSize = extract_metric(output, 'Reconstructed Binary Size:\s*([\d.]+)');
else
    % Estimate binary size
    if codebookDetected
        d = chunk;  % chunk size (e.g., 4)
        num_elements = (resH * resW * 32) / (256 * d);
        bytes_per_element = 2 * (k > 256) + 1 * (k <= 256);
        binSize = (num_elements * bytes_per_element) / 1024;  % in KB
    else
        binSize = (resH * resW * 32) / 256 / 1024;  % in KB (float vector, no quantization)
    end

    psnrVal = ''; msssimVal = ''; lpipsVal = '';
    inputSize = ''; outputSize = ''; compression = '';
end





% Determine if codebook used
codebookUsed = 'True';
if ~codebookDetected
    codebookUsed = 'False';
    k = '';
    chunk = '';
end

% Prepare row
row = {
    imageName, localReconImage, ...
    codebookUsed, chunk, k, adaptiveLabel, ...
    psnrVal, msssimVal, lpipsVal, ...
    inputSize, outputSize, binSize, compression
};

headers = {
    'ImagePath', 'ReconPath', ...
    'CodebookUsed', 'd', 'k', 'Adaptive', ...
    'PSNR', 'MS-SSIM', 'LPIPS', ...
    'InputSizeKB', 'OutputSizeKB', 'BinarySizeKB', 'CompressionRatio'
};

resultsFile = 'results_summary.xlsx';

% Load Excel and check for existing path
if exist(resultsFile, 'file')
    [~, ~, raw] = xlsread(resultsFile);
    headers_existing = raw(1, :);
    data = raw(2:end, :);

    reconCol = find(strcmpi(headers_existing, 'ReconPath'));
    existingRow = find(strcmpi(localReconImage, data(:, reconCol)));

    if ~isempty(existingRow)
        rowIndex = existingRow + 1;
    else
        rowIndex = size(data, 1) + 2;
    end
else
    % Write headers if new file
    xlswrite(resultsFile, headers, 1, 'A1');
    rowIndex = 2;
end

% Write the row
cellRange = sprintf('A%d', rowIndex);
xlswrite(resultsFile, row, 1, cellRange);

% Helper function (keep at the bottom of script)
function val = extract_metric(txt, pattern)
    match = regexp(txt, pattern, 'tokens', 'once');
    if ~isempty(match)
        val = str2double(match{1});
    else
        val = '';
    end
end
