remoteUser = 'nirushtihan';
remoteHost = '192.248.10.65';

remotePython    = '/home/nirushtihan/Tharusha/virtual_env/py310/bin/python3';
remoteScriptDir = '/home/nirushtihan/Tharusha/SwinJSCC';
remoteScript    = 'receiver2.py';

receivedLocalFile = 'combined_binary.bin';
imagePathLocal    = '';  % Set '' if no original image  Datasets/Kodak/kodim20.png

useCodebook       = ''; % 'true' or 'false' or leave empty '' 
k           = [];    % Set to [] if not needed
chunk       = [];      % Set to [] if not needed
adaptive    = '';     % Options: 'true', 'false', or ''
patch_size  = [];     % 28 or 60, or [] if not used


[~, imageName, ext] = fileparts(imagePathLocal);
remoteReceivedFile = [remoteScriptDir, '/', receivedLocalFile];
remoteImagePath = [remoteScriptDir, '/', imageName, ext];


% Upload files to server
system(sprintf('scp "%s" %s@%s:"%s"', receivedLocalFile, remoteUser, remoteHost, remoteReceivedFile));
if ~isempty(imagePathLocal)
    system(sprintf('scp "%s" %s@%s:"%s"', imagePathLocal, remoteUser, remoteHost, remoteImagePath));
end

% Build argument string
args = sprintf('--received_file "%s"', remoteReceivedFile);
if ~isempty(imagePathLocal), args = sprintf('%s --image_path "%s"', args, remoteImagePath); end
if ~isempty(useCodebook) && any(strcmpi(useCodebook, {'true', 'false'}))
    args = sprintf('%s --use_codebook %s', args, lower(useCodebook)); end
if ~isempty(k), args = sprintf('%s --k %d', args, k); end
if ~isempty(chunk), args = sprintf('%s --chunk_size %d', args, chunk); end
if ~isempty(adaptive) && any(strcmpi(adaptive, {'true','false'}))
    args = sprintf('%s --adaptive %s', args, lower(adaptive)); end
if ~isempty(patch_size), args = sprintf('%s --patch_size %d', args, patch_size); end

disp("Running the script");

cmd = sprintf('ssh %s@%s "cd %s && %s %s %s"', ...
    remoteUser, remoteHost, remoteScriptDir, remotePython, remoteScript, args);
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



% if isempty(imagePathLocal), remoteImageName = 'default_image.png';
% else, remoteImageName = [imageName, ext]; end
% 
% % Build remote reconstructed image path
% if codebookDetected
%     remoteReconImage = sprintf('%s/recon/%dd_%dk/adaptive=%s/reconstructed_%dd_%dk_%s', ...
%         remoteScriptDir, chunk, k, adaptiveLabel, chunk, k, remoteImageName);
% else
%     remoteReconImage = sprintf('%s/recon/without_codebook/adaptive=%s/reconstructed_%s', ...
%         remoteScriptDir, adaptiveLabel, remoteImageName);
% end
% 
% 
% %Prepare local directory structure
% reconIdx = strfind(remoteReconImage, '/recon/');
% remoteReconRelative = remoteReconImage(reconIdx + 1 : end);
% localReconImage = remoteReconRelative;




% Extract reconstructed image path from Python output
reconLineIdx = contains(splitlines(output), 'Reconstructed image saved at:');
reconLines = splitlines(output);
reconLine = reconLines(reconLineIdx);

if ~isempty(reconLine)
    relativeReconImage = strtrim(erase(reconLine{1}, 'Reconstructed image saved at:'));
    remoteReconImage = fullfile(remoteScriptDir, relativeReconImage);  % Make full absolute path for scp
else
    error('Failed to find reconstructed image path from Python output.');
end


remoteReconImage = strrep(remoteReconImage, '\', '/');  % normalize slashes
reconIdx = strfind(remoteReconImage, '/recon/');
if isempty(reconIdx)
    error('Could not find /recon/ in remote path.');
end
remoteReconRelative = remoteReconImage(reconIdx:end);  % includes '/recon/...'
localReconImage = fullfile('.', remoteReconRelative);  % Save locally in ./recon/...






localReconDir = fileparts(localReconImage);
if ~exist(localReconDir, 'dir')
    mkdir(localReconDir);
end

% Download the reconstructed image
disp("Downloading reconstructed image...");
scpCmd = sprintf('scp %s@%s:"%s" "%s"', remoteUser, remoteHost, remoteReconImage, localReconImage);
system(scpCmd);

% Display if downloaded successfully
if exist(localReconImage, 'file')
    warning('off', 'all');  % Disable MATLAB image size warnings
    figure; imshow(imread(localReconImage));
    title('Reconstructed Image');
end








if isempty(imagePathLocal)
    imageName = 'default_image'; ext = '.png';
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
if ~isempty(imagePathLocal)
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

excelReconPath = strrep(localReconImage, '.\', '');
excelReconPath = strrep(excelReconPath, '\', '/');


% Determine if codebook used
codebookUsed = 'True';
if ~codebookDetected
    codebookUsed = 'False';
    k = '';
    chunk = '';
end

% Prepare row
row = {
    imageName, excelReconPath, ...
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
