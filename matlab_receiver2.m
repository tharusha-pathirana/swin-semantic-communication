% Set argument values
receivedFile = 'combined_binary.bin';
imagePath = './Datasets/Clic2021/06.png';      % Leave '' if you don’t want to save output
%useCodebook = true;   % true / false

useCodebook = '';   % 'true' or 'false' or leave empty '' (Auto detect)
k = 512;             % Optional, set to [] if not used
chunk = 4;           % Optional, set to [] if not used
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

% Display if exists
% if exist(localReconImage, 'file')
%     warning('off', 'all');  % Disable MATLAB image size warnings
%     figure; imshow(imread(localReconImage));
%     title('Reconstructed Image');
% else
%     warning('Reconstructed image file not found.');
% end
