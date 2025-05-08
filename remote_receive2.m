remoteUser = 'nirushtihan';
remoteHost = '192.248.10.65';

remotePython    = '/home/nirushtihan/Tharusha/virtual_env/py310/bin/python3';
remoteScriptDir = '/home/nirushtihan/Tharusha/SwinJSCC';
remoteScript    = 'receiver2.py';

receivedLocalFile = 'combined_binary.bin';
imagePathLocal    = './Datasets/Clic2021/06.png';  % Set '' if no original image  Datasets/Kodak/kodim20.png

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


if isempty(imagePathLocal), remoteImageName = 'default_image.png';
else, remoteImageName = [imageName, ext]; end

% Build remote reconstructed image path
if codebookDetected
    remoteReconImage = sprintf('%s/recon/%dd_%dk/adaptive=%s/reconstructed_%dd_%dk_%s', ...
        remoteScriptDir, chunk, k, adaptiveLabel, chunk, k, remoteImageName);
else
    remoteReconImage = sprintf('%s/recon/without_codebook/adaptive=%s/reconstructed_%s', ...
        remoteScriptDir, adaptiveLabel, remoteImageName);
end



% Prepare local directory structure
reconIdx = strfind(remoteReconImage, '/recon/');
remoteReconRelative = remoteReconImage(reconIdx + 1 : end);
localReconImage = remoteReconRelative;

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