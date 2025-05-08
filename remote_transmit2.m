
remoteUser = 'nirushtihan';
remoteHost = '192.248.10.65';
remotePython = '/home/nirushtihan/Tharusha/virtual_env/py310/bin/python3';
remoteScriptDir = '/home/nirushtihan/Tharusha/SwinJSCC';
remoteScript = 'transmitter2.py';

%imagePath = 'Datasets/Kodak/kodim23.png';
useCodebook = false;
k = 512;         % Set to [] if not needed (Default 512)
chunk = 4;       % Set to [] if not needed (Default 4)
adaptive = '';   % 'true', 'false', or '' (auto mode)

patch_size = [];      % 28 or 60, or [] if not used
depth = [];           % 5 or 6, or [] if not used

% Local image path (on your PC)
localImagePath = './Datasets/Clic2021/06.png';  %Datasets/Kodak/kodim23.png

%remoteImagePath = fullfile(remoteScriptDir, 'testing_image.png');
[~, imageName, ext] = fileparts(localImagePath);
%remoteImagePath = fullfile(remoteScriptDir, [imageName, ext]);
remoteImagePath = [remoteScriptDir, '/', imageName, ext];


% Upload image to server first
scpUploadCmd = sprintf('scp %s %s@%s:%s', localImagePath, remoteUser, remoteHost, remoteImagePath);
[scpUploadStatus, scpUploadOutput] = system(scpUploadCmd);
disp(scpUploadOutput);

% Build argument string

args = sprintf('--image_path "%s"', remoteImagePath);
if ~isempty(k), args = sprintf('%s --k %d', args, k); end
if ~isempty(chunk), args = sprintf('%s --chunk_size %d', args, chunk); end
if useCodebook, args = sprintf('%s --use_codebook', args); end
if ~isempty(adaptive) && any(strcmpi(adaptive, {'true','false'}))
    args = sprintf('%s --adaptive %s', args, adaptive); end
if ~isempty(patch_size), args = sprintf('%s --patch_size %d', args, patch_size); end  
if ~isempty(depth), args = sprintf('%s --depth %d', args, depth); end               


% Run script remotely
cmd = sprintf('ssh %s@%s "cd %s && %s %s %s"', ...
    remoteUser, remoteHost, remoteScriptDir, remotePython, remoteScript, args);
[status, output] = system(cmd);
disp(output);

if contains(output, 'Adaptive Patch enabled : True'), adaptiveLabel = 'adaptive'; 
else, adaptiveLabel = 'non_adaptive'; end

if isempty(k), k = 512; end
if isempty(chunk), chunk = 4; end

% Build the label
label = adaptiveLabel;
if useCodebook
    label = sprintf('%s_%dd_%dk', label, chunk, k);
end

combinedBinaryName = fullfile('Binary', 'Transmitted_Binary', sprintf('%s_combined_binary_%s.bin', imageName, label));


% Copy binary file back
remoteBin = fullfile(remoteScriptDir, 'combined_binary.bin');
localBin = 'combined_binary.bin';
scpCmd = sprintf('scp %s@%s:%s %s', remoteUser, remoteHost, remoteBin, localBin);
[scpStatus, scpOutput] = system(scpCmd);
disp(scpOutput);

copyfile(localBin, combinedBinaryName);

if ~strcmpi(adaptive, 'false')
    % Build remote and local paths
    remotePatch = [remoteScriptDir, '/', 'patch_boundaries.png'];
    localPatch = 'patch_boundaries.png';
    
    % Copy from remote server to local machine
    scpPatchCmd = sprintf('scp %s@%s:%s %s', remoteUser, remoteHost, remotePatch, localPatch);
    [scpStatus, scpOutput] = system(scpPatchCmd);
    disp(scpOutput);  % Optional: show scp output
    
    % Display in MATLAB only if scp succeeded
    if scpStatus == 0 && exist(localPatch, 'file')
        img = imread(localPatch);
        figure; imshow(img);
        title('Patch Boundaries');
    end
end
