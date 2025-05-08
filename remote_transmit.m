
remoteUser = 'nirushtihan';
remoteHost = '192.248.10.65';
remotePython = '/home/nirushtihan/Tharusha/virtual_env/py310/bin/python3';
remoteScriptDir = '/home/nirushtihan/Tharusha/SwinJSCC';
remoteScript = 'transmitter.py';

%imagePath = 'Datasets/Kodak/kodim23.png';
useCodebook = true;
k = 512;         % Set to [] if not needed (Default 512)
chunk = 4;       % Set to [] if not needed (Default 4)
adaptive = '';   % 'True', 'False', or '' (auto mode)

% Local image path (on your PC)
localImagePath = 'Datasets/Kodak/kodim23.png';
remoteImagePath = fullfile(remoteScriptDir, 'testing_image.png');

% Upload image to server first
scpUploadCmd = sprintf('scp %s %s@%s:%s', imagePath, remoteUser, remoteHost, remoteImagePath);
[scpUploadStatus, scpUploadOutput] = system(scpUploadCmd);
disp(scpUploadOutput);

% Build argument string

args = sprintf('--image_path "%s"', remoteImagePath);
if ~isempty(k), args = sprintf('%s --k %d', args, k); end
if ~isempty(chunk), args = sprintf('%s --chunk_size %d', args, chunk); end
if useCodebook, args = sprintf('%s --use_codebook', args); end
if ~isempty(adaptive) && any(strcmpi(adaptive, {'true','false'}))
    args = sprintf('%s --adaptive %s', args, adaptive);
end


% Run script remotely
cmd = sprintf('ssh %s@%s "cd %s && %s %s %s"', ...
    remoteUser, remoteHost, remoteScriptDir, remotePython, remoteScript, args);
[status, output] = system(cmd);
disp(output);

% Copy binary file back
remoteBin = fullfile(remoteScriptDir, 'combined_binary.bin');
localBin = 'combined_binary.bin';
scpCmd = sprintf('scp %s@%s:%s %s', remoteUser, remoteHost, remoteBin, localBin);
[scpStatus, scpOutput] = system(scpCmd);
disp(scpOutput);
