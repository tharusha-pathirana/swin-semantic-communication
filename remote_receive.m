remoteUser = 'nirushtihan';
remoteHost = '192.248.10.65';
remotePython = '/home/nirushtihan/Tharusha/virtual_env/py310/bin/python3';
remoteScriptDir = '/home/nirushtihan/Tharusha/SwinJSCC';
remoteScript = 'receiver.py';

% Local input file
localReceivedFile = 'combined_binary.bin';
remoteReceivedFile = fullfile(remoteScriptDir, 'combined_binary.bin');

% Step 1: Send combined_binary.bin to server
scpUploadCmd = sprintf('scp %s %s@%s:%s', localReceivedFile, remoteUser, remoteHost, remoteReceivedFile);
[scpUploadStatus, scpUploadOutput] = system(scpUploadCmd);
disp(scpUploadOutput);

% Step 2: Set up arguments and run remote script
imagePath = 'Datasets/Kodak/kodim15.png';
useCodebook = true;
k = 512;
chunk = 4;
resH = [];
resW = [];
adaptive = '';

% Build argument string
args = sprintf('--received_file "%s"', 'combined_binary.bin');  % file is now on remote side
if ~isempty(imagePath), args = sprintf('%s --image_path "%s"', args, imagePath); end
if ~isempty(k), args = sprintf('%s --k %d', args, k); end
if ~isempty(chunk), args = sprintf('%s --chunk_size %d', args, chunk); end
if useCodebook, args = sprintf('%s --use_codebook', args); end
if ~isempty(resH) && ~isempty(resW), args = sprintf('%s --res_h %d --res_w %d', args, resH, resW); end
if ~isempty(adaptive), args = sprintf('%s --adaptive %s', args, adaptive); end

% Run remote script
cmd = sprintf('ssh %s@%s "cd %s && %s %s %s"', ...
    remoteUser, remoteHost, remoteScriptDir, remotePython, remoteScript, args);
[status, output] = system(cmd);
disp(output);

% Step 3: Copy output images back to PC
filesToFetch = {
    fullfile(remoteScriptDir, 'reconstructed_plot.png'), ...
    fullfile(remoteScriptDir, 'recon/reconstructed_image.png')
};
localFiles = {'reconstructed_plot_local.png', 'reconstructed_image_local.png'};

for i = 1:length(filesToFetch)
    scpCmd = sprintf('scp %s@%s:%s %s', remoteUser, remoteHost, filesToFetch{i}, localFiles{i});
    [scpStatus, scpOutput] = system(scpCmd);
    disp(scpOutput);
end

% Step 4: Display the plot
if exist('reconstructed_plot_local.png', 'file')
    figure;
    imshow(imread('reconstructed_plot_local.png'));
    title('Reconstructed Plot');
end
