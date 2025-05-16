% --- Setup ---
remoteUser = 'nirushtihan';
remoteHost = '192.248.10.65';
pythonExe = '/home/nirushtihan/Tharusha/virtual_env/py310/bin/python3';
remoteScriptDir = '/home/nirushtihan/Tharusha/SwinJSCC';
remoteScript = 'sim.py';


% --- Define arguments ---
type = 'both';  % 'tx', 'rx', or 'both'
received_file = './Binary/simulated.bin';
imagePath = 'Datasets/Kodak/kodim23.png';

useCodebook = true;
k = 512;
chunk = 4;
adaptive = '';  % 'true', 'false', or '' (auto mode)


patch_size = 28;
low_th = 100;
high_th = 200;
v_val = 100;
kernel = 1;
depth = [];  % or 4/5/6/7 or [] if not set


[~, imageName, ext] = fileparts(imagePath);
remoteImagePath = [remoteScriptDir, '/', imageName, ext];
remoteReceivedFile = [remoteScriptDir, '/Binary/simulated.bin'];

% --- Upload files ---
if strcmp(type, 'tx') || strcmp(type, 'both')
    system(sprintf('scp "%s" %s@%s:"%s"', imagePath, remoteUser, remoteHost, remoteImagePath));
end
if strcmp(type, 'rx') 
    system(sprintf('scp "%s" %s@%s:"%s"', received_file, remoteUser, remoteHost, remoteReceivedFile));
    if ~isempty(imagePath)
        system(sprintf('scp "%s" %s@%s:"%s"', imagePath, remoteUser, remoteHost, remoteImagePath));
    end
end

% --- Build argument string ---
args = sprintf('--type %s --received_file "%s"', type, remoteReceivedFile);
if ~isempty(imagePath), args = [args, sprintf(' --image_path "%s"', remoteImagePath)]; end
if useCodebook, args = [args, ' --use_codebook']; end
if ~isempty(k), args = [args, sprintf(' --k %d', k)]; end
if ~isempty(chunk), args = [args, sprintf(' --chunk_size %d', chunk)]; end
if ~isempty(adaptive), args = [args, sprintf(' --adaptive %s', adaptive)]; end
if ~isempty(patch_size), args = [args, sprintf(' --patch_size %d', patch_size)]; end
if ~isempty(noise), args = [args, sprintf(' --noise %.1f', noise)]; end
if ~isempty(low_th), args = [args, sprintf(' --low_th %d', low_th)]; end
if ~isempty(high_th), args = [args, sprintf(' --high_th %d', high_th)]; end
if ~isempty(v_val), args = [args, sprintf(' --v_val %d', v_val)]; end
if ~isempty(kernel), args = [args, sprintf(' --kernel %d', kernel)]; end
if ~isempty(depth), args = [args, sprintf(' --depth %d', depth)]; end

% --- Run remotely ---
cmd = sprintf('ssh %s@%s "cd %s && %s %s %s"', ...
    remoteUser, remoteHost, remoteScriptDir, pythonExe, remoteScript, args);
[~, output] = system(cmd);
disp(output);

% --- Copy outputs ---
if strcmp(type, 'tx')
    system(sprintf('scp %s@%s:%s/patch_boundaries.png ./patch_boundaries.png', ...
        remoteUser, remoteHost, remoteScriptDir));
    system(sprintf('scp %s@%s:%s/Binary/simulated.bin ./Binary/simulated.bin', ...
        remoteUser, remoteHost, remoteScriptDir));
    patchImgLocal = './patch_boundaries.png';
    if exist(patchImgLocal, 'file')
        figure; imshow(imread(patchImgLocal)); title('Patch Boundaries');
    end

elseif strcmp(type, 'rx')
    remoteReconImage = [remoteScriptDir, '/recon/simulated_image.png'];
    localReconImage = './recon/simulated_image.png';
    if ~exist('./recon', 'dir'), mkdir('./recon'); end
    system(sprintf('scp %s@%s:"%s" "%s"', remoteUser, remoteHost, remoteReconImage, localReconImage));
    if exist(localReconImage, 'file')
        figure; imshow(imread(localReconImage)); title('Reconstructed Image');
    end

elseif strcmp(type, 'both')
    system(sprintf('scp %s@%s:%s/patch_boundaries.png ./patch_boundaries.png', ...
        remoteUser, remoteHost, remoteScriptDir));
    patchImgLocal = './patch_boundaries.png';
    remoteReconImage = [remoteScriptDir, '/recon/simulated_image.png'];
    localReconImage = './recon/simulated_image.png';
    if ~exist('./recon', 'dir'), mkdir('./recon'); end
    system(sprintf('scp %s@%s:"%s" "%s"', remoteUser, remoteHost, remoteReconImage, localReconImage));
    if exist(patchImgLocal, 'file')
        figure; imshow(imread(patchImgLocal)); title('Patch Boundaries');
    end
    if exist(localReconImage, 'file')
        figure; imshow(imread(localReconImage)); title('Reconstructed Image');
    end
end