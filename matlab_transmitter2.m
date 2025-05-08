% Set argument values
imagePath = './Datasets/Clic2021/06.png'; %Datasets/Kodak/kodim23.png
useCodebook = true;

k = 512;               % Set to [] if not needed
chunk = 4;             % Set to [] if not needed
adaptive = '';         % 'true', 'false', or '' (auto mode)
patch_size = [];       % Add this: 28 or 60, or [] if not used
depth = [];            % Add this: 5 or 6, or [] if not used

pythonExe = '"C:\Python311\cv\Scripts\python.exe"';
script = 'transmitter2.py';

cmd = sprintf('%s %s --image_path "%s"', pythonExe, script, imagePath);


if ~isempty(k), cmd = sprintf('%s --k %d', cmd, k); end
if ~isempty(chunk), cmd = sprintf('%s --chunk_size %d', cmd, chunk); end
if useCodebook, cmd = sprintf('%s --use_codebook', cmd); end
if ~isempty(adaptive) && (strcmpi(adaptive, 'true') || strcmpi(adaptive, 'false'))
    cmd = sprintf('%s --adaptive %s', cmd, adaptive); end
if ~isempty(patch_size), cmd = sprintf('%s --patch_size %d', cmd, patch_size); end  % NEW
if ~isempty(depth), cmd = sprintf('%s --depth %d', cmd, depth); end                % NEW

[status, output] = system(cmd);
disp(output);
