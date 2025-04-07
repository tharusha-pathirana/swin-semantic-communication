% Set argument values
imagePath = 'Datasets/Kodak/kodim23.png';
useCodebook = true;

k = 512;               % Set to [] if not needed
chunk = 4;             % Set to [] if not needed
adaptive = '';         % 'True', 'False', or '' (auto mode)

pythonExe = '"C:\Python311\cv\Scripts\python.exe"';
script = 'transmitter.py';

cmd = sprintf('%s %s --image_path "%s"', pythonExe, script, imagePath);

if ~isempty(k)
    cmd = sprintf('%s --k %d', cmd, k);
end
if ~isempty(chunk)
    cmd = sprintf('%s --chunk_size %d', cmd, chunk);
end
if useCodebook
    cmd = sprintf('%s --use_codebook', cmd);
end
if ~isempty(adaptive) && (strcmpi(adaptive, 'true') || strcmpi(adaptive, 'false'))
    cmd = sprintf('%s --adaptive %s', cmd, adaptive);
end

[status, output] = system(cmd);
disp(output);
