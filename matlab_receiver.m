adaptive = '';   % 'true' or 'false' or leave empty ''

% Set argument values
receivedFile = 'combined_binary.bin';
imagePath = 'Datasets/Kodak/kodim15.png';      % Leave '' if you don’t want to save output
useCodebook = true;   % true / false


k = 512;             % Optional, set to [] if not used
chunk = 4;           % Optional, set to [] if not used
resH = [];           % Optional
resW = [];           % Optional

pythonExe = '"C:\Python311\cv\Scripts\python.exe"';
script = 'receiver.py';

cmd = sprintf('%s %s --received_file "%s"', pythonExe, script, receivedFile);

if ~isempty(imagePath)
    cmd = sprintf('%s --image_path "%s"', cmd, imagePath);
end
if ~isempty(k)
    cmd = sprintf('%s --k %d', cmd, k);
end
if ~isempty(chunk)
    cmd = sprintf('%s --chunk_size %d', cmd, chunk);
end
if useCodebook
    cmd = sprintf('%s --use_codebook', cmd);
end
if ~isempty(resH) && ~isempty(resW)
    cmd = sprintf('%s --res_h %d --res_w %d', cmd, resH, resW);
end
if ~isempty(adaptive)
    cmd = sprintf('%s --adaptive %s', cmd, adaptive);
end

[status, output] = system(cmd);
disp(output);
