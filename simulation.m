% --- Set arguments ---
type = 'both';  % 'tx', 'rx', or 'both'
received_file = './Binary/simulated.bin';
imagePath = 'Datasets/Kodak/kodim14.png';

useCodebook = true;
k = 512;
chunk = 4;
adaptive = '';  % 'true', 'false', or '' (auto mode)

noise = 3.0;

patch_size = 28;
low_th = 100;
high_th = 200;
v_val = 100;
kernel = 1;
depth = [];  % or 4/5/6/7 or [] if not set




pythonExe = '"C:\Python311\cv\Scripts\python.exe"'; 
script = 'sim.py';


% --- Build command ---
cmd = sprintf('%s %s --type %s --received_file "%s"', pythonExe, script, type, received_file);
if ~isempty(imagePath), cmd = [cmd, sprintf(' --image_path "%s"', imagePath)]; end
if useCodebook, cmd = [cmd, ' --use_codebook']; end
if ~isempty(k), cmd = [cmd, sprintf(' --k %d', k)]; end
if ~isempty(chunk), cmd = [cmd, sprintf(' --chunk_size %d', chunk)]; end
if ~isempty(adaptive), cmd = [cmd, sprintf(' --adaptive %s', adaptive)]; end
if ~isempty(patch_size), cmd = [cmd, sprintf(' --patch_size %d', patch_size)]; end
if ~isempty(noise), cmd = [cmd, sprintf(' --noise %.1f', noise)]; end
if ~isempty(low_th), cmd = [cmd, sprintf(' --low_th %d', low_th)]; end
if ~isempty(high_th), cmd = [cmd, sprintf(' --high_th %d', high_th)]; end
if ~isempty(v_val), cmd = [cmd, sprintf(' --v_val %d', v_val)]; end
if ~isempty(kernel), cmd = [cmd, sprintf(' --kernel %d', kernel)]; end
if ~isempty(depth), args = [args, sprintf(' --depth %d', depth)]; end

% --- Run ---
status = system(cmd); if status ~= 0, error('Python script failed.'); end