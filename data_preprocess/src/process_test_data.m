clear;clc;
video_dir_path = '../videos/test';
% video_size = [241 311;181 201;161 174;168 211;308 475];
scale = 4;
blur_size = 2;
name = {'Star_Fan.mp4', 'Flag.mp4','Treadmill.mp4','Turbine.mp4','Dirty_Dancing.mp4'};
% numfrms = [1 300; 1 290;1 300;1 350;49 106];
% sc = [200 440 340 650;150 330 400 600;220 380 263 436;140 307 220 430;150 457 244 718];
numfrms = [1 300; 1 292;1 125; 1 350; 49 107];
video_size = [240 320; 180 240; 174 150; 168 224; 354 474];
sc = [200 439 335 654;150 329 380 619;213 386 275 424;140 307 212 435;104 457 244 717];
if ~exist('../../data/test')
	mkdir('../../data/test');
end

H = fspecial('gaussian', [5 5], blur_size);  % gaussian filter

for i = 1:length(name)
    filepath = fullfile(video_dir_path, name{i});  % YUV file dir
    fid = VideoReader(filepath);
    numfrm = [numfrms(i, 1), numfrms(i, 2)];
    nf = numfrm(2) - numfrm(1) + 1;
    fprintf('blurring... the %d video sequences, ... %d frms\n', i, nf);
    v_s = [video_size(i,1), video_size(i,2)];
    v_s = floor(v_s / scale / 4) * (scale * 4);
    region = [sc(i,1),sc(i,2),sc(i,3),sc(i,4)];
    k = 1;
    hr_data = zeros([1,nf, 1, v_s]);
    lr_data = zeros([1,nf, 1, v_s]);
    lrs_data = zeros([1,nf, 1, v_s/scale]);
    
    cb_data = zeros([1,nf, 1, v_s]);
    cr_data = zeros([1,nf, 1, v_s]);
    
    cb = zeros([video_size(i,1), video_size(i,2)]);
    cr = zeros([video_size(i,1), video_size(i,2)]);
    for j = numfrm(1):numfrm(2)
        y = read(fid, j);
        im = rgb2ycbcr(y);
        y = im(region(1):region(2), region(3):region(4), 1);
        y = double(y) / 255;
         
        cb = im(region(1):region(2), region(3):region(4), 2);
        cb = double(cb) / 255; 
        cr = im(region(1):region(2), region(3):region(4), 3);
        cr = double(cr) / 255;
        
        hr_data(1,k,1,:,:) = modcrop(y, scale*4);
        cb_data(1,k,1,:,:) = modcrop(cb, scale*4);
        cr_data(1,k,1,:,:) = modcrop(cr, scale*4);
        
        g = imfilter(y, H, 'corr', 'replicate');
        g = modcrop(g, scale*4);
        gs = imresize(g, 1/scale, 'bicubic');
        g = imresize(gs, scale, 'bicubic');
        lr_data(1,k,1,:,:) = g;
        lrs_data(1,k,1,:,:) = gs; 
        k = k + 1;
    end
save(strcat('../../data/test/', num2str(nf), '_', name{i}(1:end-4),'_scale',num2str(scale),'_blur_2.mat'), ...
'hr_data', 'lr_data', 'lrs_data', 'cb_data', 'cr_data', '-v7.3');
end
fprintf('...make done\n');