close all
clear

addpath('MocapRoutines');
 
folders = dir('data');

for folder=3:size(folders,1)
    disp(strcat('folder: ', folders(folder).name));
    load (strcat('data/', folders(folder).name , '/sc.mat'));
    load (strcat('data/', folders(folder).name , '/XY.mat'), 'Y');
    
    % pick a portion of frames for training and the others for test.
    step = 3;
    m = min(size(sc,1),1000); %Number of frames
    n = size(sc,2);  %Number of samples
    a = 1:step:m;
    b = step-1:step:m;
    c = step:step:m;
    f = sort([b c]);

    sc1 = sc(f,:,:);
    sc2 = sc(a,:,:);
    Y1 = Y(f,:)';
    Y2 = Y(a,:)';
    m1 = size(sc1,1); 
    n1 = size(sc1,2); 
    m2 = size(sc2,1); 
    n2 = n1; 
    nsamp = n2;
    
    % compute HoSC
    disp('computing HoSC');
        %Includes all the shape contexts  sc(i,j,k): i=frameN, j=sampleN, k is 1:60 and runs over each shape context
    sigma1 = 18;
    tmp = zeros(m1*n1,60);
    for i = 1 : m1
        for j = 1 : n1
            tmp((i-1)*n1+j,:) = sc1(i,j,:);
        end
    end

        % This part is just for taking a random sample out of tmp (it is so large that k-means fails to converge)
    a = rand(size(tmp,1),1);
    [a b] = sort(a);
    tmp = tmp(b,:);
    clear a b sc Y 
    disp('kmeans started.');
    [a centers] = kmeans(tmp(1:20000,:),100);
    sc100 = zeros(m1,100);
    disp('kmeans finished.');

    for i = 1 : m1
        for j = 1 : n1
            for k = 1 : 100
                t = zeros(1,60);
                t(1,:) = sc1(i,j,:);
                sc100(i,k) = sc100(i,k) + exp(-(t-centers(k,:))*(t-centers(k,:))'/(2*sigma1^2));
            end
        end
    end
    HoSC1 = sc100';
    clear sc100;
    
    disp('HoSC1 Done');

    sc100 = zeros(m2,100);
    for i = 1 : m2
        for j = 1 : n2
            for k = 1 : 100
                t = zeros(1,60);
                t(1,:) = sc2(i,j,:);
                sc100(i,k) = sc100(i,k) + exp(-(t-centers(k,:))*(t-centers(k,:))'/(2*sigma1^2));
            end
        end
    end
    HoSC2 = sc100';
    clear sc100 tmp;
    
    disp('HoSC2 Done');
    
    save(strcat('data/', folders(folder).name , '/TempData.mat'), 'sc1','sc2','HoSC1','HoSC2','Y1','Y2', 'centers');
    
    % run algorithm 1 (HoSC)
    disp('runing algorithm 1 (HoSC)');
    Y1 = Y1(4:60,:);
    Y2 = Y2(4:60,:);
    Y1 = dewrap(Y1);
    Y2 = dewrap(Y2);

    F1 = kernelizeX(HoSC1,HoSC1);
    F2 = kernelizeX(HoSC2,HoSC1);

    A = rvm_reg(Y1,F1,1000);
    Y2r = A*F2;
    [mean1,e1] = angle_error(wrap(Y2),wrap(Y2r));
    fprintf('\n Mean test error 1 obtained = %f\n',mean1);
    figure(1);
    plot(e1)
    hold on;
    
    clear A Y2r F1 F2 Y1 Y2 HoSC1 HoSC2;
    
    save(strcat('data/', folders(folder).name , '/Results.mat'), 'e1', 'mean1');
    
    
%     ####################################
%     ####################################

	%learn the dictionary
    disp('learning the dictionary');
    load (strcat('data/',folders(folder).name,'/TempData.mat'), 'Y1', 'Y2', 'sc1', 'sc2', 'centers');
    m1 = size(sc1,1);
    m2 = size(sc2,1);
    
    disp('computing coefs');
        % computing X1
    for i=1:m1
        X(((i-1)*nsamp + 1): i*nsamp,:) = sc1(i,1:nsamp,:);
    end

    Xout = LLC_coding_appr(centers, X, 4);
    Xout = Xout';
    num = size(Xout, 2);
    X1 = zeros(100,m1);
    for j=1:num
        if j~=num
            X1(:,floor(j/nsamp + 1)) = X1(:,floor(j/nsamp + 1)) + Xout(:,j);
        else
            X1(:,floor(j/nsamp)) = X1(:,floor(j/nsamp)) + Xout(:,j);
        end
    end
    
    clear X num Xout;
    disp('X1 computed.');
    
        % computing X2
    for i=1:m2
        X(((i-1)*nsamp + 1): i*nsamp,:) = sc2(i,1:nsamp,:);
    end

    Xout = LLC_coding_appr(centers, X, 4);
    Xout = Xout';
    num = size(Xout, 2);
    X2 = zeros(100,m2);
    for j=1:num
        if j~=num
            X2(:,floor(j/nsamp + 1)) = X2(:,floor(j/nsamp + 1)) + Xout(:,j);
        else
            X2(:,floor(j/nsamp)) = X2(:,floor(j/nsamp)) + Xout(:,j);
        end
    end
    clear X num Xout;
    disp('X2 computed.');
    save(strcat('data/',folders(folder).name,'/TempData.mat'), 'X1', 'X2', '-append');
    
	% run the algorithm 2 (ours)
    disp('run the second algorithm (ours).');
    Y1 = Y1(4:60,:);
    Y2 = Y2(4:60,:);
    Y1 = dewrap(Y1);
    Y2 = dewrap(Y2);

    F1 = kernelizeX2(X1,X1);
    F2 = kernelizeX2(X2,X1);

    A = rvm_reg(Y1,F1,1000);
    Y2r = A*F2;
    
    [mean3,e3] = angle_error(wrap(Y2),wrap(Y2r));
    fprintf('\nMean test error 2 obtained = %f\n',mean3);
    figure(1);
    p = plot(e3);
    set(p,'Color','red');
    
    clear A Y2r F1 F2;
    
	% compute improvements (in each angle and overall)
    m_diff = ((mean1-mean3)/mean1) * 100;
    fprintf('\nm_diff obtained is = %f\n',m_diff);
    
	% write errors and all important (middle) variables to coresponding
     save(strcat('data/', folders(folder).name , '/Results.mat'), 'e3', 'mean3', 'm_diff', '-append');
    clear m_diff e1 e2 e3 mean1 mean3 m1 m2 Y1 Y2 X1 X2 HoSC1 HoSC2 sc1 sc2 sc tmp Xout;
    clearvars -except folders folder
end

show_results.m

