close all
clear

folders = dir('data/');
results = zeros(size(folders,1)-2,3);
sum_mean1 = 0;
sum_mean3 = 0;

for folder=3:size(folders,1)
    load (strcat('data/', folders(folder).name , '/Results.mat'), 'mean1', 'mean3');
    results(folder-2,1) = mean1;
    results(folder-2,2) = mean3;
    
    sum_mean1 = sum_mean1 + mean1;
    sum_mean3 = sum_mean3 + mean3;
    
    improvement = ((sum_mean1 - sum_mean3) /sum_mean1) * 100;
    results(folder-2,3) = improvement;   
end

sum_mean1 = sum_mean1/(size(folders,1)-2);
sum_mean3 = sum_mean3/(size(folders,1)-2);

figure(2);
bar(results,'grouped');
title('Estimation error of two methods for several datasets');
xlabel('datasets (red bars is our features)');
ylabel('mean of errors in angles');

