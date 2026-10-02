%%loadings
clear; clc
M     = 30;

load ('/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all/data_selection/attention_all/data2.mat','c');
inputPath = '/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_adjust_direction/';

% 一次性把 a1-a4 都读出来（假设都在同一个 mat 里）
matFile = fullfile(inputPath,"1_30_a.mat");
S = load(matFile, 'a1','a2','a3');

load(matFile, 'a1','a2','a3');
data_GAM=[a1(:,1),a2(:,1),a3(:,1),c];
writematrix(data_GAM,'data_GAM.csv')

A = {S.a1, S.a2, S.a3};

%%Three modalities  Sig index
A1_all = A{1}(:,1);%%------------------PIck index
A2_all = A{2}(:,1);%%------------------PIck index
A3_all = A{3}(:,1);%%-------
% ----------PIck index

A_all = {A1_all, A2_all, A3_all};
xlab_list = {'EEG','灰质体积','功能连接强度'};
for k = 1:3
    x = c(:,1);
    % GLM：ref ~ x + covariates(c(:,[1,2]))
    %     [b, dev, stats] = glmfit(x, A_all{k}, 'normal', 'link', 'identity');
    %     R = stats.resid;
    %     y_plot = b(1) + b(2)*x + R;


    figure;
    gretna_plot_regression_wq(x, A_all{k},1);

    xlabel('年龄');
    ylabel(xlab_list{k});

    file_tags = {'eeg','GMV','FCS'};
    saveas(gcf, sprintf('develop_%s.png',  file_tags{k}));
    close(gcf);
end






%%features with age
clear; clc
M     = 30;

load ('/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all/data_selection/attention_all/data2.mat','c');
inputPath = '/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_adjust_direction/';
load('s1_1_30_.mat','index')
load '/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all/data_selection/attention_all/EEG_fus_reg.mat';
eeg=[Feature_fus_reg{1},Feature_fus_reg{2},Feature_fus_reg{3},Feature_fus_reg{4},Feature_fus_reg{5}];
s1=eeg(:,index);

load('s2_1_30_.mat','index')
load '/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all/data_selection/attention_all/mri_fus_reg.mat';
s2=volume_fus_reg(:,index);

load('s3_1_30_.mat','index')
load '/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all/data_selection/attention_all/FC_fus_reg.mat';
s3=FC_fus_reg(:,index);

S = [s1, s2, s3];

xlab_list = {'EEG','灰质体积','功能连接强度'};
for k = 1:size(S,2)
    x = c(:,1);
    % GLM：ref ~ x + covariates(c(:,[1,2]))
    %     [b, dev, stats] = glmfit(x, A_all{k}, 'normal', 'link', 'identity');
    %     R = stats.resid;
    %     y_plot = b(1) + b(2)*x + R;


    figure;
    gretna_plot_regression_wq(x, S(:,k),1);

%     xlabel('年龄');
%     ylabel(xlab_list{k});
% 
%     file_tags = {'eeg','GMV','FCS'};
%     saveas(gcf, sprintf('develop_%s.png',  file_tags{k}));
%     close(gcf);
end





