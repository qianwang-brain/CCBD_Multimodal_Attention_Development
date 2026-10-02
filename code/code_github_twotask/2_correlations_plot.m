clear; clc
lamda = 1;
M     = 30;

load ('/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/data_selection/data2.mat','c','ref','TIV');
inputPath = '/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/results_1/results_adjust_direction/';
outPath='/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/results_1/results_plot/corr_plot/'

% 一次性把 a1-a4 都读出来（假设都在同一个 mat 里）
matFile = fullfile(inputPath,"1_30_a.mat");
S = load(matFile, 'a1','a2','a3');

A = {S.a1, S.a2, S.a3};
xlab_list = {'eeg','灰质体积','功能连接强度'};

index_used=find(~isnan(ref));
ref=ref(index_used);c=c(index_used,:);TIV=TIV(index_used,:);
%%delte 1 people
index_delte=find(ref==211.38);
ref(index_delte)=[];c(index_delte,:)=[];TIV(index_delte)=[];
%%daoshu
ref=1./ref;
for k = 1:3
    x = A{k}(:,1);%%------------------PIck index 1
    x=x(index_used);x(index_delte)=[];
    % GLM：ref ~ x + covariates(c(:,[1,2]))
   if k==2
        [b, dev, stats] = glmfit([x,c(:,2),TIV], ref, 'normal', 'link', 'identity');
    else
        [b, dev, stats] = glmfit([x,c(:,2)], ref, 'normal', 'link', 'identity');
    end
%  [b, dev, stats] = glmfit(x, ref, 'normal', 'link', 'identity');
    R = stats.resid;
    y_plot = b(1) + b(2)*x + R;

    figure;
    gretna_plot_regression_wq(x, y_plot, 1);

    xlabel(xlab_list{k});
    ylabel('持续性注意能力');
    [r,p] = corr(x, y_plot, 'type', 'Pearson');
    txt = sprintf('r = %.2f\np = %.3g', r, p);

    text(0.05, 0.95, txt, ...
        'Units','normalized', ...
        'FontSize', 15, ...
        'VerticalAlignment','top');

    file_tags = {'eeg','GMV','FCS'};
    saveas(gcf, sprintf([outPath,'Corr_%s_daoshu.pdf'],  file_tags{k}));
    close(gcf);
end


%%%regressed age
clear; clc
lamda = 1;
M     = 30;

load ('/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/data_selection/data2.mat','c','ref','TIV');
inputPath = '/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/results_1/results_adjust_direction/';
outPath='/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/results_1/results_plot/corr_plot/'

% 一次性把 a1-a4 都读出来（假设都在同一个 mat 里）
matFile = fullfile(inputPath,"1_30_a.mat");
S = load(matFile, 'a1','a2','a3');

A = {S.a1, S.a2, S.a3};
xlab_list = {'eeg','GMV','FCS'};

index_used=find(~isnan(ref));
ref=ref(index_used);c=c(index_used,:);TIV=TIV(index_used,:);
%%delte 1 people
index_delte=find(ref==211.38);
ref(index_delte)=[];c(index_delte,:)=[];TIV(index_delte)=[];
%%daoshu
ref=1./ref;
for k = 1:3
    x = A{k}(:,1);%%------------------PIck index 1
    x=x(index_used);x(index_delte)=[];
    % GLM：ref ~ x + covariates(c(:,[1,2]))
   if k==2
        [b, dev, stats] = glmfit([x,c(:,[1,2]),TIV], ref, 'normal', 'link', 'identity');
    else
        [b, dev, stats] = glmfit([x,c(:,[1,2])], ref, 'normal', 'link', 'identity');
    end
%  [b, dev, stats] = glmfit(x, ref, 'normal', 'link', 'identity');
    R = stats.resid;
    y_plot = b(1) + b(2)*x + R;

    figure;
    gretna_plot_regression_wq(x, y_plot, 1);

    xlabel(xlab_list{k});
    ylabel('Sustained attention');
    [r,p] = corr(x, y_plot, 'type', 'Pearson');
    txt = sprintf('r = %.2f\np = %.3g', r, p);

    text(0.05, 0.95, txt, ...
        'Units','normalized', ...
        'FontSize', 15, ...
        'VerticalAlignment','top');

    file_tags = {'eeg','GMV','FCS'};
    saveas(gcf, sprintf([outPath,'Corr_%s_daoshu_regressed_age.pdf'],  file_tags{k}));
    close(gcf);
end





% 
% %%--------------interaction  age and attention
% clear; clc
% lamda = 1;
% M     = 30;
% 
% load ('/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/data_selection/data2.mat','c','ref');
% inputPath = '/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/results_1/results_adjust_direction/';
% 
% % 一次性把 a1-a4 都读出来（假设都在同一个 mat 里）
% matFile = fullfile(inputPath,"1_30_a.mat");
% S = load(matFile, 'a1','a2','a3');
% 
% A = {S.a1, S.a2, S.a3};
% xlab_list = {'eeg）','灰质体积','功能连接强度'};
% 
% index_used=find(~isnan(ref));
% ref=ref(index_used);
% c_t=[];c_p=[];
% for k = 1:3
%     x = A{k}(:,1);%%------------------PIck index 1
%     x=x(index_used);
%     age=c(index_used,1);
%     %%central
%     x_c   = x   - mean(x);
%     age_c = age - mean(age);
% 
%     X = [c(index_used,2),x_c, age_c, x_c .* age_c];%%addtionaly contronled sex
%     [b, dev, stats] = glmfit(X, ref, 'normal', 'link', 'identity');
%     c_t=[c_t,stats.t];c_p=[c_p,stats.p];
%     % 取回系数
%     b = b;   % 你已有
% 
%     % 设定 sex 固定为 0
%     sex_fixed = 0;
% 
%     % 取 age 的 ±1SD
%     age_sd = std(age_c);
%     age_low  = -age_sd;
%     age_mean = 0;
%     age_high = age_sd;
% 
%     % X 范围
%     x_range = linspace(min(x_c), max(x_c), 100);
% 
%     % 预测函数
%     y_low  = b(1) + b(2)*sex_fixed + ...
%         b(3)*x_range + b(4)*age_low  + b(5)*(x_range*age_low);
% 
%     y_mean = b(1) + b(2)*sex_fixed + ...
%         b(3)*x_range + b(4)*age_mean + b(5)*(x_range*age_mean);
% 
%     y_high = b(1) + b(2)*sex_fixed + ...
%         b(3)*x_range + b(4)*age_high + b(5)*(x_range*age_high);
% 
%     % 作图
%     figure;
%     plot(x_range, y_low,  'b', 'LineWidth',2); hold on
%     plot(x_range, y_mean, 'k', 'LineWidth',2);
%     plot(x_range, y_high, 'r', 'LineWidth',2);
% 
%     legend('Age -1SD','Age Mean','Age +1SD');
%     xlabel('Weights (centered)');
%     ylabel('持续性注意');
%     title('Interaction: Weights × Age (Sex controlled)');
% 
%     file_tags = {'eeg','GMV','FCS'};
%     saveas(gcf, sprintf('Interac_%s.png',  file_tags{k}));
%     close(gcf);
% 
% end
% 
% save('results_interaction_age_atten.mat','c_t','c_p')