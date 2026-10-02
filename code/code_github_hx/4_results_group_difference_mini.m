%%mini
% clc;clear
% load /path/to/project/Project_att_sMRI/data_selection/ref_att_mini.mat
% att = str2double(att);
% idx_any1 = find(any(att == 1, 2));
% sub_att_any1=sub_att(idx_any1);
% idx_all0 = find(all(att == 0, 2));
% sub_att_all0=sub_att(idx_all0);


%%age sex group difference


%%age
% age1 = c1(group==1,1);  % 组1
% age0 = c1(group==0,1);  % 组0
% [h,p] = ttest2(age1, age0);
% fprintf('p = %.4f\n', p);
% 
% 
% %%sex
% sex1 = c1(group==1,2);
% sex0 = c1(group==0,2);
% 
% sex_all = [sex1; sex0];
% grp_all = [ones(size(sex1)); zeros(size(sex0))];
% 
% [tbl,chi2,p] = crosstab(grp_all, sex_all);
% 
% disp(tbl)
% fprintf('chi2 = %.4f, p = %.4f\n', chi2, p);


clc;clear
load /path/to/project/Project_att_sMRI/data_selection/ref_att_mini_correct.mat
% load('mini_match_HC.mat')
% sub_att_all0=HC_sub;
M     = 30;
load ('/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all/data_selection/attention_all/data2.mat','ref','c','sub_final','mfd');
inputPath = '/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_adjust_direction/';
outPath='/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_plot/mini_diff/'

% 一次性把 a1-a4 都读出来（假设都在同一个 mat 里）
matFile = fullfile(inputPath, sprintf('1_%d_a.mat',M));
S = load(matFile, 'a1','a2','a3');

A = {S.a1, S.a2, S.a3};
xlab_list = {'eeg','灰质体积','功能连接强度'};


for i = 1:3
    % 前10%（最小的10%）
    [~,idx_low] = intersect(sub_final,sub_att_any1);
    
    % 后10%（最大的10%）
     [~,idx_high] = intersect(sub_final,sub_att_all0);
    % 只保留前后10%的index
    idx_keep = [idx_low; idx_high];

    % 对应的 group
    group = [ ...
        ones(numel(idx_low),1);   % 最小10% → 1
        zeros(numel(idx_high),1)  % 最大10% → 0
        ];

    % 如果你要把 X 也同步裁掉
    c1 = c(idx_keep,:);
    weights = A{i}(:,1);weights=weights(idx_keep,:);%%------------------PIck index 1
% 
%     %%delete extreme values
%     index_delete=find(weights==min(weights))
%     weights(index_delete)=[];group(index_delete)=[];

    % GLM：
%      [b, dev, stats] = glmfit([group,c1(:,1)], weights, 'normal', 'link', 'identity');
    [b, dev, stats] = glmfit(group, weights, 'normal', 'link', 'identity');
    t_group = stats.t(2)
    p_group = stats.p(2)

    g = group(:);
    y = weights(:);

    %     R = stats.resid;
    %     y_plot = b(1) + b(2)*group + R;
    % %%-----------------------------Plot
    y0 = y(g==0);   % 前10%
    y1 = y(g==1);   % 后10%

    figure('Color','white'); hold on;
    set(gca,'Color','white');
    % ========= 云（核密度）=========
    [f0, xi0] = ksdensity(y0);
    [f1, xi1] = ksdensity(y1);

    % 统一成列向量，避免 patch/拼接时形状翻车
    f0  = f0(:);   xi0 = xi0(:);
    f1  = f1(:);   xi1 = xi1(:);

    % 控制云的宽度（左右两组取同一缩放，保证可比）
    scale = 0.35 / max([f0; f1]);
    f0 = f0 * scale;
    f1 = f1 * scale;

    % 颜色
    c0 = [0.3 0.6 0.9];
    c1 = [0.9 0.4 0.4];

    % 组的位置（中轴）
    xL = 0;    % 左组中心
    xR = 1;    % 右组中心

    % —— 左边：画在中轴左侧（xL - f0）
    x_cloud0 = [xL - f0; flipud( xL*ones(size(f0)) )];
    y_cloud0 = [xi0;     flipud( xi0 )];
    patch(x_cloud0, y_cloud0, c0, 'FaceAlpha',0.4, 'EdgeColor','none');

    % —— 右边：画在中轴右侧（xR + f1）
    x_cloud1 = [xR + f1; flipud( xR*ones(size(f1)) )];
    y_cloud1 = [xi1;     flipud( xi1 )];
    patch(x_cloud1, y_cloud1, c1, 'FaceAlpha',0.4, 'EdgeColor','none');

    % ========= 雨（散点）=========
    x0 = xL + (rand(size(y0)) - 0.5) * 0.15;
    x1 = xR + (rand(size(y1)) - 0.5) * 0.15;

    scatter(x0, y0, 18, [0.2 0.4 0.8], 'filled', 'MarkerFaceAlpha', 0.35);
    scatter(x1, y1, 18, [0.8 0.2 0.2], 'filled', 'MarkerFaceAlpha', 0.35);

    % ========= 中央统计（均值 + 95%CI）=========
    m0 = mean(y0);  m1 = mean(y1);
    se0 = std(y0) / sqrt(numel(y0));
    se1 = std(y1) / sqrt(numel(y1));

    errorbar([xL xR], [m0 m1], [1.96*se0 1.96*se1], ...
        'k', 'LineWidth', 2, 'CapSize', 14);

    % ========= 坐标外观 =========
    xlim([-0.5 1.5]);
    xticks([0 1]);
    % xticklabels({'前10%','后10%'});

    set(gca, 'LineWidth', 1.2, 'FontSize', 13);

    %xticklabels({'高注意能力组','低注意能力组'});

    %xlabel('注意能力分组','FontSize',15,'FontWeight','bold');
    %ylabel('权重','FontSize',15,'FontWeight','bold');

    ax = gca;
    ax.FontSize = 13;
    ax.LineWidth = 1.2;
    % grid on;
    file_tags = {'EEG','GMV','fcs'};
      saveas(gcf, sprintf([outPath,'group_%s.pdf'],  file_tags{i}));
    close(gcf);
    save(sprintf('group_%s_mini.mat', file_tags{i}),'t_group','p_group');
  
end

