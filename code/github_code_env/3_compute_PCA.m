%% 读取人工审核并统一方向后的最终环境数据

clear;
clc;

result_dir = ...
    '/path/to/project/Envrion/manual_audit_processed';

load(fullfile(result_dir, ...
    'environment_items_oriented_final.mat'), ...
    'env_oriented_final', ...
    'env_name_final', ...
    'env_sub_final', ...
    'T_final', ...
    'domain_summary_final');

fprintf('最终环境数据：%d人 × %d个变量。\n', ...
    size(env_oriented_final,1), ...
    size(env_oriented_final,2));

disp(domain_summary_final);

%% 根据T_final中的最终domain提取六个维度

idx_SES = find(T_final.domain == "SES");
idx_Parenting = find(T_final.domain == "Parenting");
idx_SchoolResource = find(T_final.domain == "SchoolResource");
idx_SchoolEcology = find(T_final.domain == "SchoolEcology");
idx_Lifestyle = find(T_final.domain == "Lifestyle");
idx_Learning = find(T_final.domain == "Learning");

%% 提取最终数据
% 注意：这里必须使用env_oriented_final，而不是原始env

SES_data = env_oriented_final(:, idx_SES);
Parenting_data = env_oriented_final(:, idx_Parenting);
SchoolResource_data = env_oriented_final(:, idx_SchoolResource);
SchoolEcology_data = env_oriented_final(:, idx_SchoolEcology);
Lifestyle_data = env_oriented_final(:, idx_Lifestyle);
Learning_data = env_oriented_final(:, idx_Learning);

X = {
    SES_data
    Parenting_data
    SchoolResource_data
    SchoolEcology_data
    Lifestyle_data
    Learning_data
    };

group_names = {
    'SES'
    'Parenting'
    'SchoolResource'
    'SchoolEcology'
    'Lifestyle'
    'Learning'
    };

%% 分别对六个环境维度进行完整案例PCA
pca_output_dir='/path/to/project/Envrion/results_PCA/'
PCA_results = struct();

% 保存到全部受试者位置；
% 没有参与某个维度PCA的人，该维度PC1记为NaN
PC1_table = table(env_sub_final, ...
    'VariableNames', {'env_sub'});

pca_summary = table();

for g = 1:numel(group_names)

    this_name = group_names{g};
    Xg = X{g};

    idx_current = find(T_final.domain == string(this_name));
    item_names = string(T_final.variable_name(idx_current));

    fprintf('\n========================================\n');
    fprintf('Running PCA for group: %s\n', this_name);
    fprintf('变量数：%d\n', size(Xg,2));
    fprintf('========================================\n');

    %% 1. 当前维度完整案例筛选
    % 要求该维度所有item都有有限值
    complete_rows = all(isfinite(Xg), 2);

    Xg_complete = Xg(complete_rows, :);
    sub_complete = env_sub_final(complete_rows);

    n_total = size(Xg,1);
    n_complete = sum(complete_rows);
    n_excluded = n_total - n_complete;

    fprintf('总人数：%d\n', n_total);
    fprintf('完整案例人数：%d\n', n_complete);
    fprintf('因缺失排除人数：%d\n', n_excluded);
    fprintf('保留比例：%.2f%%\n', 100*n_complete/n_total);

    if n_complete <= size(Xg_complete,2)
        error(['%s维度完整案例人数%d不大于变量数%d，', ...
            '无法稳定进行PCA。'], ...
            this_name, n_complete, size(Xg_complete,2));
    end

    %% 2. 检查完整案例数据中的方差
    item_sd = std(Xg_complete, 0, 1);

    zero_variance = ...
        ~isfinite(item_sd) | item_sd == 0;

    if any(zero_variance)
        fprintf('以下变量在完整案例样本中没有方差：\n');
        disp(item_names(zero_variance));

        error(['%s维度存在零方差变量。', ...
            '请先人工确认，不要静默删除。'], ...
            this_name);
    end

    %% 3. 在当前维度完整案例中进行标准化
    item_mean = mean(Xg_complete, 1);
    item_sd = std(Xg_complete, 0, 1);

    Xz_complete = ...
        (Xg_complete - item_mean) ./ item_sd;

    % 再次确认标准化后没有NaN或Inf
    if any(~isfinite(Xz_complete), 'all')
        error('%s维度标准化后仍存在NaN或Inf。', this_name);
    end

    %% 4. PCA
    [coeff, score_complete, latent, ~, explained] = ...
        pca(Xz_complete);
    %% Kaiser准则：特征值大于1的主成分数量
    n_keep_kaiser = sum(latent > 1);

    if n_keep_kaiser < 1
        n_keep_kaiser = 1;
    end


    %% 5. 统一PC1方向
    % 所有item已经统一为高分=环境更有利
    % 用方向一致item的平均z分作为锚点

    favourable_anchor = mean(Xz_complete, 2);

    direction_r = corr( ...
        score_complete(:,1), ...
        favourable_anchor);

    pc1_flipped = false;

    if direction_r < 0
        coeff(:,1) = -coeff(:,1);
        score_complete(:,1) = -score_complete(:,1);

        % PCA整体方向翻转时，所有PC1相关结果同步翻转
        direction_r = -direction_r;
        pc1_flipped = true;
    end

    %% 6. 放回原始3735人位置
    % 没参与当前维度PCA的人保持NaN

    pc1_all = nan(n_total,1);
    pc1_all(complete_rows) = score_complete(:,1);

    field_name = matlab.lang.makeValidName(this_name);

    PC1_table.(field_name) = pc1_all;

    %% 7. 保存当前维度结果

    PCA_results.(field_name).item_indices = idx_current;
    PCA_results.(field_name).item_names = item_names;

    PCA_results.(field_name).complete_rows = complete_rows;
    PCA_results.(field_name).included_subjects = sub_complete;
    PCA_results.(field_name).excluded_subjects = ...
        env_sub_final(~complete_rows);

    PCA_results.(field_name).n_total = n_total;
    PCA_results.(field_name).n_complete = n_complete;
    PCA_results.(field_name).n_excluded = n_excluded;

    PCA_results.(field_name).standardization_mean = item_mean;
    PCA_results.(field_name).standardization_sd = item_sd;

    PCA_results.(field_name).coeff = coeff;
    PCA_results.(field_name).score_complete = score_complete;
    PCA_results.(field_name).PC1_all_subjects = pc1_all;

    PCA_results.(field_name).latent = latent;
    PCA_results.(field_name).explained = explained;

    PCA_results.(field_name).direction_anchor_correlation = ...
        direction_r;

    PCA_results.(field_name).pc1_flipped = ...
        pc1_flipped;

    %% 8. PCA汇总表

    current_summary = table( ...
        string(this_name), ...
        size(Xg_complete,2), ...
        n_total, ...
        n_complete, ...
        n_excluded, ...
        100*n_complete/n_total, ...
        explained(1), ...
        direction_r, ...
        pc1_flipped, ...
        'VariableNames', { ...
        'domain', ...
        'n_items', ...
        'n_total_subjects', ...
        'n_complete_subjects', ...
        'n_excluded_subjects', ...
        'complete_case_percent', ...
        'PC1_explained_percent', ...
        'PC1_anchor_correlation', ...
        'PC1_flipped'} ...
        );

    pca_summary = [pca_summary; current_summary];

    %% ---------- 存储当前维度全部结果 ----------

    % 保证结构体字段名合法
    field_name = matlab.lang.makeValidName(this_name);

    % 把所有主成分得分放回全部被试位置
    % 未参与当前维度PCA的被试为NaN
    score_all = nan(n_total, size(score_complete, 2));
    score_all(complete_rows, :) = score_complete;

    % 被试名 + 是否参与PCA + PC1得分
    subject_score_table = table( ...
        string(env_sub_final), ...
        complete_rows, ...
        pc1_all, ...
        'VariableNames', { ...
        'env_sub', ...
        'included_in_PCA', ...
        'PC1_score'} ...
        );

    % item名称 + PC1载荷
    loading_table = table( ...
        item_names, ...
        coeff(:,1), ...
        'VariableNames', { ...
        'variable_name', ...
        'PC1_loading'} ...
        );

    % 当前维度基本信息
    PCA_results.(field_name).domain = string(this_name);

    PCA_results.(field_name).item_indices = idx_current;
    PCA_results.(field_name).item_names = item_names;
    PCA_results.(field_name).n_items = numel(item_names);

    % 被试信息
    PCA_results.(field_name).n_total_subjects = n_total;
    PCA_results.(field_name).n_complete_subjects = n_complete;
    PCA_results.(field_name).n_excluded_subjects = n_excluded;

    PCA_results.(field_name).complete_rows = complete_rows;
    PCA_results.(field_name).included_subjects = ...
        string(env_sub_final(complete_rows));
    PCA_results.(field_name).excluded_subjects = ...
        string(env_sub_final(~complete_rows));

    % 标准化参数
    PCA_results.(field_name).standardization_mean = item_mean;
    PCA_results.(field_name).standardization_sd = item_sd;

    % PCA完整结果
    PCA_results.(field_name).coeff = coeff;

    % 仅完整案例被试的全部PC得分
    PCA_results.(field_name).score_complete = score_complete;

    % 放回全部被试位置后的全部PC得分
    PCA_results.(field_name).score_all_subjects = score_all;

    % 最终使用的PC1
    PCA_results.(field_name).PC1_complete = score_complete(:,1);
    PCA_results.(field_name).PC1_all_subjects = pc1_all;

    PCA_results.(field_name).latent = latent;
    PCA_results.(field_name).explained = explained;
    PCA_results.(field_name).cumulative_explained = cumsum(explained);

    % Kaiser标准，只保存用于描述
    PCA_results.(field_name).n_keep = n_keep_kaiser;

    % PC1方向统一信息
    PCA_results.(field_name).direction_anchor_correlation = direction_r;
    PCA_results.(field_name).pc1_flipped = pc1_flipped;

    % 两张table也直接保存在结构体中
    PCA_results.(field_name).subject_score_table = ...
        subject_score_table;

    PCA_results.(field_name).loading_table = ...
        loading_table;

    fprintf('当前维度结果已保存进PCA_results.%s\n', field_name);
    fprintf('PC1解释方差：%.2f%%\n', explained(1));
    fprintf('PC1方向锚点相关：%.3f\n', direction_r);
    fprintf('PC1是否翻转：%d\n', pc1_flipped);
end

%% ---------- 保存六个环境维度的全部PCA结果 ----------

result_file = fullfile( ...
    pca_output_dir, ...
    'results_env_6factors.mat');

save( ...
    result_file, ...
    'PCA_results', ...
    'PC1_table', ...
    'pca_summary', ...
    'env_sub_final', ...
    'env_name_final', ...
    'T_final', ...
    'domain_summary_final', ...
    'group_names', ...
    '-v7.3');

fprintf('\n六个维度的PCA结果已统一保存：\n%s\n', result_file);




% 
% 
% result_file = fullfile( ...
%     'Export_results_env_6factors.mat');
% 
% save( ...
%     result_file, ...
%     'PCA_results', ...
%     'PC1_table', ...
%     'pca_summary', ...
%      'T_final', ...
%     'domain_summary_final', ...
%     '-v7.3');
