%add T est
% p_FDR = mafdr(ttest_results.P, 'BHFDR', true);
% find(p_FDR<0.05)

clear;
clc;
close all;

%% ============================================================
% 1. 加载行为变量、被试编号及协变量
% =============================================================

dataFile = [ ...
    '/path/to/project/' ...
    'Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/' ...
    'data_selection/data2.mat'];


load(dataFile, ...
    'sub_final', ...
    'c', ...
    'ref', ...
    'TIV');

% c(:,1) 如果是年龄，此处不需要使用
% c(:,2) = sex
sex = c(:,2);

% 统一转换为列向量
ref = ref(:);
sex = sex(:);

if isvector(TIV)
    TIV = TIV(:);
else
    % 如果TIV有多列，请根据实际情况修改列号
    TIV = TIV(:,1);
end


%% ============================================================
% 2. 加载整体MCCAR得到的EEG、GMV、FCS loadings
% =============================================================

inputPath = [ ...
    '/path/to/project/' ...
    'Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/' ...
    'results_1/results_adjust_direction/'];

outputPath = [ ...
    '/path/to/project/' ...
    'Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/' ...
    'results_1/results_plot/extreme_attention_ANCOVA/'];


if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end

matFile = fullfile(inputPath, '1_30_a.mat');

S = load(matFile, 'a1', 'a2', 'a3');

% A{1} = EEG loading
% A{2} = GMV loading
% A{3} = FCS loading
A = {
    S.a1
    S.a2
    S.a3
};

modality_names = {'EEG', 'GMV', 'FCS'};

n_modality = length(A);
n_sub = length(sub_final);


%% ============================================================
% 3. 检查数据长度
% =============================================================

assert(length(ref) == n_sub, ...
    'ref的人数与sub_final不一致。');

assert(length(sex) == n_sub, ...
    'sex的人数与sub_final不一致。');

assert(length(TIV) == n_sub, ...
    'TIV的人数与sub_final不一致。');

for k = 1:n_modality
    assert(size(A{k},1) == n_sub, ...
        '%s loading的人数与sub_final不一致。', ...
        modality_names{k});
end


%% ============================================================
% 4. 加载四个年龄组
% =============================================================

groupFile = ...
    '/path/to/project/Project_att_sMRI/data_selection/group.mat';

load(groupFile, 'IDs');

group_names = {
    'g1_3'
    'g4_6'
    'g7_9'
    'high'
};

n_group = length(group_names);

% age_group编码：
% 1 = g1_3
% 2 = g4_6
% 3 = g7_9
% 4 = high
age_group = nan(n_sub,1);

fprintf('\n=============================================\n');
fprintf('四个年龄组匹配人数\n');
fprintf('=============================================\n');

for g = 1:n_group

    curr_IDlist = IDs.(group_names{g});

    % idx_group是当前年龄组被试在sub_final中的行号
    [~, idx_group] = intersect( ...
        sub_final, ...
        curr_IDlist, ...
        'stable');

    age_group(idx_group) = g;

    fprintf('%s：%d 人\n', ...
        group_names{g}, ...
        length(idx_group));
end

fprintf('未进入四个年龄组的人数：%d\n', ...
    sum(isnan(age_group)));


%% ============================================================
% 5. 指定ref方向
% =============================================================
%
% 选择性注意：
% ref越高，注意能力越好
% higher_ref_is_better = true;
%
% 持续注意RT-CV：
% ref越低，注意能力越好
higher_ref_is_better = false;


%% ============================================================
% 6. 每个年龄组内部提取高、低注意能力各10%
% =============================================================
%
% attention_group：
% 1 = 低注意能力极端组
% 2 = 高注意能力极端组
%
% 中间80%的被试保持NaN，不进入后续ANCOVA

attention_group = nan(n_sub,1);

% extreme_idx{g,1}：第g年龄组的低注意能力被试
% extreme_idx{g,2}：第g年龄组的高注意能力被试
extreme_idx = cell(n_group,2);

fprintf('\n=============================================\n');
fprintf('每个年龄组内部提取高、低注意能力各10%%\n');
fprintf('=============================================\n');

for g = 1:n_group

    % 当前年龄组且ref不缺失的被试
    idx_g = find( ...
        age_group == g & ...
        ~isnan(ref));

    ref_g = ref(idx_g);

    % 从小到大排序
    [sorted_ref, order] = sort(ref_g, 'ascend');

    n_valid = length(idx_g);

    % 每一端取10%
    n_extreme = floor(n_valid * 0.10);

    if n_extreme < 2
        error([ ...
            '%s中有效ref人数为%d，' ...
            '每端10%%少于2人，无法分析。'], ...
            group_names{g}, ...
            n_valid);
    end

    if higher_ref_is_better

        % 选择性注意：
        % ref最低10% = 低注意能力
        % ref最高10% = 高注意能力
        low_local = order(1:n_extreme);
        high_local = order(end-n_extreme+1:end);

    else

        % 持续注意RT-CV：
        % RT-CV最高10% = 低注意能力
        % RT-CV最低10% = 高注意能力
        high_local = order(1:n_extreme);
        low_local = order(end-n_extreme+1:end);
    end

    low_idx = idx_g(low_local);
    high_idx = idx_g(high_local);

    % 编码
    attention_group(low_idx) = 1;
    attention_group(high_idx) = 2;

    extreme_idx{g,1} = low_idx;
    extreme_idx{g,2} = high_idx;

    fprintf('\n年龄组：%s\n', group_names{g});
    fprintf('有效ref总人数：%d\n', n_valid);
    fprintf('每端10%%人数：%d\n', n_extreme);
    fprintf('低注意能力人数：%d\n', length(low_idx));
    fprintf('高注意能力人数：%d\n', length(high_idx));

    if higher_ref_is_better

        fprintf('低注意能力ref范围：%.6f 至 %.6f\n', ...
            sorted_ref(1), ...
            sorted_ref(n_extreme));

        fprintf('高注意能力ref范围：%.6f 至 %.6f\n', ...
            sorted_ref(end-n_extreme+1), ...
            sorted_ref(end));

    else

        fprintf('高注意能力RT-CV范围：%.6f 至 %.6f\n', ...
            sorted_ref(1), ...
            sorted_ref(n_extreme));

        fprintf('低注意能力RT-CV范围：%.6f 至 %.6f\n', ...
            sorted_ref(end-n_extreme+1), ...
            sorted_ref(end));
    end
end


%% ============================================================
% 7. 预分配统计结果
% =============================================================

attention_names = {
    'Low attentional ability'
    'High attentional ability'
};

% 维度：
% 年龄组 × 注意组 × 模态
adjusted_mean = nan(n_group, 2, n_modality);
adjusted_se = nan(n_group, 2, n_modality);

% 模态 × 注意组
F_agegroup = nan(n_modality,2);
p_agegroup = nan(n_modality,2);

df1_agegroup = nan(n_modality,2);
df2_agegroup = nan(n_modality,2);

ANCOVA_results = struct();

anova_tables = cell(n_modality,2);
posthoc_tables = cell(n_modality,2);
stats_all = cell(n_modality,2);


%% ============================================================
% 8. 在高、低注意能力组内部，分别进行四年龄组ANCOVA
% =============================================================

for k = 1:n_modality

    loading_all = A{k}(:,1);

    fprintf('\n\n');
    fprintf('##################################################\n');
    fprintf('################## %s ##################\n', ...
        modality_names{k});
    fprintf('##################################################\n');

    for t = 1:2

        %% ----------------------------------------------------
        % 8.1 当前注意能力极端组的有效被试
        % -----------------------------------------------------

        idx_used = ...
            attention_group == t & ...
            ~isnan(age_group) & ...
            ~isnan(loading_all) & ...
            ~isnan(sex);

        % GMV额外控制TIV，因此要求TIV不缺失
        if k == 2
            idx_used = idx_used & ~isnan(TIV);
        end

        y = loading_all(idx_used);
        agegroup_used = age_group(idx_used);
        sex_used = sex(idx_used);

        if k == 2
            TIV_used = TIV(idx_used);
        else
            TIV_used = [];
        end


        %% ----------------------------------------------------
        % 8.2 显示四个年龄组的最终人数
        % -----------------------------------------------------

        fprintf('\n=============================================\n');
        fprintf('%s：%s\n', ...
            modality_names{k}, ...
            attention_names{t});
        fprintf('=============================================\n');

        for g = 1:n_group
            fprintf('%s：%d 人\n', ...
                group_names{g}, ...
                sum(agegroup_used == g));
        end

        if length(unique(agegroup_used)) ~= n_group
            error('%s的%s中存在没有有效数据的年龄组。', ...
                modality_names{k}, ...
                attention_names{t});
        end


        %% ----------------------------------------------------
        % 8.3 ANCOVA
        % -----------------------------------------------------

        if k == 2

            % GMV：
            % GMV loading ~ AgeGroup + Sex + TIV
            %
            % AgeGroup和Sex是分类变量
            % TIV是连续协变量

            [p, tbl, stats] = anovan( ...
                y, ...
                {agegroup_used, sex_used, TIV_used}, ...
                'model', 'linear', ...
                'continuous', 3, ...
                'varnames', {'AgeGroup', 'Sex', 'TIV'}, ...
                'display', 'off');

        else

            % EEG/FCS：
            % loading ~ AgeGroup + Sex
            %
            % AgeGroup和Sex均作为分类变量

            [p, tbl, stats] = anovan( ...
                y, ...
                {agegroup_used, sex_used}, ...
                'model', 'linear', ...
                'varnames', {'AgeGroup', 'Sex'}, ...
                'display', 'off');
        end

        fprintf('\nANCOVA结果：\n');
        disp(tbl);


        %% ----------------------------------------------------
        % 8.4 提取AgeGroup的F、p及自由度
        % -----------------------------------------------------
        %
        % anovan表格列顺序：
        % 第1列：Source
        % 第2列：Sum Sq.
        % 第3列：d.f.
        % 第4列：Singular?
        % 第5列：Mean Sq.
        % 第6列：F
        % 第7列：Prob>F，即p值

        row_age = find( ...
            strcmp(tbl(:,1), 'AgeGroup'), ...
            1);

        row_error = find( ...
            strcmp(tbl(:,1), 'Error'), ...
            1);

        if isempty(row_age)
            error('ANCOVA表中未找到AgeGroup行。');
        end

        % 正确提取
        df1_agegroup(k,t) = tbl{row_age,3};
        F_agegroup(k,t) = tbl{row_age,6};

        % p(1)对应第一个输入因素AgeGroup
        % 与tbl{row_age,7}相同
        p_agegroup(k,t) = p(1);

        if ~isempty(row_error)
            df2_agegroup(k,t) = tbl{row_error,3};
        end

        fprintf('\n%s，%s：\n', ...
            modality_names{k}, ...
            attention_names{t});

        fprintf( ...
            'AgeGroup效应：F(%d,%d) = %.4f，p = %.8g\n', ...
            df1_agegroup(k,t), ...
            df2_agegroup(k,t), ...
            F_agegroup(k,t), ...
            p_agegroup(k,t));


        %% ----------------------------------------------------
        % 8.5 Tukey-Kramer年龄组事后比较
        % -----------------------------------------------------

        [comparison, marginal_means, ~, display_names] = ...
            multcompare( ...
                stats, ...
                'Dimension', 1, ...
                'CType', 'tukey-kramer', ...
                'Display', 'off');

        posthoc_table = array2table( ...
            comparison, ...
            'VariableNames', { ...
                'AgeGroup1', ...
                'AgeGroup2', ...
                'LowerCI', ...
                'MeanDifference', ...
                'UpperCI', ...
                'AdjustedP'});

        fprintf('\n四年龄组Tukey-Kramer事后比较：\n');
        disp(posthoc_table);

        fprintf('年龄组顺序：\n');
        disp(display_names);

        mean_table = array2table( ...
            marginal_means, ...
            'VariableNames', { ...
                'AdjustedMean', ...
                'StandardError'});

        fprintf('四年龄组校正均值和标准误：\n');
        disp(mean_table);


        %% ----------------------------------------------------
        % 8.6 保存校正均值
        % -----------------------------------------------------

        if size(marginal_means,1) ~= n_group
            error('%s的%s中校正均值没有返回4个年龄组。', ...
                modality_names{k}, ...
                attention_names{t});
        end

        adjusted_mean(:,t,k) = marginal_means(:,1);
        adjusted_se(:,t,k) = marginal_means(:,2);


        %% ----------------------------------------------------
        % 8.7 保存完整统计结果
        % -----------------------------------------------------

        ANCOVA_results(k).modality = modality_names{k};

        ANCOVA_results(k).attention(t).attention_name = ...
            attention_names{t};

        ANCOVA_results(k).attention(t).p_all = p;
        ANCOVA_results(k).attention(t).anova_table = tbl;
        ANCOVA_results(k).attention(t).stats = stats;

        ANCOVA_results(k).attention(t).agegroup_F = ...
            F_agegroup(k,t);

        ANCOVA_results(k).attention(t).agegroup_p = ...
            p_agegroup(k,t);

        ANCOVA_results(k).attention(t).agegroup_df1 = ...
            df1_agegroup(k,t);

        ANCOVA_results(k).attention(t).agegroup_df2 = ...
            df2_agegroup(k,t);

        ANCOVA_results(k).attention(t).posthoc = ...
            posthoc_table;

        ANCOVA_results(k).attention(t).adjusted_means = ...
            marginal_means;

        ANCOVA_results(k).attention(t).agegroup_names = ...
            display_names;

        anova_tables{k,t} = tbl;
        posthoc_tables{k,t} = posthoc_table;
        stats_all{k,t} = stats;

    end
end


%% ============================================================
% 9. 输出汇总结果
% =============================================================

fprintf('\n\n');
fprintf('##################################################\n');
fprintf('############ AgeGroup ANCOVA汇总结果 ############\n');
fprintf('##################################################\n');

for k = 1:n_modality

    fprintf('\n%s\n', modality_names{k});

    for t = 1:2

        fprintf( ...
            '%s：F(%d,%d) = %.4f，p = %.8g\n', ...
            attention_names{t}, ...
            df1_agegroup(k,t), ...
            df2_agegroup(k,t), ...
            F_agegroup(k,t), ...
            p_agegroup(k,t));
    end
end

%% ============================================================
% 10. 每个发育组内高、低注意能力组的t检验
%     并准备云雨图数据
% =============================================================
%
% 高注意能力组 = attention_group == 2
% 低注意能力组 = attention_group == 1
%
% EEG/FCS：先校正sex
% GMV：先校正sex和TIV
%
% 随后在每个发育组内，对校正后的loading进行Welch t检验

ttest_t  = nan(n_modality, n_group);
ttest_df = nan(n_modality, n_group);
ttest_p  = nan(n_modality, n_group);
ttest_h  = nan(n_modality, n_group);

mean_high = nan(n_modality, n_group);
mean_low  = nan(n_modality, n_group);

sem_high = nan(n_modality, n_group);
sem_low  = nan(n_modality, n_group);

n_high = nan(n_modality, n_group);
n_low  = nan(n_modality, n_group);

% 保存用于画图的校正后个体值
plot_values_high = cell(n_modality, n_group);
plot_values_low  = cell(n_modality, n_group);

% 最终汇总表
ttest_results = table();

fprintf('\n\n');
fprintf('##################################################\n');
fprintf('######## 每个发育组内高、低注意能力t检验 ########\n');
fprintf('##################################################\n');

for k = 1:n_modality

    loading_all = A{k}(:,1);

    fprintf('\n=============================================\n');
    fprintf('%s\n', modality_names{k});
    fprintf('=============================================\n');

    for g = 1:n_group

        %% 当前发育组中的高、低注意能力极端被试
        idx_current = ...
            age_group == g & ...
            ~isnan(attention_group) & ...
            ~isnan(loading_all) & ...
            ~isnan(sex);

        % GMV额外要求TIV有效
        if k == 2
            idx_current = idx_current & ~isnan(TIV);
        end

        y_current = loading_all(idx_current);
        attention_current = attention_group(idx_current);
        sex_current = sex(idx_current);

        %% ----------------------------------------------------
        % 对loading进行协变量校正
        % -----------------------------------------------------

        SexFactor = categorical(sex_current);

        if k == 2

            % GMV：校正sex和TIV
            TIV_current = TIV(idx_current);

            % 中心化TIV
            TIV_centered = ...
                TIV_current - mean(TIV_current, 'omitnan');

            adjustment_table = table( ...
                y_current, ...
                SexFactor, ...
                TIV_centered, ...
                'VariableNames', { ...
                    'Loading', ...
                    'Sex', ...
                    'TIVC'});

            adjustment_model = fitlm( ...
                adjustment_table, ...
                'Loading ~ Sex + TIVC');

        else

            % EEG/FCS：校正sex
            adjustment_table = table( ...
                y_current, ...
                SexFactor, ...
                'VariableNames', { ...
                    'Loading', ...
                    'Sex'});

            adjustment_model = fitlm( ...
                adjustment_table, ...
                'Loading ~ Sex');
        end

        % 协变量校正后的个体值：
        % 残差 + 当前发育组原始loading均值
        adjusted_loading = ...
            adjustment_model.Residuals.Raw + ...
            mean(y_current, 'omitnan');


        %% 分成高、低注意能力组
        y_high = adjusted_loading(attention_current == 2);
        y_low  = adjusted_loading(attention_current == 1);

        y_high = y_high(~isnan(y_high));
        y_low  = y_low(~isnan(y_low));

        plot_values_high{k,g} = y_high;
        plot_values_low{k,g} = y_low;


        %% Welch独立样本t检验
        %
        % 输入顺序为high、low，因此：
        % t > 0 表示高注意组loading更高
        % t < 0 表示高注意组loading更低

        [h_value, p_value, ~, t_stats] = ttest2( ...
            y_high, ...
            y_low, ...
            'Vartype', 'unequal');

        ttest_h(k,g) = h_value;
        ttest_p(k,g) = p_value;
        ttest_t(k,g) = t_stats.tstat;
        ttest_df(k,g) = t_stats.df;


        %% 均值、SEM和人数
        n_high(k,g) = length(y_high);
        n_low(k,g) = length(y_low);

        mean_high(k,g) = mean(y_high, 'omitnan');
        mean_low(k,g) = mean(y_low, 'omitnan');

        sem_high(k,g) = ...
            std(y_high, 'omitnan') / sqrt(length(y_high));

        sem_low(k,g) = ...
            std(y_low, 'omitnan') / sqrt(length(y_low));


        %% 输出结果
        fprintf('\n%s：\n', group_names{g});

        fprintf('高注意能力：n = %d，mean = %.4f，SEM = %.4f\n', ...
            n_high(k,g), ...
            mean_high(k,g), ...
            sem_high(k,g));

        fprintf('低注意能力：n = %d，mean = %.4f，SEM = %.4f\n', ...
            n_low(k,g), ...
            mean_low(k,g), ...
            sem_low(k,g));

        fprintf('Welch t检验：t(%.2f) = %.4f，p = %.8g\n', ...
            ttest_df(k,g), ...
            ttest_t(k,g), ...
            ttest_p(k,g));


        %% 添加至汇总表
        new_row = table( ...
            string(modality_names{k}), ...
            string(group_names{g}), ...
            n_high(k,g), ...
            n_low(k,g), ...
            mean_high(k,g), ...
            sem_high(k,g), ...
            mean_low(k,g), ...
            sem_low(k,g), ...
            ttest_t(k,g), ...
            ttest_df(k,g), ...
            ttest_p(k,g), ...
            logical(ttest_h(k,g)), ...
            'VariableNames', { ...
                'Modality', ...
                'DevelopmentalGroup', ...
                'N_HighAttention', ...
                'N_LowAttention', ...
                'Mean_HighAttention', ...
                'SEM_HighAttention', ...
                'Mean_LowAttention', ...
                'SEM_LowAttention', ...
                'T', ...
                'DF', ...
                'P', ...
                'Significant'});

        ttest_results = [ttest_results; new_row];

    end
end



%% BH-FDR across all 12 modality-by-stage comparisons.
assert(numel(ttest_p) == 12 && all(isfinite(ttest_p(:))), ...
    'Expected 12 finite p-values for the prespecified FDR family.');
ttest_q = reshape(mafdr(ttest_p(:), 'BHFDR', true), size(ttest_p));
ttest_results.Q_FDR = mafdr(ttest_results.P, 'BHFDR', true);
ttest_results.Significant_raw = ttest_results.Significant;
ttest_results.Significant = ttest_results.Q_FDR < 0.05;
writetable(ttest_results, fullfile(outputPath, 'within_stage_Welch_FDR.csv'));
save(fullfile(outputPath, 'developmental_stage_statistics.mat'), ...
    'ANCOVA_results', 'ttest_results', 'ttest_t', 'ttest_df', 'ttest_p', 'ttest_q');

%% ============================================================
% 11. 输出两版箱线散点发展轨迹图
%
% Version 1：散点无边框
% Version 2：散点带深色红蓝边框
% =============================================================

rng(2026);

% 主色：更深的红色和蓝色
high_color = [0.12, 0.36, 0.68];   % 高注意能力：blue
low_color  = [0.78, 0.18, 0.22];   % 低注意能力：red

% 散点边框颜色：比填充色再深一些
high_edge_color = [0.03, 0.16, 0.38];
low_edge_color  =[0.45, 0.05, 0.08] ;

% 两种版本
version_names = {
    'no_scatter_edge'
    'colored_scatter_edge'
};

for version_id = 1:2

    % version_id = 1：散点无边框
    % version_id = 2：散点带深色红蓝边框

    fig_boxscatter = figure( ...
        'Color', 'w', ...
        'Position', [80, 100, 1600, 540], ...
        'Name', version_names{version_id});

    tiledlayout( ...
        1, 3, ...
        'TileSpacing', 'compact', ...
        'Padding', 'compact');

    group_centers = 1:n_group;

    % 每个发育组内高低注意组的左右位置
    pair_offset = 0.18;

    x_high_all = group_centers - pair_offset;
    x_low_all  = group_centers + pair_offset;


    for k = 1:n_modality

        nexttile;
        hold on;


        %% 当前模态全部数据范围
        all_values_current = [];

        for g = 1:n_group
            all_values_current = [
                all_values_current;
                plot_values_high{k,g}(:);
                plot_values_low{k,g}(:)
            ];
        end

        all_values_current = ...
            all_values_current(~isnan(all_values_current));

        y_min = min(all_values_current);
        y_max = max(all_values_current);
        y_range = y_max - y_min;

        if isempty(y_range) || isnan(y_range) || y_range == 0
            y_range = 1;
        end

        group_data_top = nan(1, n_group);


        %% 绘制四个发育组
        for g = 1:n_group

            y_high = plot_values_high{k,g};
            y_low  = plot_values_low{k,g};

            y_high = y_high(:);
            y_low  = y_low(:);

            y_high = y_high(~isnan(y_high));
            y_low  = y_low(~isnan(y_low));

            x_high = x_high_all(g);
            x_low  = x_low_all(g);


            %% 高注意能力组箱线图：红色
            boxchart( ...
                repmat(x_high, length(y_high), 1), ...
                y_high, ...
                'BoxFaceColor', high_color, ...
                'BoxFaceAlpha', 0.24, ...
                'WhiskerLineColor', high_color, ...
                'MarkerStyle', 'none', ...
                'BoxWidth', 0.22, ...
                'LineWidth', 1.25);


            %% 低注意能力组箱线图：蓝色
            boxchart( ...
                repmat(x_low, length(y_low), 1), ...
                y_low, ...
                'BoxFaceColor', low_color, ...
                'BoxFaceAlpha', 0.24, ...
                'WhiskerLineColor', low_color, ...
                'MarkerStyle', 'none', ...
                'BoxWidth', 0.22, ...
                'LineWidth', 1.25);


            %% 散点横向抖动
            jitter_high = ...
                (rand(length(y_high),1) - 0.5) * 0.12;

            jitter_low = ...
                (rand(length(y_low),1) - 0.5) * 0.12;


            %% Version 1：散点无边框
            if version_id == 1

                scatter( ...
                    x_high + jitter_high, ...
                    y_high, ...
                    16, ...
                    high_color, ...
                    'filled', ...
                    'MarkerFaceAlpha', 0.70, ...
                    'MarkerEdgeColor', 'none');

                scatter( ...
                    x_low + jitter_low, ...
                    y_low, ...
                    16, ...
                    low_color, ...
                    'filled', ...
                    'MarkerFaceAlpha', 0.70, ...
                    'MarkerEdgeColor', 'none');


            %% Version 2：散点带深红/深蓝边框
            else

                scatter( ...
                    x_high + jitter_high, ...
                    y_high, ...
                    17, ...
                    high_color, ...
                    'filled', ...
                    'MarkerFaceAlpha', 0.68, ...
                    'MarkerEdgeColor', high_edge_color, ...
                    'MarkerEdgeAlpha', 0.90, ...
                    'LineWidth', 0.55);

                scatter( ...
                    x_low + jitter_low, ...
                    y_low, ...
                    17, ...
                    low_color, ...
                    'filled', ...
                    'MarkerFaceAlpha', 0.68, ...
                    'MarkerEdgeColor', low_edge_color, ...
                    'MarkerEdgeAlpha', 0.90, ...
                    'LineWidth', 0.55);
            end


            %% 当前发育组最高个体值，用于放星号
            group_data_top(g) = max([y_high; y_low]);

        end


        %% 高注意能力组四阶段发展轨迹：红色
        high_line = errorbar( ...
            x_high_all, ...
            mean_high(k,:), ...
            sem_high(k,:), ...
            '-o', ...
            'Color', high_color, ...
            'MarkerFaceColor', high_color, ...
            'MarkerEdgeColor', high_edge_color, ...
            'LineWidth', 2.2, ...
            'MarkerSize', 6.5, ...
            'CapSize', 7);


        %% 低注意能力组四阶段发展轨迹：蓝色
        low_line = errorbar( ...
            x_low_all, ...
            mean_low(k,:), ...
            sem_low(k,:), ...
            '-o', ...
            'Color', low_color, ...
            'MarkerFaceColor', low_color, ...
            'MarkerEdgeColor', low_edge_color, ...
            'LineWidth', 2.2, ...
            'MarkerSize', 6.5, ...
            'CapSize', 7);


        %% 每个发育组内高低注意差异显著时标星号
        star_offset = 0.055 * y_range;

        for g = 1:n_group

            if ttest_q(k,g) < 0.05

                star_y = group_data_top(g) + star_offset;

                text( ...
                    group_centers(g), ...
                    star_y, ...
                    '*', ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'bottom', ...
                    'FontSize', 19, ...
                    'FontWeight', 'bold', ...
                    'Color', 'k');
            end
        end


        %% 坐标轴设置
        lower_limit = y_min - 0.12 * y_range;
        upper_limit = max(group_data_top) + 0.17 * y_range;

        xlim([0.55, n_group + 0.45]);
        ylim([lower_limit, upper_limit]);

        xticks(group_centers);
        xticklabels(group_names);

        ax = gca;
        ax.TickLabelInterpreter = 'none';
        ax.FontSize = 10;
        ax.LineWidth = 1;
        ax.Box = 'off';
        ax.TickDir = 'out';

        xlabel('Developmental group');
        ylabel('Covariate-adjusted MCCAR loading');

        title( ...
            modality_names{k}, ...
            'FontSize', 13, ...
            'FontWeight', 'bold');


        %% 图例只放在EEG panel
        if k == 1
            legend( ...
                [high_line, low_line], ...
                {'High attentional ability', ...
                 'Low attentional ability'}, ...
                'Location', 'best', ...
                'Box', 'off');
        end
    end


    %% 总标题
    sgtitle( ...
        ['Developmental trajectories of MCCAR loadings ' ...
         'in high- and low-attention groups'], ...
        'FontWeight', 'normal', ...
        'FontSize', 15);


    %% 保存当前版本
    output_png = fullfile( ...
        outputPath, ...
        ['three_modalities_boxscatter_' ...
         version_names{version_id}, '.png']);

    output_pdf = fullfile( ...
        outputPath, ...
        ['three_modalities_boxscatter_' ...
         version_names{version_id}, '.pdf']);

    exportgraphics( ...
        fig_boxscatter, ...
        output_png, ...
        'Resolution', 300);


    exportgraphics( ...
        fig_boxscatter, ...
        output_pdf, ...
        'Resolution', 300);

    fprintf('\n已保存：\n%s\n', output_png);
     fprintf('\n已保存：\n%s\n', output_pdf);
end