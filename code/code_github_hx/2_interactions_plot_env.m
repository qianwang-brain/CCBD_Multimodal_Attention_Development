%%PCA from env
%% plot significant env x age interaction only
clear; clc;

%% load original data
load('/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all/data_selection/attention_all/data2.mat', ...
    'c','sub_final','TIV');

matFile = fullfile('/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_adjust_direction/', ...
    '1_30_a.mat');
S = load(matFile, 'a1', 'a2', 'a3');

A = {S.a1, S.a2, S.a3};
xlab_list = {'EEG', 'GMV', 'FCS'};
file_tags = {'eeg', 'GMV', 'FCS'};

%% load env data
load('/path/to/project/Envrion/results_PCA/results_env_6factors.mat', ...
    'PC1_table');
env_name = [ ...
    "SES", ...
    "Parenting", ...
    "SchoolResource", ...
    "SchoolEcology", ...
    "Lifestyle", ...
    "Learning"];

env_sub = string(PC1_table.env_sub(:));
env = PC1_table{:, env_name};
fprintf('环境PC1数据：%d人 × %d个维度。\n', ...
    size(env,1), size(env,2));

%% match subjects
sub_ids = string(sub_final(:));
assert(numel(unique(sub_ids)) == numel(sub_ids) && ...
    numel(unique(env_sub)) == numel(env_sub), 'Duplicate participant IDs.');
[~, ind1, ind_env] = intersect(sub_ids, env_sub, 'stable');
assert(~isempty(ind1), 'No matched environmental participants.');
assert(isequal(sub_ids(ind1),env_sub(ind_env)), 'Participant ID mismatch.');
env = env(ind_env,:);
TIV = TIV(ind1,1);

age = c(:,1);
sex = c(:,2);

age = age(ind1);
sex = sex(ind1);

for k = 1:3
    A{k} = A{k}(ind1,:);
end

%% load interaction results
load('results_interaction_age_env2.mat', 'results', 'sig_env_name');

outputPath = '/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_plot/corr_plot/interaction_age_env/';
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end
%% colors
col_low  = [0.2 0.4 0.8];   % blue
col_high = [0.8 0.3 0.3];   % red
% col_low  = [0.2 0.6 0.3];   % 绿色（low env）
% col_high = [0.5 0.2 0.7];   % 紫色（high env）
%
%% plot only significant interaction
for k = 1:3

    sig_names_k = sig_env_name{k};

    if isempty(sig_names_k)
        continue
    end

    for s = 1:length(sig_names_k)

        env_idx = find(strcmp(env_name, sig_names_k{s}));

        if isempty(env_idx)
            continue
        end

        y0 = A{k}(:,1);
        x0 = env(:,env_idx);

        index_used = ~isnan(y0) & ~isnan(x0) & ~isnan(age) & ~isnan(sex);
        if k == 2
            index_used = index_used & isfinite(TIV);
        end

        y = y0(index_used);
        x = x0(index_used);
        age_use = age(index_used);
        sex_use = sex(index_used);

        %% centered predictors
        x_c = x - mean(x);
        age_c = age_use - mean(age_use);
        sex_c = sex_use - mean(sex_use);

        %% fit GLM: y ~ sex + x + age + x*age
        X = [sex_c, x_c, age_c, x_c .* age_c];
        if k == 2
            tiv_use = TIV(index_used);
            X = [X, tiv_use - mean(tiv_use)];
        end
        [b, ~, stats] = glmfit(X, y, 'normal', 'link', 'identity');

        p_int = stats.p(5);
        beta_int = b(5);

        %% age as x-axis, env as moderator lines
        env_low  = mean(x_c) - std(x_c);
        env_high = mean(x_c) + std(x_c);

        age_fit = linspace(min(age_c), max(age_c), 100)';

        %% prediction + CI
        n = length(age_fit);

        X_low = [ ...
            zeros(n,1), ...
            repmat(env_low,n,1), ...
            age_fit, ...
            env_low .* age_fit];

        X_high = [ ...
            zeros(n,1), ...
            repmat(env_high,n,1), ...
            age_fit, ...
            env_high .* age_fit];

        if k == 2
            X_low = [X_low, zeros(n,1)];
            X_high = [X_high, zeros(n,1)];
        end

        [y_low_env, ci_low] = glmval(b, X_low, 'identity', stats);
        [y_high_env, ci_high] = glmval(b, X_high, 'identity', stats);

        if size(ci_low,2) == 1
            ci_low_lower = y_low_env - ci_low;
            ci_low_upper = y_low_env + ci_low;

            ci_high_lower = y_high_env - ci_high;
            ci_high_upper = y_high_env + ci_high;
        else
            ci_low_lower = ci_low(:,1);
            ci_low_upper = ci_low(:,2);

            ci_high_lower = ci_high(:,1);
            ci_high_upper = ci_high(:,2);
        end

        %% figure
        figure('Color','w','Position',[200 200 600 500]);
        hold on;

        %% confidence bands
        fill([age_fit; flipud(age_fit)], ...
            [ci_low_lower; flipud(ci_low_upper)], ...
            col_low, ...
            'FaceAlpha', 0.18, ...
            'EdgeColor', 'none');

        fill([age_fit; flipud(age_fit)], ...
            [ci_high_lower; flipud(ci_high_upper)], ...
            col_high, ...
            'FaceAlpha', 0.18, ...
            'EdgeColor', 'none');

        %% simple slopes
        p1 = plot(age_fit, y_low_env, ...
            'Color', col_low, ...
            'LineWidth', 2.5);

        p2 = plot(age_fit, y_high_env, ...
            'Color', col_high, ...
            'LineWidth', 2.5);

        %% labels
        xlabel('Age');
        ylabel(xlab_list{k});

        title(sprintf('%s: Age × %s interaction', ...
            xlab_list{k}, sig_names_k{s}));

        legend([p1 p2], {'Low env (-1 SD)', 'High env (+1 SD)'}, ...
            'Location', 'best', ...
            'Box', 'off');

        box off;
        set(gca, 'FontSize', 14, 'LineWidth', 1.2);

        text(0.05, 0.95, ...
            sprintf('\\beta_{int}=%.3f, p=%.4f', beta_int, p_int), ...
            'Units','normalized', ...
            'FontSize', 13, ...
            'VerticalAlignment','top');

        %% save
        save_name = sprintf('%s_%s_age_interaction_CI.pdf', ...
            file_tags{k}, sig_names_k{s});

        saveas(gcf, fullfile(outputPath, save_name));
        close;

    end
end












% %%Structural
% %% plot significant env x age interaction only
% 
% clear; clc;
% %% load original data
% load('/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all/data_selection/attention_all/data2.mat', ...
%     'c','sub_final','TIV');
% 
% matFile = fullfile('/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_adjust_direction/', ...
%     '1_30_a.mat');
% S = load(matFile, 'a1', 'a2', 'a3');
% 
% A = {S.a1, S.a2, S.a3};
% xlab_list = {'EEG', 'GMV', 'FCS'};
% file_tags = {'eeg', 'GMV', 'FCS'};
% 
% %% load env data
% load('/path/to/project/Envrion/results_PCA/results_env_6factors.mat', ...
%     'PC1_table');
% env_name = [ ...
%     "SES", ...
%     "Parenting", ...
%     "SchoolResource", ...
%     "SchoolEcology", ...
%     "Lifestyle", ...
%     "Learning"];
% 
% env_sub = string(PC1_table.env_sub(:));
% env = PC1_table{:, env_name};
% fprintf('环境PC1数据：%d人 × %d个维度。\n', ...
%     size(env,1), size(env,2));
% 
% %% match subjects
% [~, ind1] = intersect(sub_final, env_sub);
% isequal(sub_final(ind1),env_sub)
% 
% age = c(:,1);
% sex = c(:,2);
% 
% age = age(ind1);
% sex = sex(ind1);
% TIV=TIV(ind1);
% 
% for k = 1:3
%     A{k} = A{k}(ind1,:);
% end
% 
% %% load interaction results
% load('results_interaction_age_env2.mat', 'results', 'sig_env_name');
% 
% outputPath = '/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_plot/corr_plot/interaction_age_env/';
% if ~exist(outputPath, 'dir')
%     mkdir(outputPath);
% end
% %% colors
% col_low  = [0.2 0.4 0.8];   % blue
% col_high = [0.8 0.3 0.3];   % red
% % col_low  = [0.2 0.6 0.3];   % 绿色（low env）
% % col_high = [0.5 0.2 0.7];   % 紫色（high env）
% 
% %% plot only significant interaction
% for k =2
% 
%     sig_names_k = sig_env_name{k};
% 
%     if isempty(sig_names_k)
%         continue
%     end
% 
%     for s = 1:length(sig_names_k)
% 
%         env_idx = find(strcmp(env_name, sig_names_k{s}));
% 
%         if isempty(env_idx)
%             continue
%         end
% 
%         y0 = A{k}(:,1);
%         x0 = env(:,env_idx);
% 
%         index_used = ~isnan(y0) & ~isnan(x0) & ~isnan(age) & ~isnan(sex);
        if k == 2
            index_used = index_used & isfinite(TIV);
        end
% 
%         y = y0(index_used);
%         x = x0(index_used);
%         age_use = age(index_used);
%         sex_use = sex(index_used);
%         TIV_use=TIV(index_used);
% 
%         %% centered predictors
%         x_c = x - mean(x);
%         age_c = age_use - mean(age_use);
%         sex_c = sex_use - mean(sex_use);
%         TIV_c=TIV_use-mean(TIV_use);
%         %% fit GLM: y ~ sex + x + age + x*age
%         X = [sex_c,TIV_c,x_c, age_c, x_c .* age_c];
%         [b, ~, stats] = glmfit(X, y, 'normal', 'link', 'identity');
% 
%         p_int = stats.p(6);
%         beta_int = b(6);
% 
%         %% age as x-axis, env as moderator lines
%         env_low  = mean(x_c) - std(x_c);
%         env_high = mean(x_c) + std(x_c);
% 
%         age_fit = linspace(min(age_c), max(age_c), 100)';
% 
%         %% prediction + CI
%         n = length(age_fit);
% 
%         X_low = [ ...
%             zeros(n,1), ...
%             zeros(n,1),...
%             repmat(env_low,n,1), ...
%             age_fit, ...
%             env_low .* age_fit];
% 
%         X_high = [ ...
%             zeros(n,1), ...
%             zeros(n,1),...
%             repmat(env_high,n,1), ...
%             age_fit, ...
%             env_high .* age_fit];
% 
%         if k == 2
            X_low = [X_low, zeros(n,1)];
            X_high = [X_high, zeros(n,1)];
        end

        [y_low_env, ci_low] = glmval(b, X_low, 'identity', stats);
%         [y_high_env, ci_high] = glmval(b, X_high, 'identity', stats);
% 
%         if size(ci_low,2) == 1
%             ci_low_lower = y_low_env - ci_low;
%             ci_low_upper = y_low_env + ci_low;
% 
%             ci_high_lower = y_high_env - ci_high;
%             ci_high_upper = y_high_env + ci_high;
%         else
%             ci_low_lower = ci_low(:,1);
%             ci_low_upper = ci_low(:,2);
% 
%             ci_high_lower = ci_high(:,1);
%             ci_high_upper = ci_high(:,2);
%         end
% 
%         %% figure
%         figure('Color','w','Position',[200 200 600 500]);
%         hold on;
% 
%         %% confidence bands
%         fill([age_fit; flipud(age_fit)], ...
%             [ci_low_lower; flipud(ci_low_upper)], ...
%             col_low, ...
%             'FaceAlpha', 0.18, ...
%             'EdgeColor', 'none');
% 
%         fill([age_fit; flipud(age_fit)], ...
%             [ci_high_lower; flipud(ci_high_upper)], ...
%             col_high, ...
%             'FaceAlpha', 0.18, ...
%             'EdgeColor', 'none');
% 
%         %% simple slopes
%         p1 = plot(age_fit, y_low_env, ...
%             'Color', col_low, ...
%             'LineWidth', 2.5);
% 
%         p2 = plot(age_fit, y_high_env, ...
%             'Color', col_high, ...
%             'LineWidth', 2.5);
% 
%         %% labels
%         xlabel('Age');
%         ylabel(xlab_list{k});
% 
%         title(sprintf('%s: Age × %s interaction', ...
%             xlab_list{k}, sig_names_k{s}));
% 
%         legend([p1 p2], {'Low env (-1 SD)', 'High env (+1 SD)'}, ...
%             'Location', 'best', ...
%             'Box', 'off');
% 
%         box off;
%         set(gca, 'FontSize', 14, 'LineWidth', 1.2);
% 
%         text(0.05, 0.95, ...
%             sprintf('\\beta_{int}=%.3f, p=%.4f', beta_int, p_int), ...
%             'Units','normalized', ...
%             'FontSize', 13, ...
%             'VerticalAlignment','top');
% 
%         %% save
%         save_name = sprintf('%s_%s_age_interaction_CI.pdf', ...
%             file_tags{k}, sig_names_k{s});
% 
%         saveas(gcf, fullfile(outputPath, save_name));
%         close;
% 
%     end
% end
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
