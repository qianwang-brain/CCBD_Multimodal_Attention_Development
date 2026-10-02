clear; clc;

load ('/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/data_selection/data2.mat','c','ref','TIV');

inputPath = '/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/results_1/results_adjust_direction/';
outPath='/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/results_1/results_plot/corr_plot/'
matFile = fullfile(inputPath, '1_30_a.mat');
S = load(matFile, 'a1', 'a2', 'a3');

A = {S.a1, S.a2, S.a3};
xlab_list = {'EEG', 'GMV', 'FCS'};
file_tags = {'eeg', 'GMV', 'FCS'};

c_t = [];
c_p = [];
c_beta=[];
c_se=[];
ci_low=[];
ci_high=[];

if ~exist(inputPath,'dir'), mkdir(inputPath); end
for k = 1:3
    x0  = A{k}(:,1);
    age = c(:,1);
    sex = c(:,2);

    index_used = ~isnan(ref) & ~isnan(x0) & ~isnan(age) & ~isnan(sex);
    if k == 2
        index_used = index_used & isfinite(TIV(:,1));
    end

    y   = ref(index_used);
    x   = x0(index_used);
    age = age(index_used);
    sex = sex(index_used);

    %% centered predictors
    x_c   = x - mean(x);
    age_c = age - mean(age);
    sex_c = sex - mean(sex);

    %% GLM: y ~ sex + x + age + x*age
    X = [sex_c, x_c, age_c, x_c .* age_c];
    if k == 2
        tiv_use = TIV(index_used,1);
        X = [X(:,1:3), tiv_use - mean(tiv_use), X(:,4)];
    end
    [b, ~, stats] = glmfit(X, y, 'normal', 'link', 'identity');

    c_t = [c_t, stats.t(end)];
    c_p = [c_p, stats.p(end)];
    c_beta = [c_beta, stats.beta(end)];
    c_se = [c_se, stats.se(end)];
    ci_low  =[ci_low, stats.beta(end) - 1.96 * stats.se(end)];
    ci_high = [ci_high,stats.beta(end) + 1.96 * stats.se(end)];


    %% representative age levels
    age_sd   = std(age_c);
    age_low  = -age_sd;
    age_high =  age_sd;

    %% x range for prediction
    x_range   = linspace(min(x_c), max(x_c), 200)';
    sex_fixed = 0;   % sex is centered, so 0 = mean sex value

    %% prediction design matrices
    X_low  = [repmat(sex_fixed, size(x_range)), x_range, repmat(age_low,  size(x_range)), x_range .* age_low];
    X_high = [repmat(sex_fixed, size(x_range)), x_range, repmat(age_high, size(x_range)), x_range .* age_high];

    if k == 2
        X_low = [X_low(:,1:3), zeros(size(x_range)), X_low(:,4)];
        X_high = [X_high(:,1:3), zeros(size(x_range)), X_high(:,4)];
    end

    %% fitted values
    y_low  = glmval(b, X_low,  'identity');
    y_high = glmval(b, X_high, 'identity');

    %% 95% CI for fitted mean
    covb = stats.covb;

    Xd_low  = [ones(size(x_range)), X_low];
    Xd_high = [ones(size(x_range)), X_high];

    se_low  = sqrt(diag(Xd_low  * covb * Xd_low'));
    se_high = sqrt(diag(Xd_high * covb * Xd_high'));

    ci_low_low   = y_low  - 1.96 * se_low;
    ci_low_high  = y_low  + 1.96 * se_low;
    ci_high_low  = y_high - 1.96 * se_high;
    ci_high_high = y_high + 1.96 * se_high;

    %% colors: muted and publication-style
    col_low   = [0.23 0.45 0.70];
    col_high  = [0.75 0.28 0.28];
    col_pts   = [0.55 0.55 0.55];
    col_axis  = [0.15 0.15 0.15];

    %% figure
    fig = figure('Color', 'w', 'Position', [100 100 560 430]);
    hold on;

%     %% raw data points (subtle background)
%     scatter(x_c, y, 16, ...
%         'MarkerFaceColor', col_pts, ...
%         'MarkerEdgeColor', 'none', ...
%         'MarkerFaceAlpha', 0.22);

    %% confidence bands
    fill([x_range; flipud(x_range)], [ci_low_low; flipud(ci_low_high)], ...
        col_low, 'FaceAlpha', 0.14, 'EdgeColor', 'none');

    fill([x_range; flipud(x_range)], [ci_high_low; flipud(ci_high_high)], ...
        col_high, 'FaceAlpha', 0.14, 'EdgeColor', 'none');

    %% simple slopes (ONLY two lines)
    p1 = plot(x_range, y_low,  '-', 'Color', col_low,  'LineWidth', 2.2);
    p2 = plot(x_range, y_high, '-', 'Color', col_high, 'LineWidth', 2.2);

    %% labels
    xlabel([xlab_list{k}, ' (centered)'], 'FontSize', 12);
    ylabel('Selective attention', 'FontSize', 12);

    %% legend
    legend([p1, p2], {'Younger age (-1 SD)', 'Older age (+1 SD)'}, ...
        'Location', 'northeast', ...
        'Box', 'off', ...
        'FontSize', 10);

    %% axis style
    ax = gca;
    ax.Box = 'off';
    ax.TickDir = 'out';
    ax.LineWidth = 1;
    ax.FontSize = 11;
    ax.XColor = col_axis;
    ax.YColor = col_axis;
    ax.TickLength = [0.015 0.015];
    grid off;

    %% optional: tighten limits a little
    xlim([min(x_range), max(x_range)]);

    %% remove title for publication style
    title('');

    %% export
    filename_pdf = fullfile(inputPath, sprintf('interaction_%s_high_impact_style.pdf', file_tags{k}));
    filename_png = fullfile(inputPath, sprintf('interaction_%s_high_impact_style.png', file_tags{k}));
    exportgraphics(fig, filename_pdf, 'ContentType', 'vector');
    exportgraphics(fig, filename_png, 'Resolution', 600);

    close(fig);
end

save('results_interaction_age_atten.mat', 'c_t', 'c_p','c_beta','c_se','ci_low','ci_high');
