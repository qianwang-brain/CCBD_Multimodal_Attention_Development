%%6 factors
clear; clc;
load('/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/data_selection/data2.mat', 'c','sub_final','TIV');

inputPath = '/path/to/project/Project_att2_CR_mCCAR_Twotask_eeg_power_all_raw/results_1/results_adjust_direction/';
matFile = fullfile(inputPath, '1_30_a.mat');
S = load(matFile, 'a1', 'a2', 'a3');

A = {S.a1, S.a2, S.a3};
xlab_list = {'EEG', 'GMV', 'FCS'};
file_tags = {'eeg', 'GMV', 'FCS'};


%%intersect with env
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


sub_ids = string(sub_final(:));
assert(numel(unique(sub_ids)) == numel(sub_ids) && ...
    numel(unique(env_sub)) == numel(env_sub), 'Duplicate participant IDs.');
[~,ind1,ind_env] = intersect(sub_ids,env_sub,'stable');
assert(~isempty(ind1), 'No matched environmental participants.');
assert(isequal(sub_ids(ind1),env_sub(ind_env)), 'Participant ID mismatch.');
c=c(ind1,:);
TIV=TIV(ind1,1);
env=env(ind_env,:);
for i=1:3
    A{i} = A{i}(ind1, :);
end

nEnv = size(env,2);
nMod = 3;

beta_sex = nan(nEnv, nMod);
p_sex    = nan(nEnv, nMod);
t_sex    = nan(nEnv, nMod);

beta_x   = nan(nEnv, nMod);
p_x      = nan(nEnv, nMod);
t_x      = nan(nEnv, nMod);

beta_age = nan(nEnv, nMod);
p_age    = nan(nEnv, nMod);
t_age    = nan(nEnv, nMod);

beta_int = nan(nEnv, nMod);
p_int    = nan(nEnv, nMod);
t_int    = nan(nEnv, nMod);
beta_sex_std = nan(nEnv, nMod);
p_sex_std    = nan(nEnv, nMod);

beta_x_std   = nan(nEnv, nMod);
p_x_std      = nan(nEnv, nMod);

beta_age_std = nan(nEnv, nMod);
p_age_std    = nan(nEnv, nMod);

beta_int_std = nan(nEnv, nMod);
t_int_std    = nan(nEnv, nMod);
p_int_std    = nan(nEnv, nMod);

for e=1:size(env,2)
    for k = 1:3
        y0  = A{k}(:,1);
        age = c(:,1);
        sex = c(:,2);
        x0=env(:,e);

        index_used = ~isnan(y0) & ~isnan(x0) & ~isnan(age) & ~isnan(sex);
        if k == 2
            index_used = index_used & isfinite(TIV);
        end

        y   = y0(index_used);
        x   = x0(index_used);
        age = age(index_used);
        sex = sex(index_used);

        %% centered predictors
        x_c   = x - mean(x);
        age_c = age - mean(age);
        sex_c = sex - mean(sex);

        %% =========================
        % raw beta model
        % Y ~ sex + env + age + env*age
        % =========================

        x_c   = x - mean(x);
        age_c = age - mean(age);
        sex_c = sex - mean(sex);

        X = [sex_c, x_c, age_c, x_c .* age_c];
        if k == 2
            tiv_use = TIV(index_used);
            X = [X, tiv_use - mean(tiv_use)];
        end
        [b, ~, stats] = glmfit(X, y, 'normal', 'link', 'identity');

        % raw beta
        beta_sex(e,k) = b(2);
        t_sex(e,k)    = stats.t(2);
        p_sex(e,k)    = stats.p(2);

        beta_x(e,k)   = b(3);
        t_x(e,k)      = stats.t(3);
        p_x(e,k)      = stats.p(3);

        beta_age(e,k) = b(4);
        t_age(e,k)    = stats.t(4);
        p_age(e,k)    = stats.p(4);

        beta_int(e,k) = b(5);
        t_int(e,k)    = stats.t(5);
        p_int(e,k)    = stats.p(5);


        %% =========================
        % standardized beta model
        % standardized Y ~ sex + standardized env + standardized age + standardized env*age
        % =========================

        y_z   = zscore(y);
        x_z   = zscore(x);
        age_z = zscore(age);

        % sex 是二分类变量，一般只中心化，不一定 zscore
        sex_c_z = sex - mean(sex);

        X_z = [sex_c_z, x_z, age_z, x_z .* age_z];
        if k == 2
            X_z = [X_z, zscore(tiv_use)];
        end

        [b_z, ~, stats_z] = glmfit(X_z, y_z, 'normal', 'link', 'identity');

        % standardized beta
        beta_sex_std(e,k) = b_z(2);
        p_sex_std(e,k)    = stats_z.p(2);

        beta_x_std(e,k)   = b_z(3);
        p_x_std(e,k)      = stats_z.p(3);

        beta_age_std(e,k) = b_z(4);
        p_age_std(e,k)    = stats_z.p(4);

        beta_int_std(e,k) = b_z(5);
        t_int_std(e,k)    = stats_z.t(5);
        p_int_std(e,k)    = stats_z.p(5);
    end
end
results = struct();

results.beta_x   = beta_x;
results.p_x      = p_x;

results.beta_age = beta_age;
results.p_age    = p_age;

results.beta_int = beta_int;
results.p_int    = p_int;

results.beta_sex = beta_sex;
results.p_sex    = p_sex;
results.beta_int_std = beta_int_std;
results.t_int_std    = t_int_std;
results.p_int_std    = p_int_std;

results.beta_x_std   = beta_x_std;
results.p_x_std      = p_x_std;

results.beta_age_std = beta_age_std;
results.p_age_std    = p_age_std;

p_env_fdr = nan(size(results.p_int));
ind={};sig_env_name={};
for i = 1:3
    p_env_fdr(:,i) = mafdr(results.p_int(:,i), 'BHFDR', true);
    ind{i}=find(p_env_fdr(:,i)<0.05);
    sig_env_name{i}=env_name(ind{i})';
end
results.p_int_fdr    = p_env_fdr;

save('results_interaction_age_env2.mat','results','sig_env_name');



