%% 导出270个环境items及其所属维度
clc;clear
load '/path/to/project/Envrion/env.mat'
env(:,[6:15])=[];
env_name(6:15)=[];%%delete covid

ind_noNAN=all(~isnan(env),2);
env=env(ind_noNAN,:);
env_sub=env_sub(ind_noNAN);


idx_SES = find(contains(env_name, {'教育','职业','拥有物','资源','书','经济'}));
idx_SES(:,[19,20,21])=[];

idx_Parenting = find(contains(env_name, {'教养','亲子','冲突','支持','一致','控制','心理','家庭'}));
idx_Parenting(:,[1:6,23:28,33:37])=[];

idx_SchoolResource = find(contains(env_name, {'教师','教材','教学','生师比','学历','设施'}));
idx_SchoolResource(:,[5,17:23])=[];

idx_SchoolEcology = find(contains(env_name, {'学校周边','安全','卫生','环境','班级','同伴','社区','邻里'}));
idx_SchoolEcology=[idx_SchoolEcology,92,132,138]

idx_Lifestyle = find(contains(env_name, {'睡眠','早餐','屏幕','视频','手机','电视','电脑','游戏'}));
idx_Lifestyle(:,[1:5,13])=[];

idx_Learning = find(contains(env_name, {'课程','学习时间','阅读','音乐','美术','书法','棋','科学'}));
idx_Learning(:,[4:9,18])=[];

group_names = [
    "SES"
    "Parenting"
    "SchoolResource"
    "SchoolEcology"
    "Lifestyle"
    "Learning"
];

group_idx = {
    idx_SES
    idx_Parenting
    idx_SchoolResource
    idx_SchoolEcology
    idx_Lifestyle
    idx_Learning
};

T = table();

for g = 1:numel(group_names)

    % 当前环境维度对应的变量位置
    idx = group_idx{g}(:);

    % 提取完整变量名称
    item_names = string(env_name(idx));
    item_names = item_names(:);

    % 构建当前维度的表
    Tg = table( ...
        idx, ...
        repmat(group_names(g), numel(idx), 1), ...
        item_names, ...
        'VariableNames', ...
        {'source_index', 'domain', 'variable_name'} ...
    );

    T = [T; Tg];
end

%% 按原始变量顺序排列
T = sortrows(T, 'source_index');

%% 添加最终序号
T.item_order = (1:height(T))';
T = movevars(T, 'item_order', 'Before', 'source_index');

%% 基本检查
fprintf('总记录数：%d\n', height(T));
fprintf('唯一变量位置数：%d\n', numel(unique(T.source_index)));
fprintf('唯一变量名称数：%d\n', numel(unique(T.variable_name)));

%% 检查同一变量是否被分到多个维度
[unique_idx, ~, ic] = unique(T.source_index);
counts_idx = accumarray(ic, 1);
duplicate_source_idx = unique_idx(counts_idx > 1);

if isempty(duplicate_source_idx)
    fprintf('未发现同一变量被重复分组。\n');
    duplicate_by_index = T([],:);
else
    warning('发现同一变量位置被分到多个维度。');
    duplicate_by_index = T( ...
        ismember(T.source_index, duplicate_source_idx), :);
    disp(duplicate_by_index);
end

%% 检查相同名称是否重复出现
[unique_names, ~, ic_name] = unique(T.variable_name);
counts_name = accumarray(ic_name, 1);
duplicate_names = unique_names(counts_name > 1);

if isempty(duplicate_names)
    fprintf('未发现重复变量名称。\n');
    duplicate_by_name = T([],:);
else
    warning('发现重复变量名称。');
    duplicate_by_name = T( ...
        ismember(T.variable_name, duplicate_names), :);
    disp(duplicate_by_name);
end

%% 各环境维度的items数量
domain_summary = groupsummary(T, 'domain');

disp('各环境维度items数量：');
disp(domain_summary(:, {'domain','GroupCount'}));

%% 检查是否为270项
if height(T) ~= 270
    warning('当前记录数不是270，而是%d，请检查遗漏或重复分组。', ...
        height(T));
else
    fprintf('检查通过：共270项。\n');
end

%% 输出文件
%writetable(T, 'environment_270_items_by_domain.xlsx');
writetable(T, 'environment_final_used_items_by_domain.csv', ...
    'Encoding', 'UTF-8');

writetable(domain_summary, ...
    'environment_domain_summary.xlsx');

if ~isempty(duplicate_by_index)
    writetable(duplicate_by_index, ...
        'duplicate_items_by_index.xlsx');
end

if ~isempty(duplicate_by_name)
    writetable(duplicate_by_name, ...
        'duplicate_items_by_name.xlsx');
end

fprintf('\n文件已输出到当前MATLAB工作目录。\n');