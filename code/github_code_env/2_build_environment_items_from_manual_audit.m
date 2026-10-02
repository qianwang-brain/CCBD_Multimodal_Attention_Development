%% After manual check,recorrect the env

clear;
clc;

%% 1. 路径
input_mat = '/path/to/project/Envrion/env.mat';
audit_csv = fullfile(pwd, 'item_direction_audit_manual_check.csv');
output_dir = '/path/to/project/Envrion/manual_audit_processed';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% 2. 读取env.mat，删除COVID变量
load(input_mat, 'env', 'env_name', 'env_sub');

env(:, 6:15) = [];
env_name(6:15) = [];

env_name = strtrim(string(env_name(:)));
env_sub_final = env_sub(:);
n_subjects = size(env, 1);

if size(env, 2) ~= numel(env_name)
    error('env列数和env_name数量不一致。');
end
if numel(env_sub_final) ~= n_subjects
    error('env_sub人数和env行数不一致。');
end

fprintf('删除COVID列后：env = %d人 x %d列。\n', size(env,1), size(env,2));

%% 3. 读取人工审核CSV
T = readtable(audit_csv, 'Encoding', 'UTF-8', ...
    'VariableNamingRule', 'preserve');

if ~isnumeric(T.item_order)
    T.item_order = str2double(string(T.item_order));
end
if ~isnumeric(T.source_index)
    T.source_index = str2double(string(T.source_index));
end

T.variable_name = strtrim(string(T.variable_name));
T.domain = strtrim(string(T.domain));
T.proposed_action = lower(strtrim(string(T.proposed_action)));
T.proposed_recoding_rule = string(T.proposed_recoding_rule);
T = sortrows(T, 'item_order');

if height(T) ~= 131
    error('人工审核CSV应有131项，当前为%d项。', height(T));
end
if numel(unique(T.item_order)) ~= height(T)
    error('item_order存在重复。');
end
if numel(unique(T.variable_name)) ~= height(T)
    error('variable_name存在重复。');
end

allowed_actions = ["keep","reverse","recode","exclude_review","delete"];
if any(~ismember(T.proposed_action, allowed_actions))
    error('proposed_action中存在未知操作。');
end

%% 4. 根据完整变量名精确匹配env_name
% matched_col = nan(height(T), 1);
% 
% for i = 1:height(T)
%     hit = find(env_name == T.variable_name(i));
% 
%     if numel(hit) ~= 1
%         error('item_order=%d，变量“%s”在env_name中匹配到%d列。', ...
%             T.item_order(i), T.variable_name(i), numel(hit));
%     end
% 
%     matched_col(i) = hit;
% end
% 
% fprintf('131个变量均已按完整名称唯一匹配。\n');
%% 4. 忽略下划线、空格和横线，按文字匹配 env_name

env_name_str = string(env_name(:));
audit_name_str = string(T.variable_name(:));

% 统一大小写并清理首尾空格
env_name_key = lower(strtrim(env_name_str));
audit_name_key = lower(strtrim(audit_name_str));

% 只删除下划线、空白字符和不同形式的横线
% 不再使用 \p{Han}，避免中文被全部删除
pattern = '[_＿\s\-–—－]+';

env_name_key = regexprep(env_name_key, pattern, '');
audit_name_key = regexprep(audit_name_key, pattern, '');

% 先显示前10项，确认中文仍然存在
n_show = min(10, numel(env_name_key));

disp(table( ...
    env_name_str(1:n_show), ...
    env_name_key(1:n_show), ...
    'VariableNames', {'original_env_name','normalized_env_name'}));

% 检查是否仍有空名称
bad_env = find(strlength(env_name_key) == 0);
bad_audit = find(strlength(audit_name_key) == 0);

if ~isempty(bad_env)
    error('env_name中以下位置标准化后为空：%s', ...
        strjoin(string(bad_env), ','));
end

if ~isempty(bad_audit)
    error('审核表中以下item_order标准化后为空：%s', ...
        strjoin(string(T.item_order(bad_audit)), ','));
end

%% 匹配
matched_col = nan(height(T), 1);

for i = 1:height(T)

    % 去掉下划线、空格和横线后，剩余文字完全一致
    hit = find(env_name_key == audit_name_key(i));

    if isempty(hit)
        fprintf('\n未匹配变量：\n');
        fprintf('item_order：%d\n', T.item_order(i));
        fprintf('审核表名称：%s\n', T.variable_name(i));
        fprintf('标准化名称：%s\n', audit_name_key(i));

        error('该变量在env_name中没有匹配结果。');

    elseif numel(hit) > 1
        fprintf('\nitem_order=%d匹配到多个变量：\n', T.item_order(i));

        for j = 1:numel(hit)
            fprintf('  env第%d列：%s\n', ...
                hit(j), env_name_str(hit(j)));
        end

        error('标准化后匹配结果不唯一。');
    end

    matched_col(i) = hit;
end

matched_col = round(matched_col);
T.matched_col = matched_col;

fprintf('%d个变量已忽略下划线、空格和横线后完成唯一匹配。\n', ...
    height(T));



%% 5. 应用确认后的维度调整
original_domain = T.domain;

T.domain(T.item_order == 41) = "Learning";
T.domain(T.item_order == 44) = "Learning";
T.domain(T.item_order == 45) = "Learning";
T.domain(T.item_order == 76) = "Learning";
T.domain(T.item_order == 77) = "Learning";

review_order = [41; 44; 45; 76; 77];
review_row = zeros(numel(review_order), 1);
for i = 1:numel(review_order)
    review_row(i) = find(T.item_order == review_order(i));
end

rationale = [ ...
    "家长教育参与更直接反映家庭学习支持和学习参与"
    "数学教师教学方式属于教学实践和学习环境"
    "数学教学活动属于教学实践和学习参与"
    "班长、副班长或团支书属于学生学习/学校参与角色"
    "班级委员属于学生学习/学校参与角色"];

domain_classification_review = table( ...
    T.item_order(review_row), T.source_index(review_row), ...
    T.variable_name(review_row), original_domain(review_row), ...
    T.domain(review_row), rationale, ...
    'VariableNames', {'item_order','source_index','variable_name', ...
    'original_domain','applied_domain','rationale'});

%% 6. 四个教师学历人数变量：构造加权平均学历指数
teacher_names = [ ...
    "教师学历_教师学历-大专及以下人数"
    "教师学历_教师学历-本科人数"
    "教师学历_教师学历-硕士人数"
    "教师学历_教师学历-博士人数"];

teacher_row = zeros(4,1);
for i = 1:4
    hit = find(T.variable_name == teacher_names(i));
    if numel(hit) ~= 1
        error('教师学历变量未唯一找到：%s', teacher_names(i));
    end
    teacher_row(i) = hit;
end

if any(T.proposed_action(teacher_row) ~= "exclude_review")
    error('四个教师学历变量必须标记为exclude_review。');
end
if any(T.proposed_action == "exclude_review" & ...
        ~ismember((1:height(T))', teacher_row))
    error('CSV中存在教师学历四项以外的exclude_review。');
end

% 顺序：大专及以下、本科、硕士、博士
teacher_counts = env(:, matched_col(teacher_row));

missing_teacher = any(isnan(teacher_counts), 2);
invalid_teacher = any((teacher_counts < 0 | ...
    teacher_counts ~= round(teacher_counts)) & ~isnan(teacher_counts), 2);

teacher_total = sum(teacher_counts, 2);
teacher_education_index = ...
    (teacher_counts(:,1) + ...
     2 .* teacher_counts(:,2) + ...
     3 .* teacher_counts(:,3) + ...
     4 .* teacher_counts(:,4)) ./ teacher_total;

teacher_education_index(missing_teacher | invalid_teacher | teacher_total <= 0) = NaN;

valid_teacher_index = teacher_education_index(~isnan(teacher_education_index));
if any(valid_teacher_index < 1 | valid_teacher_index > 4)
    error('教师学历加权平均指数超出理论范围1至4。');
end

qc_status = repmat("ok", n_subjects, 1);
qc_note = repmat("", n_subjects, 1);
qc_status(missing_teacher) = "missing_component";
qc_note(missing_teacher) = "四项教师人数至少一项缺失";
qc_status(~missing_teacher & teacher_total <= 0) = "nonpositive_total";
qc_note(~missing_teacher & teacher_total <= 0) = "teacher_total小于或等于0";
qc_status(invalid_teacher) = "invalid_count";
qc_note(invalid_teacher) = "教师人数存在负数或非整数";

teacher_education_index_qc = table( ...
    env_sub_final, teacher_counts(:,1), teacher_counts(:,2), ...
    teacher_counts(:,3), teacher_counts(:,4), teacher_total, ...
    teacher_education_index, qc_status, qc_note, ...
    'VariableNames', {'env_sub','college_or_below','bachelor','master', ...
    'doctor','teacher_total','teacher_education_index','qc_status','qc_note'});

%% 7. 按proposed_action直接处理
reverse_3_minus_x = [7, 9];
reverse_5_minus_x = [11,13,18,19,23,61,62,63,64,66,67];
reverse_6_minus_x = [38,74,83,85,89,93,112,115,116,117];
reverse_7_minus_x = [113,118];

% 以下变量只有方向决定，没有可靠理论上下限；直接乘以-1。
reverse_negative_x = [ ...
    1,2,3,15,24,25,26,65,69,70,71,96,97,98, ...
    103,104,105,119,124,125,126,127,128,129,130];

reverse_all = [reverse_3_minus_x, reverse_5_minus_x, ...
    reverse_6_minus_x, reverse_7_minus_x, reverse_negative_x];

csv_reverse = sort(T.item_order(T.proposed_action == "reverse"))';
if ~isequal(sort(reverse_all), csv_reverse)
    error('代码中的reverse列表与CSV不一致。');
end

N_delete = sum(T.proposed_action == "delete");
N_final_expected = height(T) - N_delete - 4 + 1;

env_selected_raw = nan(n_subjects, N_final_expected);
env_oriented_final = nan(n_subjects, N_final_expected);

original_item_order_final = nan(N_final_expected,1);
source_index_final = nan(N_final_expected,1);
domain_final = strings(N_final_expected,1);
variable_name_final = strings(N_final_expected,1);
action_final = strings(N_final_expected,1);
rule_final = strings(N_final_expected,1);
processing_note_final = strings(N_final_expected,1);
derived_from_final = strings(N_final_expected,1);

applied_rule_log = strings(height(T),1);
status_log = strings(height(T),1);

out_col = 0;
teacher_first_order = min(T.item_order(teacher_row));

for i = 1:height(T)
    item_order = T.item_order(i);
    action = T.proposed_action(i);

    % 四个教师学历人数变量由一个派生指标替换。
    if ismember(i, teacher_row)
        status_log(i) = "replaced_by_teacher_index";
        applied_rule_log(i) = "四项教师人数替换为一个加权平均学历指数";

        if item_order == teacher_first_order
            out_col = out_col + 1;
            env_selected_raw(:,out_col) = teacher_education_index;
            env_oriented_final(:,out_col) = teacher_education_index;
            original_item_order_final(out_col) = item_order;
            source_index_final(out_col) = NaN;
            domain_final(out_col) = "SchoolResource";
            variable_name_final(out_col) = "教师学历结构_加权平均学历指数";
            action_final(out_col) = "keep";
            rule_final(out_col) = "大专及以下=1，本科=2，硕士=3，博士=4";
            processing_note_final(out_col) = "四个教师学历人数的加权平均";
            derived_from_final(out_col) = strjoin(teacher_names, ' | ');
        end
        continue;
    end

    % delete不进入最终矩阵。
    if action == "delete"
        status_log(i) = "deleted";
        applied_rule_log(i) = T.proposed_recoding_rule(i);
        continue;
    end

    x_raw = env(:, matched_col(i));
    x_new = x_raw;

    if action == "keep"
        applied_rule = "keep原值";

    elseif action == "reverse"
        if ismember(item_order, reverse_3_minus_x)
            x_new = 3 - x_raw;
            applied_rule = "new=3-old";
        elseif ismember(item_order, reverse_5_minus_x)
            x_new = 5 - x_raw;
            applied_rule = "new=5-old";
        elseif ismember(item_order, reverse_6_minus_x)
            x_new = 6 - x_raw;
            applied_rule = "new=6-old";
        elseif ismember(item_order, reverse_7_minus_x)
            x_new = 7 - x_raw;
            applied_rule = "new=7-old";
        elseif ismember(item_order, reverse_negative_x)
            x_new = -x_raw;
            applied_rule = "new=-old";
        else
            error('item_order=%d没有对应的reverse规则。', item_order);
        end

    elseif action == "recode"
        if item_order ~= 42
            error('发现未实现的recode：item_order=%d。', item_order);
        end

        % 教师职称：1至6保持有序；7=其他，设为NaN。
        unexpected_code = ~isnan(x_raw) & ~ismember(x_raw, 1:7);
        if any(unexpected_code)
            error('教师职称中出现1至7以外的编码。');
        end
        x_new(x_raw == 7) = NaN;
        applied_rule = "教师职称1-6保留，7=其他设为NaN";

    else
        error('未处理的proposed_action：%s。', action);
    end

    out_col = out_col + 1;
    env_selected_raw(:,out_col) = x_raw;
    env_oriented_final(:,out_col) = x_new;

    original_item_order_final(out_col) = item_order;
    source_index_final(out_col) = matched_col(i);
    domain_final(out_col) = T.domain(i);
    variable_name_final(out_col) = T.variable_name(i);
    action_final(out_col) = action;
    rule_final(out_col) = T.proposed_recoding_rule(i);
    processing_note_final(out_col) = applied_rule;
    derived_from_final(out_col) = "";

    applied_rule_log(i) = applied_rule;
    status_log(i) = "included";
end

if out_col ~= N_final_expected
    error('最终变量数错误：实际%d，预期%d。', out_col, N_final_expected);
end

%% 8. 最终变量表
final_item_order = (1:out_col)';

T_final = table( ...
    final_item_order, original_item_order_final, source_index_final, ...
    domain_final, variable_name_final, action_final, rule_final, ...
    processing_note_final, derived_from_final, ...
    'VariableNames', {'final_item_order','original_item_order','source_index', ...
    'domain','variable_name','proposed_action','proposed_recoding_rule', ...
    'processing_note','derived_from'});

env_name_final = variable_name_final;

recoding_log = table( ...
    T.item_order, T.source_index, matched_col, T.domain, T.variable_name, ...
    T.proposed_action, applied_rule_log, status_log, ...
    'VariableNames', {'item_order','csv_source_index','matched_env_column', ...
    'domain','variable_name','proposed_action','applied_rule','status'});

deleted_items = T(T.proposed_action == "delete", ...
    {'item_order','source_index','domain','variable_name','proposed_recoding_rule'});

%% 9. 各维度数量和最终检查
group_names = ["SES";"Parenting";"SchoolResource"; ...
    "SchoolEcology";"Lifestyle";"Learning"];
domain_count = zeros(6,1);

for i = 1:6
    domain_count(i) = sum(T_final.domain == group_names(i));
end

domain_summary_final = table(group_names, domain_count, ...
    'VariableNames', {'domain','item_count'});

if size(env_oriented_final,2) ~= height(T_final)
    error('最终数据列数与T_final行数不一致。');
end
if size(env_oriented_final,1) ~= numel(env_sub_final)
    error('最终数据人数与env_sub人数不一致。');
end
if numel(unique(T_final.variable_name)) ~= height(T_final)
    error('最终变量名存在重复。');
end

fprintf('\n最终环境数据：%d人 x %d个变量。\n', ...
    size(env_oriented_final,1), size(env_oriented_final,2));
disp(domain_summary_final);

%% 10. 保存结果
save(fullfile(output_dir, 'environment_items_oriented_final.mat'), ...
    'env_selected_raw', 'env_oriented_final', 'env_name_final', ...
    'env_sub_final', 'T_final', 'recoding_log', ...
    'domain_summary_final', 'teacher_education_index_qc', '-v7.3');

writetable(T_final, ...
    fullfile(output_dir, 'environment_items_final_by_domain.csv'), ...
    'Encoding', 'UTF-8');
writetable(T_final, ...
    fullfile(output_dir, 'environment_items_final_by_domain.xlsx'));

writetable(domain_summary_final, ...
    fullfile(output_dir, 'environment_domain_summary_final.csv'), ...
    'Encoding', 'UTF-8');
writetable(domain_summary_final, ...
    fullfile(output_dir, 'environment_domain_summary_final.xlsx'));

writetable(recoding_log, ...
    fullfile(output_dir, 'recoding_log.csv'), 'Encoding', 'UTF-8');
writetable(deleted_items, ...
    fullfile(output_dir, 'deleted_items.csv'), 'Encoding', 'UTF-8');
writetable(domain_classification_review, ...
    fullfile(output_dir, 'domain_classification_review.csv'), 'Encoding', 'UTF-8');
writetable(teacher_education_index_qc, ...
    fullfile(output_dir, 'teacher_education_index_qc.csv'), 'Encoding', 'UTF-8');

fprintf('结果已保存到：%s\n', output_dir);
