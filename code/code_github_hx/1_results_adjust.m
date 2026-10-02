%% directions
clc;clear
%%adjust first
load '/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/1_30_r.mat'
load '/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/1_30_a.mat'
load '/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/1_30_s.mat'
outpath='/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_adjust_direction/'
r_f_s

v = cell2mat(r_f_s);
mask = [ v(1) > 0, v(2) > 0, v(3) > 0 ];   % 逻辑是否符合
ind_adjust = find(~mask)
% 对 a 反符号
for i = ind_adjust
    A = eval(['a' num2str(i)]);
    A = -A;
    eval(['a' num2str(i) ' = A;']);
end
save([outpath,'1_30_a.mat'],'a1','a2','a3')

% 对 s 取负号
for i =ind_adjust
    A = eval(['s' num2str(i)]);
    A = -A;
    eval(['s' num2str(i) ' = A;']);
end
save([outpath,'1_30_s.mat'],'s1','s2','s3')




% eeg
clear;clc
load '/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all/results_2/results_plot/label_150.mat'

inputPath='/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_adjust_direction/';
inputPath2='/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/';

outpath='/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_plot/'

M=30;
load ([inputPath '1_' num2str(M),'_s.mat'])
load ([inputPath2 '1_' num2str(M),'_indmat.mat'])

for i=1
    s= eval(['s' num2str(i)]);
    s=s(indmat_f_s{i},:);
    feature_z=(s-mean(s))/std(s);
    % EEG feature selection: |Z| > 2; keep full Z map for downstream use.
    sig_roi_name=label_150(abs(feature_z) > 2)
    index=find(abs(feature_z) > 2)
    save([outpath,['s' num2str(i)],'_','1_' num2str(M),'_.mat'],'sig_roi_name','feature_z','index')
end
feature_z(index)'






% mri
clear;clc
load('BN_name.mat')
inputPath='/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_adjust_direction/';
inputPath2='/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/';

outpath='/path/to/project/Project_att2_CR_mCCAR_HX_eeg_power_all_raw/results_1/results_plot/'

for M=30%[25,50,75,100]
    load ([inputPath '1_' num2str(M),'_s.mat'])
    load ([inputPath2 '1_' num2str(M),'_indmat.mat'])
    for i=2:3
        s= eval(['s' num2str(i)]);
        s=s(indmat_f_s{i},:);
        feature_z_raw=(s-mean(s))/std(s);
        feature_z=(s-mean(s))/std(s);
        feature_z(abs(feature_z) <= 2) = 0;
        sig_roi_name=BN_name(abs(feature_z) > 2,:)
        index=find(abs(feature_z) > 2)
        save([outpath,['s' num2str(i)],'_','1_' num2str(M),'_.mat'],'sig_roi_name','feature_z','index','feature_z_raw')
    end
end




