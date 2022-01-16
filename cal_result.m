clear; close all;

data = load('data/Records.mat');

for i = 1:100
    isCirculatory(i) = [data.records(i).info.subject.isCircutory];
end

circulatoryIndex = find(isCirculatory == 1);
nonCirculatoryIndex = find(isCirculatory == 0);

S(1) = load('Results/result_1s_intra_L1_0001_L2_0001.mat');
S(2) = load('Results/result_2s_intra_L1_0001_L2_0001.mat');
S(3) = load('Results/result_3s_intra_L1_0001_L2_0001.mat');
S(4) = load('Results/result_4s_intra_L1_0001_L2_0001.mat');


figure;
index1 = 1;
index2 = 1;
subplot(2,4,1)
plot_result(S(1).result.test_ecg(circulatoryIndex(index1),1,:),S(1).result.test_result(circulatoryIndex(index1),1,:),'(I)');
subplot(2,4,2)
plot_result(S(2).result.test_ecg(circulatoryIndex(index1),1,:),S(2).result.test_result(circulatoryIndex(index1),1,:),'(II)');
subplot(2,4,3)
plot_result(S(3).result.test_ecg(circulatoryIndex(index1),1,:),S(3).result.test_result(circulatoryIndex(index1),1,:),'(III)');
subplot(2,4,4)
plot_result(S(4).result.test_ecg(circulatoryIndex(index1),1,:),S(4).result.test_result(circulatoryIndex(index1),1,:),'(IV)');

subplot(2,4,5)
plot_result(S(1).result.test_ecg(nonCirculatoryIndex(index2),1,:),S(1).result.test_result(nonCirculatoryIndex(index2),1,:),'(V)');
subplot(2,4,6)
plot_result(S(2).result.test_ecg(nonCirculatoryIndex(index2),1,:),S(2).result.test_result(nonCirculatoryIndex(index2),1,:),'(VI)');
subplot(2,4,7)
plot_result(S(3).result.test_ecg(nonCirculatoryIndex(index2),1,:),S(3).result.test_result(nonCirculatoryIndex(index2),1,:),'(VII)');
subplot(2,4,8)
plot_result(S(4).result.test_ecg(nonCirculatoryIndex(index2),1,:),S(4).result.test_result(nonCirculatoryIndex(index2),1,:),'(VIII)');




T(1) = load('Results/result_train1s_intra_L1_0001_L2_0001.mat');
T(2) = load('Results/result_train2s_intra_L1_0001_L2_0001.mat');
T(3) = load('Results/result_train3s_intra_L1_0001_L2_0001.mat');
T(4) = load('Results/result_train4s_intra_L1_0001_L2_0001.mat');
for j = 1:4
    clear train_r;
    clear test_r;
    clear validation;
    for i = 1:100
        [test_r(i), predict_1,test_1] = cal_measure(S(j).result.test_ecg(i,:,:),S(j).result.test_result(i,:,:));
        [train_r(i),preidct_2,test_2] = cal_measure(T(j).result.train_ecg(i,:,:),T(j).result.train_result(i,:,:));               
        test_ecg = squeeze(T(j).result.train_ecg(i,:,:));
        predict_ecg = squeeze(T(j).result.train_result(i,:,:));
        %plot_dtw(predict_1, test_1,predict_ecg(2,:), test_ecg(2,:),'plot',[18000 18500],'\bf Reconstruction ECG','\bf Reference ECG');
        validation(i) = cal_measure(S(j).result.validation_ecg(i,:,:),S(j).result.validation_result(i,:,:));
    end   
    train_result(j) = cal_all(train_r);
    test_result(j) = cal_all(test_r);
    validation_result(j) = cal_all(validation);
end
index = 1;
B{1,index} = train_result(1,1).distance_average;
B{2,index} = validation_result(1,1).distance_average;
B{3,index} = test_result(1,1).distance_average;

B{4,index} = train_result(1,2).distance_average;
B{5,index} = validation_result(1,2).distance_average;
B{6,index} = test_result(1,2).distance_average;

B{7,index} = train_result(1,3).distance_average;
B{8,index} = validation_result(1,3).distance_average;
B{9,index} = test_result(1,3).distance_average;

B{10,index} = train_result(1,4).distance_average;
B{11,index} = validation_result(1,4).distance_average;
B{12,index} = test_result(1,4).distance_all_average;
index = 2;
B{1,index} = train_result(1,1).distance_all_average;
B{2,index} = validation_result(1,1).distance_all_average;
B{3,index} = test_result(1,1).distance_all_average;

B{4,index} = train_result(1,2).distance_all_average;
B{5,index} = validation_result(1,2).distance_all_average;
B{6,index} = test_result(1,2).distance_all_average;

B{7,index} = train_result(1,3).distance_all_average;
B{8,index} = validation_result(1,3).distance_all_average;
B{9,index} = test_result(1,3).distance_all_average;

B{10,index} = train_result(1,4).distance_all_average;
B{11,index} = validation_result(1,4).distance_all_average;
B{12,index} = test_result(1,4).distance_all_average;





function [result, predict_all, test_all] = cal_measure(test_ecg,predict_ecg)
    test_ecg = squeeze(test_ecg);
    predict_ecg = squeeze(predict_ecg);
    [m,n] = size(test_ecg);
    test_all = [];
    predict_all = [];
    for i = 1:m
        [r(i), r_aligned(i), rmse(i), rmse_aligned(i), distance(i)] = cal_sore(test_ecg(i,:), predict_ecg(i,:));
        test_all = [test_all(:)' test_ecg(i,:)];
        predict_all = [predict_all(:)' predict_ecg(i,:)];
    end
    test_all = double(test_all);
    predict_all = double(predict_all);
    [result.r_all, result.r_all_aligned, result.rmse_all, result.rmse_all_aligned, result.distance_all] = cal_sore(test_all, predict_all);
    [~,r_test_index] = pan_tompkin(test_all,125,0);
    [~,r_predict_index] = pan_tompkin(predict_all,125,0);
    
    result.r = r;
    result.r_aligned = r_aligned;
    result.rmse = rmse;
    result.rmse_aligned = rmse_aligned;
    result.distance = distance;
    result.distance_average = distance/(n/125);
    result.distance_all_average = result.distance_all/(length(test_all)/125);
    result.ratio = length(r_predict_index)/length(r_test_index);
    % [distance, ix, iy] = dtw(query, reference);
end

function plot_result(test,predict,title_text)
    t = (1:length(test))/125;
    r = corr(test(:),predict(:));
    rmse = sqrt(mean((test(:)' - predict(:)').^2)); 
    plot(t,test(:),'k');
    hold on;
    plot(t,predict(:),'r');
    xlim auto;
    data = test(:);
    distance = dtw(test(:),predict(:));
    min_value = min(data) - 0.1*(max(data)-min(data));
    max_value = max(data) + 0.4*(max(data)-min(data));
    data1 = max(data) + 0.1*(max(data)-min(data));
    data2 = max(data) + 0.2*(max(data)-min(data));
    data3 = max(data) + 0.3*(max(data)-min(data));
    ylim([min_value max_value]);
    text(t(round(length(t)/4)), data2, ['\it r \rm = ' num2str(r,'%.3f')]);
    text(t(round(length(t)/4)), data1, ['\it rmse \rm = ' num2str(rmse,'%.3f')]);
    text(t(round(length(t)/4)), data3, ['\it d \rm = ' num2str(distance,'%.3f')]);
    xlabel('Time (s)');
    ylabel('ECG (mV)');
    title(title_text );
end

function result = cal_all(data)
    result.r_stitched = cal_mean_std([data(:).r_all]);
    result.r_stitched_aligned = cal_mean_std([data(:).r_all_aligned]);
    result.rmse_stitched = cal_mean_std([data(:).rmse_all]);
    result.rmse_stitched_aligned = cal_mean_std([data(:).rmse_all_aligned]);
    result.r = cal_mean_std([data(:).r;]);
    result.r_aligned = cal_mean_std([data(:).r_aligned;]);
    result.rmse = cal_mean_std([data(:).rmse;]);
    result.rmse_aligned = cal_mean_std([data(:).rmse_aligned;]);
    result.distance = cal_mean_std([data(:).distance;]);
    result.distance_stitch = cal_mean_std([data(:).distance_all;]);
    result.ratio = cal_mean_std([data(:).ratio;]);
    result.distance_average = cal_mean_std([data(:).distance_average;]);
    result.distance_all_average = cal_mean_std([data(:).distance_all_average;]);
    
end

function result = cal_mean_std(data)
    result = [num2str(mean(data),'%.3f') '±' num2str(std(data),'%.3f')];
end


function [r, r_aligned, rmse, rmse_aligned, distance,lag] = cal_sore(test_ecg, predict_ecg)
        r = corr(test_ecg(:), predict_ecg(:));
        [c , lags] = xcorr(test_ecg, predict_ecg, 30);
        [~,lag_i] = max(c);
        lag = lags(lag_i);
        if lag <0
            test_ecg_aligned = test_ecg(1:end-abs(lag));
            predict_ecg_aligned = predict_ecg(abs(lag)+1:end);
        else
            predict_ecg_aligned = predict_ecg(1:end-abs(lag));
            test_ecg_aligned = test_ecg(abs(lag)+1:end);
        end
        r_aligned = corr(test_ecg_aligned(:), predict_ecg_aligned(:));
        % 
        [distance, ix, iy] = dtw(predict_ecg, test_ecg);
%         plot(ix,iy);
        rmse = sqrt(mean((test_ecg(:)' - predict_ecg(:)').^2));  
        rmse_aligned = sqrt(mean((test_ecg_aligned(:)' - predict_ecg_aligned(:)').^2));  
end


function [distance] = plot_dtw(query, reference,query_2, reference_2, type,window,title_query,title_reference)
        [distance, ix, iy] = dtw(query, reference);
        if strcmp(type,'plot')
            figure;
            set(gcf,'position',[100 100 800 500])
            [distance, ix, iy] = dtw(query_2, reference_2);
            t = 1:length(query_2);
            ax(1) = subplot(4,2,1);
            plot(reference_2,t);
            ylim([0 length(query_2)]);
            set(ax(1),'XDir','reverse');
            ylabel(title_reference);
            ax(2) = subplot(4,2,2);       
            plot(ix,iy);
            xlim([0 length(query_2)]);
            ylim([0 length(query_2)]);
            title(['$$d = ' num2str(distance,'%.3f') '$$'],'interpreter','latex');
            ax(3) = subplot(4,2,3);
            plot(t,query_2);
            xlim([0 length(query_2)]);
            xlabel(title_query);
            
            [distance, ix, iy] = dtw(query, reference);
            t = 1:length(query);
            ax(4) = subplot(4,2,4);
            plot(reference,t);
            ylim([0 length(query)]);
            set(ax(4),'XDir','reverse');
            ylabel(title_reference);
            ax(5) = subplot(4,2,5);       
            plot(ix,iy);
            xlim([0 length(query)]);
            ylim([0 length(query)]);
            pos = [window(1) window(1)  window(2)-window(1) window(2)-window(1)];
            rectangle('Position',pos)
            title(['$$d = ' num2str(distance,'%.3f') ', \bar{d} = ' num2str(distance/224,'%.3f') '$$'],'interpreter','latex');
            ax(6) = subplot(4,2,6);
            plot(t,query);
            xlim([0 length(query)]);
            xlabel(title_query);
            ax(7) = subplot(4,2,7);
            plot(ix,iy);
            xlim(window);
            ylim(window);
            set( gca, 'XTick', [], 'YTick', [] );
            
            set(ax(1),'position',[0.1, 0.7,0.15, 0.25])
            set(ax(2),'position',[0.3, 0.7,0.65, 0.25])
            set(ax(3),'position',[0.3, 0.58,0.65, 0.1])
            
            set(ax(4),'position',[0.05, 0.2,0.12, 0.3])
            set(ax(5),'position',[0.2, 0.2,0.55, 0.3])
            set(ax(6),'position',[0.2, 0.05,0.55, 0.13])
            set(ax(7),'position',[0.78, 0.3,0.2, 0.1])
            
            
        end
end

function [distance] = plot_dtw2(query, reference,type,title_query,title_reference)
        [distance, ix, iy] = dtw(query, reference);
        if strcmp(type,'plot')
            figure;
            set(gcf,'position',[100 100 800 500])
            t = 1:length(query);
            ax(1) = subplot(2,2,1);
            plot(reference,t);
            ylim([0 length(query)]);
            set(ax(1),'XDir','reverse');
            ylabel(title_reference);
            ax(2) = subplot(2,2,2);       
            plot(ix,iy);
            xlim([0 length(query)]);
            ylim([0 length(query)]);
            title(['$$ d = ' num2str(distance,'%.3f') '$$'],'interpreter','latex');
            ax(3) = subplot(2,2,3);
            plot(t,query);
            xlim([0 length(query)]);
            xlabel(title_query);
            
            set(ax(1),'position',[0.1, 0.3,0.15, 0.65])
            set(ax(2),'position',[0.3, 0.3,0.65, 0.65])
            set(ax(3),'position',[0.3, 0.1,0.65, 0.1])
        end
end