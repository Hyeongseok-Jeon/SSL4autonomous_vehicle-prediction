clear
clc

% data = load('val_data.mat');
% data = data.valdata;
data = load('train_data.mat');
data = data.traindata;
X = categorical({'Go Straight','LLC','RLC','Right Turn', 'Left Turn', 'NA', '0'});
X = reordercats(X,{'Go Straight','LLC','RLC','Right Turn', 'Left Turn', 'NA', '0'});


ego_num_GS = 0;
ego_num_LLC = 0;
ego_num_RLC = 0;
ego_num_RightTurn = 0;
ego_num_LeftTurn = 0;
ego_num_NA = 0;
ego_num_0 = 0;

sur_num_GS = 0;
sur_num_LLC = 0;
sur_num_RLC = 0;
sur_num_RightTurn = 0;
sur_num_LeftTurn = 0;
sur_num_NA = 0;
sur_num_0 = 0;


x_dist = 0:2.5:125;
dist = zeros(1, length(x_dist)-1);
for i = 1:length(data)
    ego_maneuver = data{i,2};
    sur_maneuver = data{i,4};
    min_dis = data{i,5};
    dist(floor(min_dis*2/5) + 1) = dist(floor(min_dis*2/5) + 1) + 1;
    
    if strcmp(ego_maneuver, 'go_straight')
        ego_num_GS = ego_num_GS + 1;   
    elseif strcmp(ego_maneuver, 'left_lane_change')
        ego_num_LLC = ego_num_LLC + 1;
    elseif strcmp(ego_maneuver, 'right_lane_change')
        ego_num_RLC = ego_num_RLC + 1;
    elseif strcmp(ego_maneuver, 'RIGHT')
        ego_num_RightTurn = ego_num_RightTurn + 1;
    elseif strcmp(ego_maneuver, 'LEFT')
        ego_num_LeftTurn = ego_num_LeftTurn + 1;
    elseif strcmp(ego_maneuver, 'not_defined')
        ego_num_NA = ego_num_NA + 1;
    elseif strcmp(num2str(ego_maneuver), '0')
        ego_num_0 = ego_num_0 + 1;
    end
    
    if strcmp(sur_maneuver, 'go_straight')
        sur_num_GS = sur_num_GS + 1;   
    elseif strcmp(sur_maneuver, 'left_lane_change')
        sur_num_LLC = sur_num_LLC + 1;
    elseif strcmp(sur_maneuver, 'right_lane_change')
        sur_num_RLC = sur_num_RLC + 1;
    elseif strcmp(sur_maneuver, 'RIGHT')
        sur_num_RightTurn = sur_num_RightTurn + 1;
    elseif strcmp(sur_maneuver, 'LEFT')
        sur_num_LeftTurn = sur_num_LeftTurn + 1;
    elseif strcmp(sur_maneuver, 'not_defined')
        sur_num_NA = sur_num_NA + 1;
    elseif strcmp(num2str(sur_maneuver), '0')
        sur_num_0 = sur_num_0 + 1;
    elseif strcmp(sur_maneuver, 'error: no nearby lanes found')
        sur_num_NA = sur_num_NA + 1;
    end
    
end

figure()
histogram('BinEdges',x_dist,'BinCounts',dist)
xlabel('Closest distance between ego and target (m)')
ylabel('Number of instance')
title("distribution of minimum distance btw ego and target")

ego = [ego_num_GS, ego_num_LLC, ego_num_RLC, ego_num_RightTurn, ego_num_LeftTurn, ego_num_NA, ego_num_0];
sur = [sur_num_GS, sur_num_LLC, sur_num_RLC, sur_num_RightTurn, sur_num_LeftTurn, sur_num_NA, sur_num_0];
figure()
b_ego = bar(X, ego);
xtips2 = b_ego.XEndPoints;
ytips2 = b_ego.YEndPoints;
ytips2(1) = ytips2(1)/2;
labels2 = string(fix((b_ego.YData/sum(b_ego.YData))*10000)/100);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
title("ego vehicle's maneuver distribution")

figure()
b_sur = bar(X, sur);
xtips2 = b_sur.XEndPoints;
ytips2 = b_sur.YEndPoints;
ytips2(1) = ytips2(1)/2;
labels2 = string(fix((b_sur.YData/sum(b_sur.YData))*10000)/100);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
title("target vehicle's maneuver distribution")


ego_num_GS + ego_num_LLC + ego_num_RLC + ego_num_RightTurn + ego_num_LeftTurn + ego_num_NA + ego_num_0
sur_num_GS + sur_num_LLC + sur_num_RLC + sur_num_RightTurn + sur_num_LeftTurn + sur_num_NA + sur_num_0
