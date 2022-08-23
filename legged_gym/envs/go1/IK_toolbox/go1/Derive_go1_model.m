clear; clc;

syms q_hip q_thigh q_calf 'real'   % states

syms leg_offset_x leg_offset_y thigh_y thigh_length calf_length

P_hip = sym([leg_offset_x;leg_offset_y; 0]);%%% hip->orirgin
P_thigh = sym([0;thigh_y; 0]);              %%% thigh->hip
P_calf = sym([0;0;thigh_length]);           %%% calf->thigh
P_feet = sym([0;0;calf_length]);            %%% feet->calf

%%% Rx, Ry, Rz;
R_hip = [1,0,0;0,cos(q_hip), -sin(q_hip); 0, sin(q_hip), cos(q_hip)]; 

R_thigh = [cos(q_thigh),0,sin(q_thigh);0,1,0;-sin(q_thigh),0,cos(q_thigh)]; 

R_calf = [cos(q_calf),0,sin(q_calf);0,1,0;-sin(q_calf),0,cos(q_calf)]; 


%%%% Kinematics:

T_thigh_origin = [R_hip,P_hip;0,0,0,1];

x = T_thigh_origin * [P_thigh;1];
pos_thigh = x(1:3,:);

T_calf_hip = [R_thigh,P_thigh;0,0,0,1];
x = T_thigh_origin * T_calf_hip*[P_calf;1];
pos_calf = x(1:3,:);

T_feet_calf = [R_calf,P_calf;0,0,0,1];
x = T_thigh_origin * T_calf_hip*T_feet_calf*[P_feet;1];
pos_feet = x(1:3,:);

Jaco = jacobian(pos_feet,[q_hip q_thigh q_calf]);
%%%% Generate a function for computing the forward kinematics:
% syms empty 'real'  %fixes a bug in matlabFunction related to vectorization
% p1(2) = p1(2) + empty;
% dp1(2) = dp1(2) + empty;
matlabFunction(pos_thigh,pos_calf,pos_feet,Jaco,...
    'file','autoGen_go1Kinematics.m',...
    'vars',{q_hip q_thigh q_calf leg_offset_x leg_offset_y thigh_y thigh_length calf_length},...
    'outputs',{'pos_thigh','pos_calf','pos_feet','Jaco'});

% 
% %%% Generate a function to computing inverse kinematics
% syms feetx feety feetz 'real'
% xxx = pos_feet(1);
% 
% equal1 = pos_feet(1) - feetx;
% equal2 = pos_feet(2) - feety;
% equal3 = pos_feet(3) - feetz;
% sol = solve(equal1,equal2,equal3, q_hip, q_thigh, q_calf);
% q_hip_desired = sol.q_hip;
% q_thigh_desired = sol.q_thigh;
% q_calf_desired = sol.q_calf;
% matlabFunction(q_hip_desired,q_thigh_desired,q_calf_desired,...
%     'file','autoGen_go1_inverse_Kinematics.m',...
%     'vars',{feetx feety feetz leg_offset_x leg_offset_y thigh_y thigh_length calf_length},...
%     'outputs',{'q_hip_desired','q_thigh_desired','q_calf_desired'});



