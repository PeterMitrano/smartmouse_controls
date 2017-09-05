%function dxvec = ode_tracking(t,xvec,avec, bvec)
% evaluate the desired state.
dt=0.01;
tsteps=[0:dt:Tf];
N=size(tsteps,2);
X = zeros(3,N);

% with some initial error
% with no-initial error 
X(:,1)=[1, -3, -2];

Xdes = zeros(3,N);

for i=1:N-1
    xvec = X(:,i);
x = xvec(1);
y = xvec(2);
theta = xvec(3);
theta= wrapTo2Pi(theta);
%theta = theta - 2*pi*floor(theta/(2*pi));


l=1;
t=tsteps(i);
basis = [1; t; t^2; t^3];
dbasis = [0; 1; 2*t; 3*t^2];
ddbasis = [0; 0;2; 6*t];
xdes = avec*basis;
dxdes = avec*dbasis;
ddxdes = avec*ddbasis;

ydes = bvec*basis;
dydes = bvec*dbasis;
ddydes = bvec*ddbasis;

% compute sin(theta_d)

thetades = atan2(ydes, xdes);
Xdes(:,i)= [xdes;ydes;thetades];

% The desired state.
xdes_vec = [xdes; ydes; thetades];

% compute the feedforward in the input.
vf = dxdes*cos(thetades) + dydes*sin(thetades);
dthetades = 1/vf*(ddydes*cos(thetades) - ddxdes*sin(thetades));
wf = dthetades;
% A =  [ 0, 0,  vf*cos(thetades);
%  0, 0, -vf*sin(thetades);
%  0, 0,             0];
% B = [ sin(thetades),                    0;
%  cos(thetades),                    0;
%  tan(deltaf), vf*(tan(deltaf)^2 + 1)]; 


A = [ 0, 0, -vf*sin(thetades);
    0, 0,  vf*cos(thetades);
    0, 0,             0];
 
B = [ cos(thetades), 0;
    sin(thetades), 0;
    0, 1];

Q= eye(3);
R = eye(2);
%if any(eig(A-B*K))>=0;
K= lqr(A,B,Q,R);
%end

u = -K*(xvec - xdes_vec) + [vf; wf];

dxvec = [u(1)*cos(theta);u(1)*sin(theta);u(2)];
% 
% % without noise
 X(:,i+1)= dxvec*dt+ X(:,i);

% with noise
%X(:,i+1)= dxvec*dt+ X(:,i) +0.05*randn(1);


end




for i=1:N;
    t=tsteps(i);
    basis = [1; t; t^2; t^3];
dbasis = [0; 1; 2*t; 3*t^2];
ddbasis = [0; 0;2; 6*t];
Xdes(1,i) = avec*basis;
    Xdes(2,i)= bvec*basis;
end

figure
plot(X(1,:), X(2,:),'LineWidth', 4);
hold on 
plot(Xdes(1,:), Xdes(2,:), 'LineWidth', 4);