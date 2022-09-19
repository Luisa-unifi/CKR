function range = enc(CV,Delta)
%
% matlab script associated with the python file CKR.py.
%
params.tFinal = Delta;
R0 = zonotope(interval(CV(:,1),CV(:,2)));
% settings
options.timeStep = Delta;                           
options.taylorTerms = 4;                            
options.zonotopeOrder =  50;       
options.intermediateOrder =  50;
options.errorOrder = 20;

% reachability algorithm
options.alg = 'lin';%'poly';%'lin';
options.tensorOrder = 3;

% System Dynamics 
vanderPol = nonlinearSys(@vanderPolEq3);

STB1eq = @(x,u)([-x(1)^3+x(2);-x(1)^3-x(2)^3]);
STB1 = nonlinearSys(STB1eq);

STB2eq = @(x,u)([x(1)*(x(1)^2+x(2)^2-2)-4*x(1)*x(2)^2; 4*x(1)^2*x(2)+x(2)*(x(1)^2+x(2)^2-2)]);
STB2 = nonlinearSys(STB2eq);

NL2eq = @(x,u)([4*x(2)*(x(1)+sqrt(3.0)); -4*(x(1)-sqrt(3.0))^2-4*(x(2)+1)^2+16]);
NL2 = nonlinearSys(NL2eq);

LV2Deq = @(x,u)([x(1)*(1.5 - x(2));-x(2)*(3 - x(1))]);
LV2D = nonlinearSys(LV2Deq);

% Reachability Analysis    
params.R0 = R0;
R = reach(LV2D, params, options);

% example completed
reachset=R.timeInterval.set;
enc = interval(reachset{1});
range =  [enc.inf, enc.sup];
