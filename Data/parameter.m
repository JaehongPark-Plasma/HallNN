EPS_0 = 8.85418782e-12;      % C/(V*m), vacuum permittivity
QE = 1.602176565e-19;        % C, electron charge
AMU = 1.660538921e-27;       % kg, atomic mass unit
Me = 9.10938215e-31;         % kg, electron mass
MXe = 131.3*AMU;
K_b = 1.380648e-23;          % J/K, Boltzmann constant
PI = 3.141592653;            % pi
EvToK = QE / K_b;            % 1eV in K ~ 11604
%sccm2mg_s = 0.0983009;       % 1 sccm (Xe) = 0.0983009 [mg/s] 
sccm2mg_s = 0.09763;       % 1 sccm (Xe) = 0.0983009 [mg/s] 
sccm2kg_s = sccm2mg_s/1e6;   % 1 sccm (Xe) = 0.0983009 [mg/s]
epsilion = 1e-10;
Torr2Pa = 133.322;           % 1 Torr to Pa : 1 Torr = 101325/760 Pa
Ei = 12.13;                  % Xe 1st ionization energy [eV]
QE_ME = QE / Me;             % For easy cozy
g0 = 9.8067;
s2mgs = sccm2mg_s;

