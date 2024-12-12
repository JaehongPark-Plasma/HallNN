% Regenerate the Br data to fit with - Lch <=> Lch range
function [zgen, B] = B_regen(data,Lch)
data(:,1) = data(:,1);
if (max(abs(data(:,2))) < 10)
    data(:,2) = abs(data(:,2))*1e4;
else
    data(:,2) = abs(data(:,2));
end

for i=numel(data(:,1)):-1:1
    if (data(i,1) <= 2*Lch)
        z_(i) = data(i,1);
        B_(i) = data(i,2);
    end
end

% -L_ch <=> L_ch
zgen = linspace(0,2*Lch,201);
B = interp1(z_, B_, zgen, 'spline');
% smoothing in case of noisy data (e.g., when you using a measurement data)
B = smooth(B,10,'sgolay',3);

end