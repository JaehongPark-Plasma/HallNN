% Radial magnetic field fitting fuction
function [z_gen,B_gen] = B_fitfcn(z,Bm,Lm,B1,f1,s1,B2,f2,s2)
z_gen = z;
B_gen = z*0;
for i = 1:numel(z)
    if (z(i) <= Lm)
        B_gen(i) = (Bm-B1)*exp(-(abs(-z(i)+Lm)/s1)^f1) + B1;
    else
        B_gen(i) = (Bm-B2)*exp(-(abs(z(i)-Lm)/s2)^f2) + B2;
    end
end

