function B_fitcoff = B_fit(Bdata,Lch,plot_flag)
[z_, B_] = B_regen(Bdata,Lch);
z_ = z_';
[B_max,ind] = max(B_);
z_1 = z_(1:ind);
B_1 = B_(1:ind);
z_2 = z_(ind:end);
B_2 = B_(ind:end);
% z < Bmax fitting
[xData, yData] = prepareCurveData( z_1, B_1 );
Bm_ = num2str(round(B_max,3));
Lm_ = num2str(round(z_(ind),3));
fun_fit1 = ['(',Bm_,'-b)*exp(-(abs(-x+',Lm_,')/s)^f)+b'];
ft = fittype( fun_fit1, 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [0 0 0];
opts.StartPoint = [50 1 10];
[fitresult_1, gof_1] = fit( xData, yData, ft, opts ); % Fit model to data.
% z > Bmax fitting
[xData, yData] = prepareCurveData( z_2, B_2 );
fun_fit2 = ['(',Bm_,'-b)*exp(-(abs(x-',Lm_,')/s)^f)+b'];
ft = fittype( fun_fit2, 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [0 0 0];
opts.StartPoint = [50 1 10];
[fitresult_2, gof_2] = fit( xData, yData, ft, opts ); % Fit model to data.
% result
Bm = round(B_max,3);
Lm = round(z_(ind),3);
B1 = fitresult_1.b;
B2 = fitresult_2.b;
f1 = fitresult_1.f;
f2 = fitresult_2.f;
s1 = fitresult_1.s;
s2 = fitresult_2.s;
B_fitcoff = [Bm Lm B1 f1 s1 B2 f2 s2];

font = 18;
LW = 1.5;
% fitted B field plot
if (plot_flag == 1)
    [z_gen,B_gen] = B_fitfcn(z_,Bm,Lm,B1,f1,s1,B2,f2,s2);
    figure
    P1 = plot(z_gen,B_gen,'r','LineWidth', LW);
    hold on;
    P2 = plot(z_,B_,'k','LineWidth', LW);
    P3 = plot(Lch*ones(10,1),linspace(0,900,10),'--b','LineWidth', LW-0.5);
    set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
    ylim([0 max(B_gen)*1.4])

    legend([P1, P2, P3],'Function fitted','B_r input','L_{ch} (exit)','Location', 'Northeast','Fontsize',font-5,'NumColumns',1);
    xlabel('Axial position (mm)','FontSize',font);
    ylabel('Radial magnetic field (G)','FontSize',font);

    hold off;
end

end