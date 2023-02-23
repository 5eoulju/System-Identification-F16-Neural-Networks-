%%% Plotting Script for IEKF

tstart = 0;
dt = 0.01;
tend = dt*(N-1);
tspan = tstart:dt:tend;

% True measurements results from IEKF
a_true=(Z_k1k1(1,:));
b_true=(Z_k1k1(2,:));
V_true=(Z_k1k1(3,:));

figure()
% subplot(211)

plot(tspan, Z_k1k1(1,:));
hold on
plot(tspan, alpha_m);
hold on
plot(tspan, a_true);
hold on
plot(tspan ,X_k1k1(4,:));
pbaspect([2 1 1])
title('alpha');
grid()
l=legend('Estimated IEKF alpha','Measured alpha','reconstructed true alpha', 'estimated upwash coeficient', 'Location','northwest');
l.FontSize=8;
xlabel('time [s]')
ylabel('angle of attack [rad]')
% if savef
% saveas(gcf,'Report/plots/alpharecon.eps','epsc')
% end


figure
plot(tspan, Z_k1k1(2,:));
hold on
plot(tspan, beta_m);
hold on
plot(tspan, b_true); 
title('beta');
legend('predicted output','measured output','estimated true beta');
xlabel('time [s]')
ylabel('sideslip angle [rad]')


figure
plot(tspan, Z_k1k1(3,:));
hold on
plot(tspan, Vtot);
plot(tspan, V_true);
title('V')
legend('predicted output','measured output','estimated true V');
xlabel('time [s]')
ylabel('velocity [m/s]')

figure
plot(tspan,(Z_k1k1(1,:)'-alpha_m(:)));

TRIeval = delaunayn([a_true' b_true']);

figure
plot3(a_true',b_true',Cm,'.k');
xlabel('alpha')
ylabel('beta');
hold on
plot3(alpha_m',beta_m',Cm,'.b');
grid();
figure
subplot(221)

plot(tspan(2:end), est_err(1,:))
title('u')
subplot(222)
plot(tspan(2:end), est_err(2,:))
title('v')
subplot(223)
plot(tspan(2:end), est_err(3,:))
title('w')
subplot(224)
plot(tspan(2:end), est_err(4,:))
title('Ca')

figure()
plot(tspan, X_k1k1(4,:))
grid()
xlabel('Time [s]')
ylabel('C_{\alpha_{up}}[-]')
title('Estimated Upwash coefficient')
pbaspect([4 1 1])
ylim([min(X_k1k1(4,:)) max(X_k1k1(4,:))*1.1])
% if savef
% saveas(gcf,'Report/plots/caup.eps','epsc')
% end


% atrue=atrue';
% Btrue=Btrue';
% Vtrue=Vtrue';

% if savef
% save('Data/atrue.mat','atrue');
% save('Data/Btrue.mat','Btrue');
% save('Data/Vtrue.mat','Vtrue');
% save('Data/T.mat','T');
