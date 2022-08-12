% Numerical computation of the proportion of states which imply coexistence
% or extinction of the different variants for the general model

% Pablo Rosillo, 2022
% Final Master Thesis

set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultAxesTickLabelinterpreter','latex');
set(groot,'defaultlegendinterpreter','latex');

s1min = 0.5;

c2 = 0.8;

c1 = 0:0.25:1;

c12 = 0:0.025:1;

cntr = zeros(1,length(c12)); cntrcoex = cntr;
cntrx0 = cntrcoex; cntrx1 = cntrcoex;

% figure(1)
%
% for i = 1:length(c1)
%
%     [s1vecst, s2vecst, Xvecst, s1divxnc, s1divxcn, s1divonc, s1divocn, s2divxnc, s2divxcn, s2divonc, s2divocn, ...
%         omegavecst, cntr(i), cntrcoex(i)] = stdiagram(s1min, 1, s1min, c1(i), 0.01);
%
%     subplot(2,length(c1),i)
%     colormap(jet)
%     scatter(s1vecst, s2vecst, 30, Xvecst, 'filled')
%     xlabel('$s_1$'); ylabel('$s_2$'); title("$\c1 =$ " + c1(i))
%     hold on
%     plot(s1divxnc, s2divxnc, 'LineWidth', 1.5, 'Color', '#D95319')
%     hold on
%     plot(s1divxcn, s2divxcn, 'LineWidth', 1.5, 'Color', '#FFFFFF')
%     axis([s1min 1 s1min 1])
%     xtickformat('%.2f'); ytickformat('%.2f')
%     clim([-1, 1])
%
%     subplot(2,length(c1),i+length(c1))
%     colormap(jet)
%     scatter(s1vecst, s2vecst, 30, omegavecst, 'filled')
%     xlabel('$s_1$'); ylabel('$s_2$');
%     axis([s1min 1 s1min 1])
%     xtickformat('%.2f'); ytickformat('%.2f')
%     clim([-1, 1])
%
%
% end
%
% figure(1)
% colorbar

parfor i = 1:length(c12)
    disp("c1 = " + c12(i))
    tic
    [s1vecst, s2vecst, Xvecst, s1divxnc, s1divxcn, s1divonc, s1divocn, ...
        s2divxnc, s2divxcn, s2divonc, s2divocn, omegavecst, zvecst, cntr(i),...
        cntrcoex(i), cntrx1(i), cntrx0(i)] = ...
        stdiagram(s1min, 1, s1min, c2, c12(i), 0.01);
    disp("c1 = " + c12(i) + " finished in " + toc + " s.")
end

save('c2_08')

% figure(3)
% plot(c12, cntrcoex4, '.', 'MarkerSize',10)
% hold on
% plot(c12, cntrx04, '.')
% hold on
% plot(c12, cntrx14, '.')
% title('$c_2 = 0.499$');
% xlabel('$c1$'); ylabel('$P_\mathrm{st}$');
% legend('$0 < X < 1$', '$X = 0$', '$X = 1$')
% xtickformat('%.1f'); ytickformat('%.1f')
% ylim([-0.05 1.05]);

% c12(51) = [];
% cntr5(51) = []; cntr(51)=[];
% figure(4)
% plot(c12, cntr5, '.', 'MarkerSize',14)
% hold on
% plot(c12, cntr-1, '.')
% legend('$c_2 > 0.5$', '$c_2 < 0.5$');
% xlabel('$c1$'); ylabel('$P_\mathrm{st}$');
% xtickformat('%.1f'); ytickformat('%.2f')

%%%%%%%%%%%%%%%%%%%%%%


function [s1vecst, s2vecst, Xvecst, s1divxnc, s1divxcn, s1divonc, s1divocn, s2divxnc, s2divxcn, s2divonc, s2divocn, ...
    omegavecst, zvecst, indst, indstcoex, indstx1, indstx0] = stdiagram(s1min, s1max, s2min, c2, c1, sstep)

syms X
syms omega
syms z
syms s1
syms s2


s1vecst = []; s2vecst = [];
Xvecst = []; omegavecst = []; zvecst = [];
s1divxnc = []; s1divxcn = []; s1divonc = []; s1divocn = [];
s2divxnc = []; s2divxcn = []; s2divonc = []; s2divocn = [];
indst = 0; count = 0; indstcoex = 0;
indstx1 = 0; indstx0 = 0;

eqx = 0.5*(X*z-omega+X*omega+s1*(-2*X^2+omega-X*(-2+z+omega))+s2*(2*X^2+omega-X*(2+z+omega)));
eqomega = 0.5*(-2*(-1+s1+s2)*X^2+omega*(-2+s1-s2+z-2*c2*z-omega+2*c2*omega)+X*(-1-2*c2+2*s2-2*z+s2*z+2*omega+s2*omega-s1*(-2+z+omega)));
eqz = 0.5*(-1-2*(-1+s1+s2)*X^2+z^2-2*omega+s1*omega-s2*omega-z*omega-2*c1*(-1+X+z^2-z*omega)+X*(-1-2*z+2*omega-s1*(-2+z+omega)+s2*(2+z+omega)));

dxdx = diff(eqx,X); dxdo = diff(eqx,omega); dxdz = diff(eqx,z);
dodx = diff(eqomega, X); dodo = diff(eqomega, omega); dodz = diff(eqomega,z);
dzdx = diff(eqz,X); dzdo = diff(eqz,omega); dzdz = diff(eqz,z);

for s1it = s1min:sstep:s1max

    disp(100*(s1it-s1min)/(s1max-s1min) + " %")

    s2it = s2min;

    count = count + 1;

    eqxit = vpa(subs(eqx, [s1, s2], [s1it, s2it]));
    eqomegait = vpa(subs(eqomega, [s1, s2], [s1it, s2it]));
    eqzit = vpa(subs(eqz, [s1, s2], [s1it, s2it]));

    assume(omega,'real')
    assume(z,'real')
    assume(X,'real')

    sols = vpasolve([eqxit == 0, eqomegait == 0, eqzit == 0], [X, omega, z]);


    for i = 1:length(sols.X)


        J = [vpa(subs(dxdx, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)])),...
            vpa(subs(dxdo, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)])),...
            vpa(subs(dxdz, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)]));...
            vpa(subs(dodx, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)])),...
            vpa(subs(dodo, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)])),...
            vpa(subs(dodz, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)]));
            vpa(subs(dzdx, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)])),...
            vpa(subs(dzdo, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)])),...
            vpa(subs(dzdz, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)]))];

        e = eig(J);



        if real(e(1)) < 0 && real(e(2)) < 0 && real(e(3)) < 0 && sols.X(i) >= 0 &&...
                sols.X(i) <= 1 && sols.omega(i) >= -1 &&...
                sols.omega(i) <= 1 && sols.z(i) >= -1 &&...
                sols.z(i) <= 1

            indst = indst + 1;
            s1vecst(indst) = s1it;
            s2vecst(indst) = s2it;
            Xvecst(indst) = sols.X(i);
            omegavecst(indst) = sols.omega(i);
            zvecst(indst) = sols.z(i);

            if sols.X(i) < 1 && sols.X(i) > 0

                indstcoex = indstcoex + 1;

                if (indst >= 2 && Xvecst(indst-1) == 1) || (indst >= 2 && Xvecst(indst-1) == 0)

                    s1divxcn = [s1divxcn s1it];
                    s2divxcn = [s2divxcn s2it];

                end
                %
                %                 elseif (indst >= 2 && sols.X(i) == 1) || (indst >= 2 && sols.X(i) == 0)
                %
                %                     if Xvecst(indst-1) < 1 && Xvecst(indst-1) > 0
                %
                %                         s1divxnc = [s1divxnc s1it];
                %                         s2divxnc = [s2divxnc s2it];
                %
                %                     end
                %
            elseif sols.X(i) == 1

                indstx1 = indstx1 + 1;

            else

                indstx0 = indstx0 + 1;


            end

        end

    end




    for s2it = (s2min+sstep):sstep:s1it

        count = count + 1;

        eqxit = vpa(subs(eqx, [s1, s2], [s1it, s2it]));
        eqomegait = vpa(subs(eqomega, [s1, s2], [s1it, s2it]));
        eqzit = vpa(subs(eqz, [s1, s2], [s1it, s2it]));

        assume(omega,'real')
        assume(z,'real')
        assume(X,'real')

        sols = vpasolve([eqxit == 0, eqomegait == 0, eqzit == 0], [X, omega, z],...
            [[0,1];[-1,1];[-1,1]]);

        for i = 1:length(sols.X)



            J = [vpa(subs(dxdx, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)])),...
                vpa(subs(dxdo, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)])),...
                vpa(subs(dxdz, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)]));...
                vpa(subs(dodx, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)])),...
                vpa(subs(dodo, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)])),...
                vpa(subs(dodz, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)]));
                vpa(subs(dzdx, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)])),...
                vpa(subs(dzdo, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)])),...
                vpa(subs(dzdz, [s1, s2, X, omega, z], [s1it, s2it, sols.X(i), sols.omega(i), sols.z(i)]))];

            e = eig(J);



            if real(e(1)) < 0 && real(e(2)) < 0 && real(e(3)) < 0 && sols.X(i) >= 0 &&...
                    sols.X(i) <= 1 && sols.omega(i) >= -1 &&...
                    sols.omega(i) <= 1 && sols.z(i) >= -1 &&...
                    sols.z(i) <= 1

                indst = indst + 1;
                s1vecst(indst) = s1it;
                s2vecst(indst) = s2it;
                Xvecst(indst) = sols.X(i);
                omegavecst(indst) = sols.omega(i);
                zvecst(indst) = sols.z(i);

                if sols.X(i) < 1 && sols.X(i) > 0

                    indstcoex = indstcoex + 1;

                    if (indst >= 2 && Xvecst(indst-1) == 1) || (indst >= 2 && Xvecst(indst-1) == 0)

                        s1divxcn = [s1divxcn s1it];
                        s2divxcn = [s2divxcn s2it];

                    end
% 
%                 elseif (indst >= 2 && sols.X(i) == 1) || (indst >= 2 && sols.X(i) == 0)
% 
%                     if Xvecst(indst-1) < 1 && Xvecst(indst-1) > 0
% 
%                         s1divxnc = [s1divxnc s1it];
%                         s2divxnc = [s2divxnc s2it];
% 
%                     end
% 
                elseif sols.X(i) == 1

                    indstx1 = indstx1 + 1;

                else 

                    indstx0 = indstx0 + 1;


                end

            end

        end


    end
end

indstcoex = indstcoex/indst;
indstx1 = indstx1/indst;
indstx0 = indstx0/indst;
indst = indst/count;

end


             








