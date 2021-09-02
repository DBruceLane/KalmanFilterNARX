%-------------------- Filtro de Kalman Unscented -------------------------%
%                                                                         %
% Baseado no Template elaborado por: Prof. Dr. Victor Frencl              %
%                                                                         %
% Versao final desenvolvida por Daniel Bruce Lane                         %
%                                                                         %
%-------------------------------------------------------------------------%

clear
close all
clc

%-------------------- Dados e Variaveis iniciais -------------------------%

mostrarEstimativaNARX = 0;           % Optar por 0 ou 1, sendo 1 verdadeiro
mostrarEstimativaUKF = 1;
lineW = 1.5;                       % Tamanho em pixels das curvas dos plots


Data = importdata("levitadorMag.txt");
%t = Data(2001:8000,1);                                             % Tempo
u = Data(2001:8000,2);                                            % Entrada
y = -Data(2001:8000,3);                                             % Saída


%-------------------------- Estimativa NARX ------------------------------%

ry = 13;            % Numero de regressores de y (foi suposto igual para u)  
bestFit = -Inf;
iMax = 7;             % Iteracao maxima, relacionada ao expoente de u(k-1)

MSEvet = zeros(iMax,1);                                      % Pre alocacao
FITvet = zeros(iMax,1);
for j=ry:ry             % Loops apenas para testes de multiplos regressores
    for i=1:iMax
        NLy = {'y', 1, 2};             % Equivale a 1 regressor de y(k-1)^2
        NLu = {'u', 0, 2};      % Equivale a 8 regressores de u(k-1)^9
        NLc = {'c', 2, [1 1]};    % Equivale a 1 regressor de u(k-1)*y(k-1) 
        fNonLinear = {NLy NLu NLc};                     % Funcao nao linear 

        [fitness,MSE,y_hat,Psi,Theta,ksi] = genNARX(y,u,j,j,fNonLinear,0);
        MSEvet(i) = MSE;
        FITvet(i) = fitness;
        if fitness>bestFit && fitness>0     % Seleciona a melhor estimativa
            bestYhat = y_hat;
            bestReg = fNonLinear; 
            bestFit = fitness;
            bestTheta = Theta;
            bestPsi = Psi;
            bestI = i;
            bestJ = j;
            bestExp = 1+i/10;
        end
    end
end
%%
%--------------------- Filtro de Kalman Unscented ------------------------%
rMax = 1;
qMax = 1;

rng(2);                                      % Torna simulacao reprodutivel

Patualiz = zeros(2,2,ry+1);                                 % Pre alocacoes
vetFIT = zeros(rMax,qMax);
vetFIT2 = zeros(rMax,qMax);
for r = 1:rMax                  % Loops para testes do ajuste do filtro UKF
    for q = 1:qMax
        for i=1:ry+1
            vetY = 2*rand(1);
            vetU = 1*rand(1);
            Patualiz(:,:,i) = diag([vetY vetU]);  % covariancia do erro t=0
        end

        vetY = 0.02*rand(1);
        vetU = 0.01*rand(1);
        Q = diag([vetY vetU]); % matriz de covariancia do ruido de processo
        
        R = 0.1;                   % matriz de covariancia do ruido de medida

        [fitness,xAtualiz,xExato] = ...
            filtroUKF(y,u,ry,bestExp,bestTheta,Patualiz,Q,R);
        vetFIT(r,q) = fitness; 
        NRMSE = goodnessOfFit(xAtualiz(1,:)',bestYhat(:),'NRMSE');
        fitnessNARX = 100*(1-NRMSE);
        vetFIT2(r,q) = fitnessNARX;
    end
end

%%
%plot(vetFIT);
%hold on;
%plot(vetFIT2(r,:));
%legend('Fitness com medidas','Fitness com NARX')

%%
%----------------------------- Graficos ----------------------------------%

                                                   % Graficos NARX opcional
if mostrarEstimativaNARX
    figure;                   % Plot do fitness com variacao de regressores
    yyaxis left
    plot(MSEvet,'LineWidth',lineW);
    hold on;
    yyaxis right
    plot(FITvet,'LineWidth',lineW);
    bestIndex = find(FITvet==max(FITvet));
    plot(bestIndex,max(FITvet),'k*','LineWidth',3);
    legend('MSE','Fitness','Melhor Estimativa');
    title(['Simulando variacao do expoente de u(k), melhor: ' ...
        num2str(1+bestI/10) newline num2str(2*bestJ) ...
        ' regressores lineares e 10 nao lineares']);
    estiloPlot();

    figure;                                               % Plot da Entrada
    plot(u,'LineWidth',lineW);
    ylabel('Tensao [V]');
    xlabel('Tempo [ms]');
    title('Sinal de Entrada');
    estiloPlot();
    
    figure;                                                 % Plot da Saida
    plot(bestYhat,'--','LineWidth',lineW);
    hold on;
    plot(y,'LineWidth',lineW);
    ylabel('Deslocamento [mm]');
    xlabel('Tempo [ms]');
    legend('Estimativa NARX','Medidas');
    %title('Sinal de Saida');
    title(['Sinal de Saida, Fitness=' num2str(max(FITvet)) '%']);
    estiloPlot();
end

if mostrarEstimativaUKF
figure;                          % Plot das Estimativa NARX e do filtro UKF
hold on;
plot(xExato(:,1),'k','LineWidth',lineW);
plot(bestYhat(:),'b--','LineWidth',lineW);
plot(xAtualiz(1,:),'r','LineWidth',lineW);
ylabel('Deslocamento [mm]');
xlabel('Tempo [ms]');
legend('Medidas','Estimativa NARX','Estimativa do Estado');
estiloPlot();

figure;                                            % Plot da estimativa UKF
plot(xExato(:,1),'k','LineWidth',lineW);
hold on;
plot(xAtualiz(1,:)','r.','LineWidth',lineW);
legend('Medida','Estimativa');
ylabel('Deslocamento [mm]');
xlabel('Tempo [ms]');
title(['Estimativa do Estado' newline 'Fitness com as medidas = ' ...
    num2str(fitness) '%'...
    newline 'Fitness com o modelo NARX = ' num2str(fitnessNARX) '%']);
estiloPlot();

                                             % Plot do Erro das estimativas
figure;
subplot(211);
erroMedida = xExato(:,1)-xAtualiz(1,:)';
%plot(xcorr(erroMedida),'LineWidth',lineW);
plot(abs(erroMedida),'LineWidth',lineW);
ylabel('Erro da Saida');
xlabel('Tempo [ms]');
title('Módulo do Erro de Estimativa quanto à medida');
estiloPlot();

subplot(212);
ErroProcesso = bestYhat(:)-xAtualiz(1,:)';
%%plot(xcorr(ErroProcesso),'LineWidth',lineW);
plot(abs(ErroProcesso),'LineWidth',lineW);
ylabel('Erro da Saida');
xlabel('Tempo [ms]');
title('Modulo do Erro de Estimativa quanto ao processo');
estiloPlot();
end