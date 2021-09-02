function [fitness,xAtualiz,xExato] = ...
    filtroUKF(y,u,ry,bestExp,T,Patualiz,Q,R)

% Funcao para Filtro de Kalman Unscented
% Desenvolvida por Daniel Bruce Lane

%T = ;
yMedidas = y;

xAtualiz(:,1:ry+1) = [y(1:ry+1) u(1:ry+1)]'; % Valor inicial estado inicial

n = size(xAtualiz(:,1),1);                               % ordem do sistema

N = length(yMedidas);                                   % numero de medidas


%--------------------- Filtro de Kalman Unscented ------------------------%

% Condicoes iniciais zero
Pprev = zeros(n,n,N);
Pxy = zeros(n,N);
MatrizK = zeros(n,N);
Py = zeros(N,1);

for k = 2:length(y)
    %k = 2;
    
% ------------------------------ Previsao --------------------------------%
    % 1. Geracao de pontos-sigma
    for i = 1:2*n
        if i<=n
            temp = sqrtm(n*Patualiz(:,:,k-1));
            xTil(:,i) = temp(i,:)';
        else
            temp = -sqrtm(n*Patualiz(:,:,k-1));
            xTil(:,i) = temp(i-n,:)';
        end
        xSigma(:,i,k-1) = xAtualiz(:,k-1) + xTil(:,i);
    end
    
    if k>ry+1
    % 2. Transformacao dos pontos-sigma
        for i = 1:2*n
            xSigma(1,i,k) = ...
                T(1)* xSigma(1,i,k-1)+...
                T(2)* xSigma(1,i,k-2)+...
                T(3)* xSigma(1,i,k-3)+...
                T(4)* xSigma(1,i,k-4)+...
                T(5)* xSigma(1,i,k-5)+...
                T(6)* xSigma(1,i,k-6)+...
                T(7)* xSigma(1,i,k-7)+...
                T(8)* xSigma(1,i,k-8)+...
                T(9)* xSigma(1,i,k-9)+...
                T(10)*xSigma(1,i,k-10)+...
                T(11)*xSigma(1,i,k-11)+...
                T(12)*xSigma(1,i,k-12)+...
                T(13)*xSigma(1,i,k-13)+...
                T(14)*xSigma(2,i,k-1)+...
                T(15)*xSigma(2,i,k-2)+...
                T(16)*xSigma(2,i,k-3)+...
                T(17)*xSigma(2,i,k-4)+...
                T(18)*xSigma(2,i,k-5)+...
                T(19)*xSigma(2,i,k-6)+...
                T(20)*xSigma(2,i,k-7)+...
                T(21)*xSigma(2,i,k-8)+...
                T(22)*xSigma(2,i,k-9)+...
                T(23)*xSigma(2,i,k-10)+...
                T(24)*xSigma(2,i,k-11)+...
                T(25)*xSigma(2,i,k-12)+...
                T(26)*xSigma(2,i,k-13)+...
                T(27)*xSigma(1,i,k-1).^2+...
                T(28)*xSigma(1,i,k-1)*xSigma(2,i,k-1)+...
                T(29)*xSigma(1,i,k-2)*xSigma(2,i,k-2);
            %xSigma(2,i,k) = u(k);
            xSigma(2,i,k) = xSigma(2,i,k-1);
        end

        % 3. Combinacao dos pontos-sigma
        xHatPrev(:,k-1) = sum(xSigma(:,:,k),2)/(2*n);    

        % 4. Matriz de covariancia prevista
        for i = 1:2*n
            Pprev(:,:,k-1) = Pprev(:,:,k-1) + (xSigma(:,i,k) - ...
                xHatPrev(:,k-1))*(xSigma(:,i,k) - xHatPrev(:,k-1))';
        end
        Pprev(:,:,k-1) = Pprev(:,:,k-1)/(2*n) + Q;

% ------------------------------- Atualizacao ----------------------------%    
        % 5. Previsao da medida
        for i = 1:2*n
            if i<=n
                temp = sqrtm(n*Pprev(:,:,k-1));
                xTil(:,i) = temp(i,:)';
            else
                temp = -sqrtm(n*Pprev(:,:,k-1));
                xTil(:,i) = temp(i-n,:)';
            end
            xSigma(:,i,k) = xHatPrev(:,k-1) + xTil(:,i);
        end

        for i = 1:2*n
            yHat(i,k) = xSigma(1,i,k);
        end

        % 6. Combinacao para obter y previsto
        yPrev(k) = sum(yHat(:,k))/(2*n);


        % 7. Covariancia da inovacao
        for i = 1:2*n
            Py(k-1) = Py(k-1) + ...
                (yHat(i,k) - yPrev(k)) * (yHat(i,k) - yPrev(k))';
        end
        Py(k-1) = Py(k-1)/(2*n) + R;


        % 8. Covariancia cruzada
        for i = 1:2*n
            Pxy(:,k-1) = Pxy(:,k-1) + ...
                (xSigma(:,i,k) - xHatPrev(:,k-1))*(yHat(i,k) - yPrev(k))';
        end
        Pxy(:,k-1) = Pxy(:,k-1)/(2*n);


        % 9. "Eqs. de Kalman"
        MatrizK(:,k) = Pxy(:,k-1)*inv(Py(k-1));
        
        xAtualiz(:,k) = xHatPrev(:,k-1) + ...
            MatrizK(:,k)*(yMedidas(k) - yPrev(k));
        
        Patualiz(:,:,k) = Pprev(:,:,k-1) - ...
            MatrizK(:,k)*Py(k-1)*(MatrizK(:,k)');
    end
end

xExato = [y u];
NRMSE = goodnessOfFit(xAtualiz(1,:)',xExato(:,1),'NRMSE');
fitness = 100*(1-NRMSE);