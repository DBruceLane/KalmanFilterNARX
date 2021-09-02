function [fitness,MSE,y_hat,Psi,Theta,ksi] = genNARX(y,u,ry,ru,fNonLinear,mostrarEstimativa)
% Generalizacao de modelagem NARX
n = size(u,1);

% Se o menor indice na equacao de y(k) eh (k-1), o menor k possível eh 2 (pro Matlab iniciar indice em 1)
% Se fosse o caso kmin = 2

if ru>=(ry+2)
    kmin = ru;  
    %Quando ha 2 regressores a mais de entrada que saida, ex: ry = 1 e ru = 3
    %y(k) = y(k-1) + u(k) + u(k-1) + u(k-2) 
    %Nesse caso observa-se o ultimo elemento de entrada, kmin seria 3
else
    kmin = 1+ry;
    %Nos outros casos observa-se o ultimo elemento de saida
end

Y = y(kmin:n);

Psiy = zeros(1+n-kmin,ry); %Elementos de saida na matriz Psi
for i=1:ry
        Psiy(1:1+n-kmin,i) = y(kmin-i:n-i); 
end

Psiu = zeros(1+n-kmin,ru); %Elementos de entrada na matriz Psi
for i=1:ru
       Psiu(1:1+n-kmin,i) = u(kmin-i+1:n-i+1);     
end

% fNonLinear = {NLy NLu NLc};

% Regressores Nao lineares
NLy = fNonLinear{1};
NLu = fNonLinear{2};
NLc = fNonLinear{3};

rNLy = NLy{2};
rNLu = NLu{2};
rNLc = NLc{2};

if rNLy ~= 0
    for i=1:rNLy
        PsiNLy(1:1+n-kmin,i) = y(kmin-i+1:n-i+1).^NLy{3};
    end
else
    PsiNLy = [];
end

if rNLu ~= 0
    for i=1:rNLu
        PsiNLu(1:1+n-kmin,i) = u(kmin-i+1:n-i+1).^NLu{3};
    end
else
    PsiNLu = [];
end

expC = NLc{3};
if rNLc ~= 0
    for i=1:rNLc
        PsiNLc(1:1+n-kmin,i) = (y(kmin-i+1:n-i+1).^expC(1)).*(u(kmin-i+1:n-i+1).^expC(2));
    end
else
    PsiNLc = [];
end

Psif = [PsiNLy PsiNLu PsiNLc];

Psi = [Psiy Psiu Psif];
Theta = inv(Psi'*Psi)*Psi'*Y; %Pseudo Inversa

%Free Run Simulation
y_hat = zeros(1,ry);
for i=1:ry
    y_hat(i) = y(i); %Condicoes de Contorno
end

parcial = zeros(1,ry+ru); %Parciais de y_hat
%Exemplo de equacao pra mostrar os indices do for
%y_hat(k) = Theta(j=1)*y(k-1) + Theta(j=2)*y(k-2) + Theta(j=3)*u(k) + Theta(j=4)*u(k-1)
for k=kmin:n  
   y_hat(k) = 0;
   for j=1:ry+ru+rNLy+rNLu+rNLc
       if j>ry+ru+rNLy+rNLu
           parcial(j) = Theta(j)*u(k-j+ry+1+ru+rNLy+rNLu)*y(k-j+ry+1+ru+rNLy+rNLu); %Parcial NLc
       elseif j>ry+ru+rNLy
           parcial(j) = Theta(j)*u(k-j+ry+1+ru+rNLy)^NLu{3}; %Parcial NLu
       elseif j>ry+ru
           parcial(j) = Theta(j)*y_hat(k-j+ry+1+ru)^NLy{3}; %Parcial NLy
       elseif j>ry
           parcial(j) = Theta(j)*u(k-j+ry+1); %Parcial dependente da entrada  
       else
           parcial(j) = Theta(j)*y_hat(k-j); %Parcial dependente da saida
       end
       y_hat(k) = y_hat(k)+parcial(j);
   end   
end

MSE = 0;
for i=1:n
    MSE = MSE + i*((y(i)-y_hat(i))^2)/n; %Mean Squared Error
end 

ksi = y - y_hat';

NRMSE = goodnessOfFit(y_hat',y,'NRMSE');
fitness = 100*(1-NRMSE);

if mostrarEstimativa
    t = (0:n-1)';
    hold on;
    plot(t,y_hat,'--','DisplayName',[num2str(ry) 'y/',num2str(ru) 'u: ',num2str(MSE) ' MSE']);
    stem(ksi,'Marker','none','DisplayName','Residuo');
    legend('off');
    legend('show'); %Update da legenda pra adicionar texto em vez de substituir
    grid on;
end
