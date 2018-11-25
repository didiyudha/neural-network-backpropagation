clc;
clear;

P = ...
    [3 3 2;  % P1
     3 3 2;  % P2
     3 2 1;  % P3
     3 1 1;  % P4
     2 3 2;  % P5
     2 2 2;  % P6
     2 2 1;  % P7
     2 1 1;  % P8
     1 3 2;  % P9
     1 2 1;  % P10
     1 1 1]; % P11
 
T = ...
    [1;
    1;
    1;
    0;
    1;
    1;
    1;
    0;
    1;
    0;
    1];
 
JumPola = length(P(:, 1));  % Jumlah semua pola latih.
DimPola = length(P(1, :));  % Dimensi pola latih.
JONeuron = length(T(1, :)); % Jumlah neuron pada output layer.
 
JHNeuron = 5;   % Jumlah neuron pada hidden layer.
LR = 0.1;       % Learning rate.
Epoch = 5000;   % Maximum iterasi.
MaxMSE = 10^-5; % Maximum MSE.
 
% ------------------------------------------------------------------------
% Bangkitkan weigth antara input layer dan hidden layer secara acak
% dalam interval -1 through 1. Simpan sebagai W1.
% ------------------------------------------------------------------------

W1 = [];
for ii = 1:JHNeuron
    W1 = [W1; (rand(1, DimPola) * 2-1)];
end
W1 = W1';

% ------------------------------------------------------------------------
% Bangkitkan weight antara hidden layer dan output layer secara acak
% dalam interval -1 through 1. Simpan sebagai W2.
% ------------------------------------------------------------------------

W2 = [];
for ii = 1:JONeuron
    W2 = [W2; (rand(1, JHNeuron) * 2-1)];
end
W2 = W2';

MSEepoch = MaxMSE + 1;  % Mean Square Error untuk 1 epoch.
MSE = [];               % List MSE untuk seluruh epoch.
ee = 1;                 % Index epoch.

while (ee <= Epoch) && (MSEepoch > MaxMSE)
    MSEepoch = 0;
    for pp = 1:JumPola
        CP = P(pp, :); % Current pattern.
        CT = T(pp, :); % Current target.
        
        % -----------------------------------------------------------------
        % Perhitungan maju untuk mendapatkan Output, Error, dan MSE
        % -----------------------------------------------------------------
        
        A1 = [];
        for ii = 1:JHNeuron
            v = CP*W1(:, ii);
            A1 = [A1 1/(1+exp(-v))];
        end
        A2 = [];
        for jj = 1:JONeuron
            v = A1 * W2(:, jj);
            A2 = [A2 1/(1+exp(-v))];
        end
        Error = CT - A2;
        
        for kk = 1:length(Error)
            MSEepoch = MSEepoch + Error(kk)^2;
        end
        
        % -----------------------------------------------------------------
        % Perhitungan mundur untuk mengupdate W1 dan W2
        % -----------------------------------------------------------------
        for kk = 1:JONeuron
            D2(kk) = A2(kk) * (1 - A2(kk)) * Error(kk);
        end
        dW2 = [];
        for jj = 1:JHNeuron
            for kk =1:JONeuron
                delta2(kk) = LR * D2(kk) * A1(jj);
            end
            dW2 = [dW2; delta2];
        end
        for jj =1:JHNeuron
            D1(jj) = A1 * (1-A1)' * D2 * W2(jj, :);
        end
        dW1 = [];
        for ii = 1:DimPola
            for jj = 1:JHNeuron
                delta1(jj) = LR * D1(jj) * CP(ii);
            end
            dW1 = [dW1; delta1];
        end
        W1 = W1 + dW1; % W1 baru.
        W2 = W2 + dW2; % W2 baru.
    end
    MSE = [MSE (MSEepoch/JumPola)];
    ee = ee + 1;
end

plot(MSE);
xlabel('Epoch');
ylabel('MSE');

 