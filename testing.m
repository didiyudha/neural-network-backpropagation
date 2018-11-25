clc;
clear;

% ------------------------------------------------------------------------
% Load hasil pelatihan dari file TrainingResult.mat
% ------------------------------------------------------------------------

load('TrainingResult.mat');

TestSet = ...
    [3 3 1;
     3 1 2;
     2 3 1;
     2 1 2;
     1 3 1;
     1 2 2;
     1 1 1];

TestKelas = [0 1 0 1 0 1 0];

JumPola = length(TestSet(:, 1));
JumBenar = 0;

for pp = 1:JumPola
    CP = TestSet(pp, :);
    A1 = [];
    for ii = 1:JHNeuron
        v = CP * W1(:, ii);
        A1 = [A1 1 / (1+exp(-v))];
    end
    A2 = [];
    for jj = 1:JONeuron
        v = A1 * W2(:, jj);
        A2 = [A2 1 / (1+exp(-v))];
    end
    
    % ---------------------------------------------------------------------
    % Pemetaan A2 menjadi kelas keputusan
    % Jika A2 < 0.5 maka kelas =
    % ---------------------------------------------------------------------
    for jj =1:JONeuron
       if A2(jj) < 0.5
           Kelas = 0;
       else
           Kelas = 1;
       end
    end
    
    if Kelas == TestKelas(pp)
        JumBenar = JumBenar + 1;
    end
end
display(['Akurasi JST = ' num2str(JumBenar/JumPola)]);