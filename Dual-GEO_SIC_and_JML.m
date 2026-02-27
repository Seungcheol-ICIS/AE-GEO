%% Scha 2025.05.05 Start

%% Initialization
clear all; clc;
close all;

%% Inline Functions
LinearTodB = @(x)( 10*log10(x) ) ;
dBToLinear = @(x)( 10.^(x./10) );

%% Parameter Setting
nSymbol = 1e3;
iteration = 1e3;
nAntenna = 2;

M1 = 8; bps1 = log2(M1);    % Sat-1 Modulation
M2 = 4; bps2 = log2(M2);    % Sat-2 Modulation

if M1 == 2
    M1_name = 'BPSK';
elseif M1 == 4
    M1_name = 'QPSK';
elseif M1 == 8
    M1_name = '8-QAM';
elseif M1 == 16
    M1_name = '16-APSK';
end

if M2 == 2
    M2_name = 'BPSK';
elseif M2 == 4
    M2_name = 'QPSK';
elseif M2 == 8
    M2_name = '8-QAM';
elseif M2 == 16
    M2_name = '16-APSK';
end

%% SNR
%%% receiver diversity
% SNRdB = 0:2:20;
% SNR_User1 = 10.^(SNRdB/10);
% SNRdB_Difference = 3;
% SNR_User2 = 10.^((SNRdB-SNRdB_Difference)/10);

%%% transmit diversity
SNRdB = 0:2:20;
SNR_User1 = 10.^(SNRdB/10) /2;
SNRdB_Difference = 12;
SNR_User2 = 10.^((SNRdB-SNRdB_Difference)/10) /2;

%% Shadowed-Rician Fading Parameter
m = 10.1;                         % Nakagami-m shadowing parameter
b = 0.126;                        % 각 성분의 분산(산란파의 절반 전력)
omega = 0.835;                    % 평균 LOS 전력
K_factor = omega / (2*b);         % Rician K-factor
r_hat = 1;                        % RMS of the signal
phi = 0;                          % Phase parameter

%% BER
SER1_SIC = zeros(1, length(SNRdB));
SER2_SIC = zeros(1, length(SNRdB));

SER1_JML = zeros(1, length(SNRdB));
SER2_JML = zeros(1, length(SNRdB));

%% Simulation
parfor SNR_loop = 1 : length(SNRdB)
% for SNR_loop = 1 : length(SNRdB)
    tic;

    for iteration_loop = 1 : iteration
        %% Modulation
        % Sat-1
        userBit1 = randi([0,1], nSymbol * bps1, 1);
        userData1 = bit2int(userBit1, bps1);
        userModulatedSymbol1 = func_mapper_v(userData1, M1, 2);
        User1_TX_Signal = sqrt(SNR_User1(SNR_loop)) * userModulatedSymbol1;

        % Sat-1
        userBit2 = randi([0,1], nSymbol * bps2, 1);
        userData2 = bit2int(userBit2, bps2);
        userModulatedSymbol2 = func_mapper_v(userData2, M2, 2);
        User2_TX_Signal = sqrt(SNR_User2(SNR_loop)) * userModulatedSymbol2;

        %% Noise
        n = 1/sqrt(2) * randn(nAntenna, nSymbol) + 1j / sqrt(2) * randn(nAntenna,nSymbol);

        %% Shadowed Rician Fading Channel
        % h1 = func_shadowed_rician(User1_TX_Signal, nAntenna, m, K_factor, r_hat, phi);
        % h2 = func_shadowed_rician(User2_TX_Signal, nAntenna, m, K_factor, r_hat, phi);
        
        %% Rayleigh fading channel
        h1 = (randn(nAntenna, nSymbol) + 1j * randn(nAntenna, nSymbol)) / sqrt(2);
        h2 = (randn(nAntenna, nSymbol) + 1j * randn(nAntenna, nSymbol)) / sqrt(2);

        %% Air-Interface
        MU_RX_Signal_SIC = User1_TX_Signal .* h1 + User2_TX_Signal .* h2 + n;
        MU_RX_Signal_JML = User1_TX_Signal .* h1 + User2_TX_Signal .* h2 + n;

        %% Demodulation With SIC (MRC)
        y_combined_1 = sum(conj(h1) .* MU_RX_Signal_SIC, 1);
        y_hat_1 = y_combined_1 ./ sum(abs(h1).^2, 1);
        userDemodulatedData1_SIC = func_demapper_v(y_hat_1, M1, 2);

        demodBits1_SIC = de2bi(userDemodulatedData1_SIC, bps1, 'left-msb');
        demodBits1_SIC = transpose(demodBits1_SIC);
        demodBits1_SIC = demodBits1_SIC(:);

        User1_Recovered_DataSymbol = func_mapper_v(userDemodulatedData1_SIC, M1, 2);
        MU_RX_Signal_SIC = MU_RX_Signal_SIC - h1 .* sqrt(SNR_User1(SNR_loop)) .* User1_Recovered_DataSymbol;

        y_combined_2 = sum(conj(h2) .* MU_RX_Signal_SIC, 1);
        y_hat_2 = y_combined_2 ./ sum(abs(h2).^2, 1);
        userDemodulatedData2_SIC = func_demapper_v(y_hat_2, M2, 2);

        demodBits2_SIC = de2bi(userDemodulatedData2_SIC, bps2, 'left-msb');
        demodBits2_SIC = transpose(demodBits2_SIC);
        demodBits2_SIC = demodBits2_SIC(:);

        %% Demodulation with JML
        demodulatedData1_JML = zeros(1, length(User1_TX_Signal));
        demodulatedData2_JML = zeros(1, length(User2_TX_Signal));
        minIndex = zeros(2,1); % nUser x 1

        for JML_loop = 1 : nSymbol
            minValue = inf;
            minIndex(:) = -1;
            isUpdated = false;

            for i = 1 : 1 : M1
                for j = 1 : 1 : M2
                    testSymbol1 = func_mapper_v(i-1, M1, 2);
                    testSymbol2 = func_mapper_v(j-1, M2, 2);
                    y_test1 = sqrt(SNR_User1(SNR_loop)) * h1(:,JML_loop) .* testSymbol1;
                    y_test2 = sqrt(SNR_User2(SNR_loop)) * h2(:,JML_loop) .* testSymbol2;

                    testValue = sum(abs(MU_RX_Signal_JML(:,JML_loop) - y_test1 - y_test2).^2);

                    if ~isnan(testValue) && ~isinf(testValue) && testValue < minValue
                        minValue = testValue;
                        minIndex(1) = i - 1;
                        minIndex(2) = j - 1;
                        isUpdated = true;
                    end
                end
            end

            if ~isUpdated
                warning('MinIndex not updated at symbol #%d, fallback to (0,0)', JML_loop);
                minIndex(:) = 0;
            end

            demodulatedData1_JML(JML_loop) = minIndex(1);
            demodulatedData2_JML(JML_loop) = minIndex(2);
        end

        % Sat-1
        demodBits1_JML = de2bi(demodulatedData1_JML, bps1, 'left-msb');
        demodBits1_JML = transpose(demodBits1_JML);
        demodBits1_JML = demodBits1_JML(:);

        % Sat-2
        demodBits2_JML = de2bi(demodulatedData2_JML, bps2, 'left-msb');
        demodBits2_JML = transpose(demodBits2_JML);
        demodBits2_JML = demodBits2_JML(:);

        %% Error counting
        SER1_SIC(SNR_loop) = SER1_SIC(SNR_loop) + sum(userData1 ~= userDemodulatedData1_SIC(:)) / nSymbol;
        SER2_SIC(SNR_loop) = SER2_SIC(SNR_loop) + sum(userData2 ~= userDemodulatedData2_SIC(:)) / nSymbol;

        SER1_JML(SNR_loop) = SER1_JML(SNR_loop) + sum(userData1 ~= demodulatedData1_JML(:)) / nSymbol;
        SER2_JML(SNR_loop) = SER2_JML(SNR_loop) + sum(userData2 ~= demodulatedData2_JML(:)) / nSymbol;
    end

    toc;
end
%% Comparison
SER1_SIC = SER1_SIC / iteration;
SER2_SIC = SER2_SIC / iteration;

SER1_JML = SER1_JML / iteration;
SER2_JML = SER2_JML / iteration;

%% Figure ploting
PDFexportON = true;

% Figure Setting
LW = 1.5; MS = 5; FS = 11; LFS = 17; AFS = 14; TFS = 13;
fig = figure();

% Plotting
semilogy(SNRdB, SER1_SIC, 'k--o', 'LineWidth', LW, 'MarkerSize', MS); hold on; grid on;
semilogy(SNRdB, SER2_SIC, 'r--o', 'LineWidth', LW, 'MarkerSize', MS); hold on; grid on;
semilogy(SNRdB, SER1_JML, 'k-.p', 'LineWidth', LW, 'MarkerSize', MS); hold on; grid on;
semilogy(SNRdB, SER2_JML, 'r-.p', 'LineWidth', LW, 'MarkerSize', MS); hold on; grid on;

% Legend
fig_legend = legend({ ...
    sprintf('[SIC] User #1 (%s)', M1_name), ...
    sprintf('[SIC] User #2 (%s)', M2_name), ...
    sprintf('[JML] User #1 (%s)', M1_name), ...
    sprintf('[JML] User #2 (%s)', M2_name)}, ...
    'Location','southwest');

set(fig_legend,'interpreter','latex','fontsize', FS,'FontName','Times New Roman');
set(fig,'position', [200, 200, 650, 500],'PaperPositionMode','auto');
set(gca,'fontsize',AFS,'FontName','Times New Roman');
set(0,'DefaultAxesFontName', 'Times New Roman');
xlabel('SNR (dB)','interpreter','latex','fontsize', FS,'FontName','Times New Roman');
ylabel('SER','interpreter','latex','fontsize', FS,'FontName','Times New Roman');
ylim([1e-4 1]); xlim([0 SNRdB(end)]);

%% SAVE file
save_path = '/Users/scha/Library/CloudStorage/Dropbox/ICIS lab/research_24Seungcheol/manuscript-SC/[revision] IEEE Communications Letters (AE design)/revision-20260130/reply_letter/simulation_code/20260218/DataFile/SER_data/AE_vs_D2';

SER_table = table( ...
    SER1_SIC(:), SER2_SIC(:), ...
    SER1_JML(:), SER2_JML(:), ...
    'VariableNames', {'SER1_SIC','SER2_SIC','SER1_JML','SER2_JML'});

writetable(SER_table, fullfile(save_path, '20260102_SIC_JML.csv'));

if PDFexportON
    exportgraphics(fig, fullfile(save_path, '20260102_SIC_JML.pdf'), 'ContentType','vector');
    exportgraphics(fig, fullfile(save_path, '20260102_SIC_JML.jpg'), 'Resolution', 300);
end