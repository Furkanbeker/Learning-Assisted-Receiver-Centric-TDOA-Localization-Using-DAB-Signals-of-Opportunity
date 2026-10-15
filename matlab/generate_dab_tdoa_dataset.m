%% ========================================================================
%  generate_dab_tdoa_dataset.m
%
%  Dataset generation for the paper
%    "Learning-Assisted Receiver-Centric TDOA Localization Using DAB
%     Signals of Opportunity" (submitted to IEEE Access).
%
%  What this script does
%  ---------------------
%  For each random scenario, four DAB transmitters (TX), one static
%  reference receiver (REF) and one UAV receiver are placed in a
%  5000 m x 5000 m field. The UAV follows one of four trajectories
%  (circle, line, sine, sine2). At every time step:
%    1) each TX-RX link is simulated (delay + TX timing offset, FSPL, AWGN)
%       using a real DAB Mode-I baseband waveform,
%    2) the TOA is estimated from the DAB Phase Reference Symbol (PRS),
%    3) receiver-centric TDOAs (UAV minus REF) are solved with a
%       multi-start weighted Gauss-Newton (GN) solver,
%    4) the GN fixes are tracked by a maneuver-adaptive constant-turn
%       extended Kalman filter (CT-EKF).
%  All features, labels and errors are stored per measurement. The
%  AI-ready CSV is the input of the Python training pipeline (../python).
%
%  How to run
%  ----------
%    1) Put the DAB IQ file in ../data/dab_iq/ (see README).
%    2) Adjust the USER SETTINGS below if needed (e.g. a smaller
%       numScenarios for a quick test).
%    3) Run this script from MATLAB. All helper functions in ./functions
%       are added to the path automatically.
%
%  Outputs are written to ../results/dab_soop_tdoa_dataset_<timestamp>/.
%  See README.md for the list of output files and the requirements.
%% ========================================================================

clear; clc; close all;

%% 0) PATHS

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;   % e.g. when the code is run section by section
end
repoRoot = fileparts(thisDir);
addpath(genpath(fullfile(thisDir, 'functions')));

%% 1) USER SETTINGS

% --- DAB waveform -------------------------------------------------------
% DAB Mode-I baseband IQ file (interleaved complex float32, fs = 2.048 MHz).
iqFile = fullfile(repoRoot, 'data', 'dab_iq', 'basic.cf32');
fs = 2.048e6;             % Sampling rate of the IQ file (Hz)
masterDurationSec = 2;    % Length of the waveform segment used (s)

% Carrier frequencies (Hz) of the four DAB transmitters. They are used
% only for the FSPL computation.
txFreqHz = [195.936e6, 199.360e6, 202.928e6, 206.352e6];

% --- Signal processing --------------------------------------------------
c = 3e8;                  % Propagation speed (m/s)
searchMaxLag = 256;       % Half-width of the PRS TOA search window (samples)

% Length of each received observation window. A DAB Mode-I frame lasts
% 96 ms, so with a 10 ms window only the first PRS of the waveform falls
% inside the window and the multi-frame TOA estimator uses a single PRS.
viewDuration_ms = 10;

% --- Scenarios ------------------------------------------------------------
numScenarios = 10000;     % Number of random scenarios (use e.g. 20 for a quick test)
numMeas = 50;             % UAV measurements (time steps) per scenario
measDt = 1.0;             % Time between measurements (s)

areaMin = 0;              % Square operating field [areaMin, areaMax]^2 (m)
areaMax = 5000;
minRefToTx = 250;         % Minimum REF-to-TX distance (m)

% Per-TX timing offsets, drawn uniformly in [offsetMinNs, offsetMaxNs] for
% each scenario. In receiver-centric TDOA the offset is seen by both REF
% and UAV and cancels in tau_uav - tau_ref, but it still shifts the
% absolute PRS arrival time and therefore affects the TOA extraction.
offsetMinNs = -5000;
offsetMaxNs =  5000;

% Candidate AWGN power levels (normalized units, transmit power = 0 dB).
% One level is drawn per scenario.
noiseCandidates = [1e-14, 5e-14, 1e-13, 5e-13, 1e-12];

% --- Reproducibility ------------------------------------------------------
scenarioSeedBase = 42000; % Scenario i uses seed scenarioSeedBase + i
splitSeed = 2026;         % Seed of the train/val/test split

% --- Execution and outputs ----------------------------------------------
useParallel = true;       % Use a parallel pool if the toolbox is available
maxParallelWorkers = [];  % [] = default pool size
makeFigures = true;       % Save diagnostic figures
outputRoot = fullfile(repoRoot, 'results');

%% 2) LOAD THE DAB IQ WAVEFORM

txIQ_full = load_cf32_iq(iqFile);

fprintf('Loaded IQ file: %s\n', iqFile);
fprintf('Complex samples: %d\n', numel(txIQ_full));
fprintf('Duration: %.3f s\n', numel(txIQ_full)/fs);

% The first masterDurationSec seconds are used as the transmitted
% waveform of every TX. The segment is peak-normalized.
Nmaster = round(masterDurationSec * fs);
assert(numel(txIQ_full) >= Nmaster, 'IQ file shorter than requested master duration.');

masterIQ = double(txIQ_full(1:Nmaster));
masterIQ = masterIQ / max(abs(masterIQ) + eps);

% Identifier of the waveform, stored in every dataset row.
[~, iqName, iqExt] = fileparts(iqFile);
master_waveform_id_val = string([iqName iqExt]);

%% 3) DAB TIMING REFERENCE (NULL SYMBOL AND PRS)

% Locate the Null symbol and the following PRS in the master waveform.
% The PRS useful part is used as the matched-filter template.
dabTimingInfo = extract_dab_timing_reference(masterIQ, fs);
template = dabTimingInfo.prsTemplate;
toaSearchCenterLag = dabTimingInfo.prsStart - 1;

fprintf('DAB timing reference extracted:\n');
fprintf('- Null start sample : %d\n', dabTimingInfo.nullStart);
fprintf('- Null end sample   : %d\n', dabTimingInfo.nullEnd);
fprintf('- PRS start sample  : %d\n', dabTimingInfo.prsStart);
fprintf('- PRS length        : %d samples\n', dabTimingInfo.prsLen);
fprintf('- TOA search center : %d samples\n', toaSearchCenterLag);
fprintf('- Valid PRS centers : %d frames\n', numel(dabTimingInfo.prsSearchCenters));
fprintf('- Null energy ratio : %.4f\n', dabTimingInfo.nullEnergyRatio);

rxKeepLen = round(viewDuration_ms * 1e-3 * fs);
numTx = numel(txFreqHz);

%% 4) PARALLEL POOL (OPTIONAL)

% Results are identical with or without a pool, because every scenario
% seeds its own random generator. Without the Parallel Computing Toolbox
% the parfor loop below simply runs serially.
hasParallelToolbox = ~isempty(ver('parallel')) && ...
    license('test', 'Distrib_Computing_Toolbox');

if useParallel && hasParallelToolbox
    if isempty(gcp('nocreate'))
        if isempty(maxParallelWorkers)
            parpool;
        else
            parpool(maxParallelWorkers);
        end
    end
elseif useParallel
    fprintf('Parallel Computing Toolbox not available: running serially.\n');
end

%% 5) CONFIGURATION STRUCT FOR THE SCENARIO SIMULATION

cfg = struct();
cfg.numTx = numTx;
cfg.numMeas = numMeas;
cfg.numScenarios = numScenarios;
cfg.measDt = measDt;
cfg.areaMin = areaMin;
cfg.areaMax = areaMax;
cfg.minRefToTx = minRefToTx;
cfg.offsetMinNs = offsetMinNs;
cfg.offsetMaxNs = offsetMaxNs;
cfg.noiseCandidates = noiseCandidates;
cfg.scenarioSeedBase = scenarioSeedBase;
cfg.c = c;
cfg.fs = fs;
cfg.txFreqHz = txFreqHz;
cfg.masterIQ = masterIQ;
cfg.rxKeepLen = rxKeepLen;
cfg.template = template;
cfg.dabTimingInfo = dabTimingInfo;
cfg.searchMaxLag = searchMaxLag;
cfg.master_waveform_id_val = master_waveform_id_val;

% Empty row/summary structs that define the output columns.
cfg.rowTemplate = make_empty_dataset_row();
cfg.summaryTemplate = make_empty_summary_row();

%% 6) SIMULATE ALL SCENARIOS

datasetRowsByScenario = cell(numScenarios, 1);
summaryRows(numScenarios,1) = cfg.summaryTemplate;

parfor iscenario = 1:numScenarios
    [rowsThisScenario, summaryThisScenario] = simulate_scenario(iscenario, cfg);
    datasetRowsByScenario{iscenario} = rowsThisScenario;
    summaryRows(iscenario) = summaryThisScenario;
end

%% 7) BUILD TABLES

% datasetBalanced : full per-measurement table (all scenarios, all columns).
%                   The variable name is kept for compatibility with the
%                   saved MAT file; no class re-balancing is applied.
% scenarioSummary : one row per scenario.
datasetRows = vertcat(datasetRowsByScenario{:});
datasetBalanced = struct2table(datasetRows);
scenarioSummary = struct2table(summaryRows);

%% 8) AI-READY DATASET AND SCENARIO-WISE SPLIT

datasetAI = datasetBalanced(:, get_ai_variable_names());

% 70 / 15 / 15 split at the scenario level.
[trainSc, valSc, testSc, splitLabel] = split_scenarios(datasetAI.scenario_id, splitSeed);
datasetAI.split = splitLabel;

% Learning targets:
%   targetNames     : GN position error (true minus GN estimate, REF frame)
%   tdoaTargetNames : per-channel TDOA error (true minus estimated TDOA, s)
targetNames = {'gn_err_x','gn_err_y'};
tdoaTargetNames = {'delta_err_ch1','delta_err_ch2','delta_err_ch3','delta_err_ch4'};

% Columns that must not be used as model inputs: identifiers, labels,
% ground-truth-based quantities and simulation parameters.
excludedFeatureNames = [ ...
    {'scenario_id','meas_idx', ...
     'motion_type','motion_code', ...
     'geometry_class_est','observation_quality_class', ...
     'uav_rel_x','uav_rel_y', ...
     'gn_err_x','gn_err_y','gn_pos_err_m', ...
     'ctekf_err_x','ctekf_err_y','ctekf_pos_err_m', ...
     'noise_power', ...
     'tx_offset_ch1_ns','tx_offset_ch2_ns','tx_offset_ch3_ns','tx_offset_ch4_ns', ...
     'split'}, ...
    tdoaTargetNames];

% Candidate model inputs: all remaining numeric columns. The Python
% pipeline builds its own feature set from these columns.
featureNames = setdiff(datasetAI.Properties.VariableNames, excludedFeatureNames, 'stable');
featureNames = keep_numeric_table_variables(datasetAI, featureNames);

%% 9) SAVE OUTPUTS

timeStamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
outputFolder = fullfile(outputRoot, ['dab_soop_tdoa_dataset_' timeStamp]);

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

figFolder = fullfile(outputFolder, 'figures');
if makeFigures && ~exist(figFolder, 'dir')
    mkdir(figFolder);
end

csvBalanced = fullfile(outputFolder, 'dab_soop_tdoa_ctekf.csv');
csvSummary  = fullfile(outputFolder, 'dab_soop_tdoa_summary.csv');
csvAI       = fullfile(outputFolder, 'dab_soop_tdoa_ai_ready_gn_correction.csv');

matBalanced = fullfile(outputFolder, 'dab_soop_tdoa_gn_ctekf.mat');
matAI       = fullfile(outputFolder, 'dab_soop_tdoa_ai_ready_gn_correction.mat');

writetable(datasetBalanced, csvBalanced);
writetable(scenarioSummary, csvSummary);
writetable(datasetAI, csvAI);

save(matBalanced, ...
    'datasetBalanced', 'scenarioSummary', ...
    'txFreqHz', 'noiseCandidates', 'offsetMinNs', 'offsetMaxNs', ...
    'fs', 'masterDurationSec', 'numScenarios', 'numMeas', ...
    '-v7.3');

save(matAI, ...
    'datasetAI', 'featureNames', 'targetNames', 'tdoaTargetNames', ...
    'trainSc', 'valSc', 'testSc', ...
    '-v7.3');

disp(' ');
disp('Saved files:');
disp(['- ' csvBalanced]);
disp(['- ' csvSummary]);
disp(['- ' csvAI]);
disp(['- ' matBalanced]);
disp(['- ' matAI]);

%% 10) FIGURES

if makeFigures
    make_dataset_figures(datasetBalanced, scenarioSummary, figFolder);
    disp(' ');
    disp('Figures saved to:');
    disp(figFolder);
end

%% 11) GN VS CT-EKF PERFORMANCE TABLES (TABLES 2 AND 3 OF THE PAPER)

fprintf('\n\n');
fprintf('============================================================\n');
fprintf('FINAL RESULT TABLES: GN vs CT-EKF PERFORMANCE SUMMARY\n');
fprintf('============================================================\n');

[Table2_Overall_GN_CTEKF, Table3_Motion_GN_CTEKF] = ...
    compute_performance_tables(datasetBalanced, outputFolder);

fprintf('============================================================\n\n');

disp(' ');
disp('Dataset generation completed.');
