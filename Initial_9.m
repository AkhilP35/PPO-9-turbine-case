function [Wp, sol, sys] = Initial_9()
    % Add required directories to the MATLAB path
    addpath('layoutDefinitions');
    addpath('solverDefinitions');
    
    % Select wind farm layout and define solver options
    Wp = layoutSet_sowfa_9turb_apc_alm_turbl(); 
    modelOptions = solverSet_default(Wp); 

    % Start from a uniform flow field (1) or from a fully developed waked flow field (0).
    Wp.sim.startUniform = 0;
    
    % Set display and visualization preferences
    verboseOptions.printProgress = 1;    % Show progress in command window
    verboseOptions.Animate       = 0;    % Enable flow field animation
    verboseOptions.plotMesh      = 0;    % Disable mesh plotting
    
    % Initialize simulation environment and core variables
    run('WFSim_addpaths.m'); 
    [Wp, sol, sys] = InitWFSim(Wp, modelOptions, verboseOptions.plotMesh); 
    
    % Save the initial state (Step 0) to the designated output folder
    save('/Users/akhilpatel/Desktop/Dissertation/WFSim-master/output0.mat', 'Wp', 'sol', 'sys');
end
