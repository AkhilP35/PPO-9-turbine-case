function [Wp, sol, sys] = Initial_9(wind_speed, wind_dir_deg)
    % Add required directories to the MATLAB path
    addpath('layoutDefinitions');
    addpath('solverDefinitions');

    % Defaults: match the layout defaults if not provided
    if nargin < 1 || isempty(wind_speed)
        wind_speed = 12.0214;
    end
    if nargin < 2 || isempty(wind_dir_deg)
        wind_dir_deg = 0.0;
    end

    % Select wind farm layout and define solver options
    Wp = layoutSet_sowfa_9turb_apc_alm_turbl();
    modelOptions = solverSet_default(Wp);

    % Start from a uniform flow field (1) or from a fully developed waked flow field (0).
    Wp.sim.startUniform = 0;

    % Convert (speed, direction) -> (u_Inf, v_Inf)
    % Convention:
    %   wind_dir_deg is the direction the wind is blowing TOWARD,
    %   measured counter-clockwise from +x (mathematical convention).
    theta = deg2rad(wind_dir_deg);
    Wp.site.u_Inf = wind_speed * cos(theta);
    Wp.site.v_Inf = wind_speed * sin(theta);

    % Store for debugging/logging (optional)
    Wp.user.wind_speed = wind_speed;
    Wp.user.wind_dir_deg = wind_dir_deg;

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
