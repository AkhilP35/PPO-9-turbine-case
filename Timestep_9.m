function [power] = Timestep_9(sim_time, phi, CT_prime)
    % Define and load the previous simulation state
    if sim_time == 0
        filename = 'output0.mat';
    else
        filename = sprintf('output%d.mat', sim_time-1);
    end
    load(filename, 'Wp', 'sol', 'sys');

    % Set solver options and turbine control inputs
    modelOptions = solverSet_default(Wp);
    turbInput = struct('t', sim_time);
    turbInput.phi = phi;            % Yaw angle settings
    turbInput.CT_prime = CT_prime;  % Thrust coefficient settings

    % Advance the simulation by one time step
    [sol, sys] = WFSim_timestepping(sol, sys, Wp, turbInput, modelOptions);

    % Save the current state for the next time step
    save_path = sprintf('/Users/akhilpatel/Desktop/Dissertation/WFSim-master/output%d.mat', sim_time);
    save(save_path, 'Wp', 'sol', 'sys');

    % Extract power output
    power = sol.turbine.power;
end