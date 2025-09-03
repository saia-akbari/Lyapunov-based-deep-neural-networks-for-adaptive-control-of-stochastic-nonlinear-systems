import numpy as np
from dynamics import State_dynamics, target_dynamics, integrate
from neural_network import NeuralNetwork
from data_manager import save_state_data, save_nn_data, generate_comparison_plot
import json

class State:
    def __init__(self, num_states, time_steps):
        self.positions = np.zeros((num_states, time_steps))
        self.positions[:,0] = np.array([2, -1, 2, -1, 2])
        self.velocities = np.zeros((num_states, time_steps))

    def update_dynamics(self, step, control_input, time_step_delta, final_time, mean, covariance, current_time):
        self.velocities[:, step] = State_dynamics(self.positions[:, step - 1], step, time_step_delta, final_time, control_input, mean, covariance, current_time)
        integrate(step, self.positions, self.velocities, time_step_delta)

class DesiredTrajectory:
    def __init__(self, num_states, time_steps):
        self.positions = np.zeros((num_states, time_steps))
        self.velocities = np.zeros((num_states, time_steps))

    def update_dynamics(self, step, time_step_delta, current_time):
        self.velocities[:, step] = target_dynamics(self.positions[:, step - 1], step, current_time)
        integrate(step, self.positions, self.velocities, time_step_delta)

class Controller:
    def __init__(self, config):
        self.ke = config['ke']

        # Define inputs for all potential NNs
        deep_non_mod_input = lambda step, state, trajectory: np.concatenate([state.positions[:, step - 1], trajectory.positions[:, step - 1]]).reshape(-1, 1)
        deep_mod1_input = lambda step, state, trajectory: np.concatenate([state.positions[:, step - 1]]).reshape(-1, 1)
        deep_mod2_input = lambda step, state, trajectory: np.concatenate([state.positions[:, step - 1], trajectory.positions[:, step - 1]]).reshape(-1, 1)
        deep_mod3_input = lambda step, state, trajectory: np.concatenate([state.positions[:, step - 1], trajectory.positions[:, step - 1]]).reshape(-1, 1)

        # Initialize all NNs
        self.deep_mod1 = NeuralNetwork(deep_mod1_input, config, "DNN1", config['num_inputs1'], config['num_outputs1'], config['num_layers1'], config['num_neurons1'], config['learning_rate1'], config['forgetting_factor1'])
        self.deep_mod2 = NeuralNetwork(deep_mod2_input, config, "DNN2", config['num_inputs2'], config['num_outputs2'], config['num_layers2'], config['num_neurons2'], config['learning_rate2'], config['forgetting_factor2'])
        self.deep_mod3 = NeuralNetwork(deep_mod3_input, config, "DNN3", config['num_inputs3'], config['num_outputs3'], config['num_layers3'], config['num_neurons3'], config['learning_rate3'], config['forgetting_factor3'])
        self.deep_drift = NeuralNetwork(deep_mod1_input, config, "DNN1", config['num_inputs1'], config['num_outputs1'], config['num_layers1'], config['num_neurons1'], config['learning_rate1'], config['forgetting_factor1'])

    def get_nn_input(self, step, state, trajectory, nn):
        return nn.input_func(step, state, trajectory)

    def compute_deep_modular(self, step, tracking_error, desired_velocity, state, trajectory):
        deep_mod1_output = self.deep_mod1.compute_neural_network_output(step, tracking_error, self.get_nn_input(step, state, trajectory, self.deep_mod1))
        deep_mod2_output = self.deep_mod2.compute_neural_network_output(step, tracking_error, self.get_nn_input(step, state, trajectory, self.deep_mod2))
        deep_mod3_output = self.deep_mod3.compute_neural_network_output(step, tracking_error, self.get_nn_input(step, state, trajectory, self.deep_mod3))
        control_input = desired_velocity - self.ke * tracking_error - deep_mod1_output.reshape(-1) - 0.5 * tracking_error * deep_mod2_output.reshape(-1) - deep_mod3_output.reshape(-1)
        return control_input

    def compute_deep_drift(self, step, tracking_error, desired_velocity, state, trajectory):
        deep_drift_output = self.deep_drift.compute_neural_network_output(step, tracking_error, self.get_nn_input(step, state, trajectory, self.deep_drift))
        control_input = desired_velocity - self.ke * tracking_error - deep_drift_output.reshape(-1)
        return control_input
    
    def compute_no_DNN(self, step, tracking_error, desired_velocity, state, trajectory):
        control_input = desired_velocity - self.ke * tracking_error #- deep_drift_output.reshape(-1)
        return control_input

class Simulation:
    def __init__(self, config):
        self.final_time = config['final_time']
        self.time_step_delta = config['time_step_delta']
        self.time_steps = int(self.final_time / self.time_step_delta)
        self.num_states = config['num_states']
        self.mean = config['mean']
        self.covariance = config['covariance']
        
        self.State = State(self.num_states, self.time_steps)
        self.DesiredTrajectory = DesiredTrajectory(self.num_states, self.time_steps)
        self.controller = Controller(config)

    def run(self, model_type):
        for step in range(1, self.time_steps):
            current_time = step * self.time_step_delta
            tracking_error = self.State.positions[:, step - 1] - self.DesiredTrajectory.positions[:, step - 1]

            # --- Select and compute control input based on model_type ---
            if model_type == 'deep_modular':
                control_input = self.controller.compute_deep_modular(step, tracking_error, self.DesiredTrajectory.velocities[:, step - 1], self.State, self.DesiredTrajectory)
            elif model_type == 'deep_drift':
                control_input = self.controller.compute_deep_drift(step, tracking_error, self.DesiredTrajectory.velocities[:, step - 1], self.State, self.DesiredTrajectory)
            else:
                control_input = self.controller.compute_no_DNN(step, tracking_error, self.DesiredTrajectory.velocities[:, step - 1], self.State, self.DesiredTrajectory)

            # Update dynamics
            self.State.update_dynamics(step, control_input, self.time_step_delta, self.final_time, self.mean, self.covariance, current_time)
            self.DesiredTrajectory.update_dynamics(step, self.time_step_delta, current_time)

            # --- Save data with a model-specific identifier ---
            save_state_data(step, current_time, tracking_error, self.num_states, model_type)
            
            if model_type == 'deep_modular':
                save_nn_data(step, current_time, self.controller.deep_mod1, model_type)
                save_nn_data(step, current_time, self.controller.deep_mod2, model_type)
                save_nn_data(step, current_time, self.controller.deep_mod3, model_type)
            elif model_type == 'deep_drift':
                save_nn_data(step, current_time, self.controller.deep_drift, model_type)
            
            print(f"\rProgress ({model_type}): {step/self.time_steps*100:.2f}%", end='', flush=True)

if __name__ == "__main__":
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # --- Run Deep Modular Simulation ---
    print("Running Deep Modular simulation...")
    sim_modular = Simulation(config)
    sim_modular.run(model_type='deep_modular')
    print("\nDeep Modular simulation completed.")

    # --- Run Deep Drift Simulation ---
    print("\nRunning Deep Drift simulation...")
    sim_drift = Simulation(config)
    sim_drift.run(model_type='deep_drift')
    print("\nDeep Drift simulation completed.")
    
    # --- Run No DNN Simulation ---
    print("\nRunning No DNN simulation...")
    sim_drift = Simulation(config)
    sim_drift.run(model_type='no_DNN')
    print("\nNo DNN simulation completed.")

    # --- Generate Comparison Plot ---
    print("\nGenerating comparison plot...")
    generate_comparison_plot()