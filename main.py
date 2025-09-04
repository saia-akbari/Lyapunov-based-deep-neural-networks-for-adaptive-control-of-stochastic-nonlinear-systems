import numpy as np
import pandas as pd
import json
import os
from dynamics import State_dynamics, target_dynamics, integrate
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Classes (State, DesiredTrajectory, Controller, Simulation) remain the same ---
# (Content omitted for brevity, no changes are needed in these classes)
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
        deep_mod1_input = lambda step, state, trajectory: np.concatenate([state.positions[:, step - 1]]).reshape(-1, 1)
        deep_mod2_input = lambda step, state, trajectory: np.concatenate([state.positions[:, step - 1], trajectory.positions[:, step - 1]]).reshape(-1, 1)
        deep_mod3_input = lambda step, state, trajectory: np.concatenate([state.positions[:, step - 1], trajectory.positions[:, step - 1]]).reshape(-1, 1)

        # Initialize all NNs
        self.deep_mod1 = NeuralNetwork(deep_mod1_input, config, "DNN1", config['num_inputs1'], config['num_outputs1'], config['num_layers1'], config['num_neurons1'], config['learning_rate1'], config['forgetting_factor1'])
        self.deep_mod2 = NeuralNetwork(deep_mod2_input, config, "DNN2", config['num_inputs2'], config['num_outputs2'], config['num_layers2'], config['num_neurons2'], config['learning_rate2'], config['forgetting_factor2'])
        self.deep_mod3 = NeuralNetwork(deep_mod3_input, config, "DNN3", config['num_inputs3'], config['num_outputs3'], config['num_layers3'], config['num_neurons3'], config['learning_rate3'], config['forgetting_factor3'])

    def get_nn_input(self, step, state, trajectory, nn):
        return nn.input_func(step, state, trajectory)

    def compute_deep_modular(self, step, tracking_error, desired_velocity, state, trajectory):
        deep_mod1_output = self.deep_mod1.compute_neural_network_output(step, tracking_error, self.get_nn_input(step, state, trajectory, self.deep_mod1))
        deep_mod2_output = self.deep_mod2.compute_neural_network_output(step, tracking_error, self.get_nn_input(step, state, trajectory, self.deep_mod2))
        deep_mod3_output = self.deep_mod3.compute_neural_network_output(step, tracking_error, self.get_nn_input(step, state, trajectory, self.deep_mod3))
        control_input = desired_velocity - self.ke * tracking_error - deep_mod1_output.reshape(-1) - 0.5 * tracking_error * deep_mod2_output.reshape(-1) - deep_mod3_output.reshape(-1)
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

    def run(self):
        tracking_error_norms = []
        for step in range(1, self.time_steps):
            current_time = step * self.time_step_delta
            tracking_error = self.State.positions[:, step - 1] - self.DesiredTrajectory.positions[:, step - 1]
            tracking_error_norms.append(np.linalg.norm(tracking_error))

            control_input = self.controller.compute_deep_modular(step, tracking_error, self.DesiredTrajectory.velocities[:, step - 1], self.State, self.DesiredTrajectory)

            self.State.update_dynamics(step, control_input, self.time_step_delta, self.final_time, self.mean, self.covariance, current_time)
            self.DesiredTrajectory.update_dynamics(step, self.time_step_delta, current_time)
            
        rms_error = np.sqrt(np.mean(np.square(tracking_error_norms)))
        return rms_error


def create_3d_surface_plot(excel_path):
    """
    Reads the simulation results from an Excel file and creates a 3D surface plot.
    """
    print(f"\nGenerating 3D surface plot from '{excel_path}'...")
    try:
        # Read the data from the excel file, setting the first column as the index
        df = pd.read_excel(excel_path, index_col=0)

        # Create the meshgrid for the plot
        # X values are the means (columns)
        # Y values are the covariances (index)
        X_vals = df.columns.astype(float).values
        Y_vals = df.index.astype(float).values
        X, Y = np.meshgrid(X_vals, Y_vals)
        
        # Z values are the RMS errors from the DataFrame
        Z = df.values

        # Create the plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface with a colormap and edges to mimic the example image
        # The 'terrain' colormap is similar to the one in the example
        surf = ax.plot_surface(X, Y, Z, cmap='terrain', rstride=1, cstride=1, alpha=0.9, linewidth=0.3, edgecolor='k')

        # Set labels and title
        ax.set_xlabel('MEAN', fontweight='bold')
        ax.set_ylabel('COVARIANCE', fontweight='bold')
        ax.set_zlabel('RMS OF ||e||', fontweight='bold')
        ax.set_title('RMS Tracking Error vs. Mean and Covariance', fontsize=16)

        # Set the viewing angle (elevation and azimuth)
        ax.view_init(elev=30, azim=-120)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=10, label='RMS Error Value')
        
        # --- MODIFIED LINES ---
        # 1. Change the filename to end with .svg
        plot_path = os.path.join(os.path.dirname(excel_path), '3d_surface_plot.svg')
        
        # 2. Save the figure in SVG format. The dpi argument is not needed for SVG.
        plt.savefig(plot_path, format='svg') 
        
        print(f"3D plot saved to '{plot_path}'")
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find the data file at {excel_path}. Cannot generate plot.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")


if __name__ == "__main__":
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    means = np.round(np.arange(-0.1, 0.11, 0.02), 2)
    covariances = np.arange(1, 11)

    results_df = pd.DataFrame(index=covariances, columns=means)
    results_df.index.name = 'Covariance'
    results_df.columns.name = 'Mean'

    output_dir = 'simulation_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'tracking_error_results.xlsx')

    # --- Run Simulations ---
    # (This entire block is unchanged)
    total_simulations = len(means) * len(covariances)
    current_simulation = 0
    print("Starting simulation runs for a range of means and covariances...")
    for cov in covariances:
        for mean in means:
            current_simulation += 1
            print(f"\nRunning simulation {current_simulation}/{total_simulations}: Mean = {mean}, Covariance = {cov}")
            
            current_config = config.copy()
            current_config['mean'] = mean
            current_config['covariance'] = cov
            
            sim = Simulation(current_config)
            rms_error = sim.run()
            
            results_df.loc[cov, mean] = rms_error
            print(f"--> Completed. RMS Tracking Error: {rms_error:.4f}")

    # --- Save Results and Generate Plot ---
    results_df.to_excel(output_path)
    print(f"\nAll simulations completed. Results saved to '{output_path}'")

    # Call the new function to create the plot from the saved results
    create_3d_surface_plot(output_path)