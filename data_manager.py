import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def save_state_data(step, time, tracking_error, num_states, model_type):
    # Calculate norms
    tracking_error_norm = np.linalg.norm(tracking_error[:num_states])

    # Create dataframe with state data
    state_data = pd.DataFrame({
        'Time': time,
        'Tracking_Error_Norm': tracking_error_norm,
    }, index=[step])

    # Create directory if it doesn't exist
    os.makedirs('simulation_data', exist_ok=True)

    # Define file path for state data using model_type
    state_file_path = f'simulation_data/state_data_{model_type}.csv'

    # For the first step, delete existing file and write new one with headers
    if step == 1:
        if os.path.exists(state_file_path):
            os.remove(state_file_path)
        state_data.to_csv(state_file_path, mode='w', index=False)
    else:
        # For subsequent steps, append without writing the header
        state_data.to_csv(state_file_path, mode='a', header=False, index=False)

def save_nn_data(step, time, nn, model_type):
    nn_output = nn.neural_network_output
    neural_network_weights = nn.weights
    num_outputs = nn.num_outputs
    nn_id = nn.label
    
    # Create dataframe with neural network data
    
    nn_data = pd.DataFrame({
        'Time': time,
        **{f'NN_Output_{i}': nn_output[i] for i in range(num_outputs)},
        **{f'Weight_{i}': w for i, w in enumerate(neural_network_weights)}
    }, index=[step])

    # Create directory if it doesn't exist
    os.makedirs('simulation_data', exist_ok=True)

    # Define file path using nn_id and model_type
    nn_file_path = f'simulation_data/{nn_id.lower()}_{model_type}_data.csv'

    # For the first step, delete existing file and write new one with headers
    if step == 1:
        if os.path.exists(nn_file_path):
            os.remove(nn_file_path)
        nn_data.to_csv(nn_file_path, mode='w', index=False)
    else:
        # For subsequent steps, append without writing the header
        nn_data.to_csv(nn_file_path, mode='a', header=False, index=False)

def generate_comparison_plot():
    """
    Reads the state data from the 'deep_modular' and 'deep_drift' runs
    and plots their tracking error norms on the same figure for comparison.
    """
    try:
        # Define file paths
        modular_path = 'simulation_data/state_data_deep_modular.csv'
        drift_path = 'simulation_data/state_data_deep_drift.csv'
        noDNN_path = 'simulation_data/state_data_no_DNN.csv'
        
        # Load the datasets
        df_multi = pd.read_csv(modular_path)
        df_drift = pd.read_csv(drift_path)
        df_noDNN = pd.read_csv(noDNN_path)

        # Set plot aesthetics
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Times New Roman"
        
        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot data from both simulations
        plt.plot(df_multi['Time'], df_multi['Tracking_Error_Norm'], label='Multi Lb-DNNs')
        plt.plot(df_drift['Time'], df_drift['Tracking_Error_Norm'], label='Single Lb-DNN for drift uncertainty')
        plt.plot(df_drift['Time'], df_noDNN['Tracking_Error_Norm'], label='No Lb-DNN')

        # Add labels, title, and legend
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Tracking Error Norm', fontsize=14)
        plt.title('Tracking Error Norm Comparison', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)

        # Save and display the plot
        plot_path = 'simulation_data/comparison_plot.svg'
        plt.savefig(plot_path, format='svg')
        print(f"Comparison plot saved to {plot_path}")
        plt.show()

        # Calculate and print RMS errors for both models
        rms_error_multi = np.sqrt(np.mean(df_multi['Tracking_Error_Norm'] ** 2))
        rms_error_drift = np.sqrt(np.mean(df_drift['Tracking_Error_Norm'] ** 2))
        rms_error_noDNN = np.sqrt(np.mean(df_noDNN['Tracking_Error_Norm'] ** 2))
        
        print(f"\nRMS Error (Deep Modular): {rms_error_multi:.3f}")
        print(f"RMS Error (Deep Drift): {rms_error_drift:.3f}")
        print(f"RMS Error (No DNN): {rms_error_noDNN:.3f}")

    except FileNotFoundError as e:
        print(f"Error: Could not find data file. Make sure both simulations ran successfully. Details: {e}")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")