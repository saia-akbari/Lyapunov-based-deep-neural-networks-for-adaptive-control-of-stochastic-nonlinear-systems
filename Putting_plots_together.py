import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df_drift = pd.read_csv('C:\\Users\\akbaris\\Saia\\Papers\\[2024-TAC] [Submitted] - Lb Stochastic DNN\\Simulations\\Python\\Neural-Network-Tracking-Controller\\simulation_data\\data DNN only for drift\\state_data_drift.csv')
df_multi = pd.read_csv('C:\\Users\\akbaris\\Saia\\Papers\\[2024-TAC] [Submitted] - Lb Stochastic DNN\\Simulations\\Python\\Neural-Network-Tracking-Controller\\simulation_data\\data Multi-DNN\\state_data_multi.csv')

# Clean leading/trailing spaces from column names
df_drift.columns = df_drift.columns.str.strip()
df_multi.columns = df_multi.columns.str.strip()

# Set the font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Create the plot
plt.figure(figsize=(10, 6))

# Plot data from the multi Lb-DNNs file
plt.plot(df_multi['Time'], df_multi['Tracking_Error_Norm'], 
         label='Multi Lb-DNNs to compensate for drift and diffusion uncertainties')

# Plot data from the single Lb-DNN file
plt.plot(df_drift['Time'], df_drift['Tracking_Error_Norm'], 
         label='Single Lb-DNN to compensate for drift uncertainty')

# Add labels and title
plt.xlabel('Time (sec)')
plt.ylabel('Tracking Error Norm')
plt.title('Tracking Error Norm vs. Time')

# Add legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()