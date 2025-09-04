[Read Me.txt](https://github.com/user-attachments/files/22141783/Read.Me.txt)
This repository contains the code and simulations for the research on Lyapunov-based Deep Neural Networks (Lb-DNNs) for the adaptive control of stochastic nonlinear systems. The project explores the effectiveness of Lb-DNN controllers in compensating for uncertainties in both drift and diffusion terms.

This repository is organized into two main branches, each focusing on a different aspect of the controller's evaluation.

Comparison-with-baselines:
	This branch provides a comparative analysis of different control 	strategies to demonstrate the effectiveness of the proposed Lb-DNN 	approach. It includes three distinct simulations:
	Simulation 1: Multi Lb-DNN Controller
	This simulation showcases the performance of the fully developed 	multi Lb-DNN controller, which is designed to compensate for 	uncertainties in both the drift and diffusion terms of the 	stochastic system.
	Simulation 2: Single Lb-DNN Controller (Drift Uncertainty Only)
	This simulation evaluates a simplified Lb-DNN controller that 	utilizes only one DNN to compensate for drift uncertainty, allowing 	for an analysis of the contribution of diffusion compensation.
	Simulation 3: Simple Feedback Controller
	This simulation assesses a baseline feedback controller without any 	DNN components. The results from this simulation serve as a 	benchmark to highlight the performance improvements gained 	from the Lb-DNN architectures.
Mean-Covariance-Variations:
	This branch is dedicated to evaluating the robustness and tracking 	performance of the developed Lb-DNN controller under varying 	stochastic noise conditions.
	Simulation 4: Response to Noise Variations
	This simulation illustrates the controller's tracking performance 	when subjected to changes in the mean and covariance of the 	stochastic noise, demonstrating its adaptability and stability.

