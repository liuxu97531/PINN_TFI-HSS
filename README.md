# PINN_TFI-HSS
The code for the paper [Temperature field inversion of heat-source systems via physics-informed neural networks](https://www.sciencedirect.com/science/article/abs/pii/S095219762200135X)

Temperature field inversion of heat-source systems (TFI-HSS) with limited observations is essential to monitor the system health. Although some methods such as interpolation have been proposed to solve TFI-HSS, those existing methods ignore correlations between data constraints and physics constraints, causing the low precision. In this work, we develop a physics-informed neural network-based temperature field inversion (PINN-TFI) method to solve the TFI-HSS task and a coefficient matrix condition number based position selection of observations (CMCN-PSO) method to select optimal positions of noisy observations. For the TFI-HSS task, the PINN-TFI method encodes constrain terms into the loss function and thus the task is transformed into an optimization problem of minimizing the loss function. In addition, we have found that noise significantly affect reconstruction performances of the PINN-TFI method. To alleviate the effect of noises in observations, we propose the CMCN-PSO method to find optimal positions, where the condition number of observations is used to evaluate positions. The results demonstrate that the PINN-TFI method can significantly improve prediction precisions and the CMCN-PSO method can find good positions to improve the robustness of the PINN-TFI method.


# Requirement

The exact TFI-HSS data in this paper can be obtained by the two following method
- The finite difference (see the FD_solver)
- The layout-generator


# Cite PINN_TFI-HSS
@article{liu2022temperature,
  
  title={Temperature field inversion of heat-source systems via physics-informed neural networks},
  
  author={Liu, Xu and Peng, Wei and Gong, Zhiqiang and Zhou, Weien and Yao, Wen},
  
  journal={Engineering Applications of Artificial Intelligence},
  
  volume={113},
  
  pages={104902},
  
  year={2022},
  
  publisher={Elsevier}
}
