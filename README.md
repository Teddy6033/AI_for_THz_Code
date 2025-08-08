# AI for Terahertz Metasurface Sensing
This is the code repository accompanying the paper "DeepSeek-Motivated AI Physics Exploration: Mining Accidental BICs in Metasurfaces for Terahertz Sensing".

We summarize the proposed framework flowchart of AI-driven hierarchical physics inference in the diagram below. It enables profound insight into accidental bound states in the continuum (BICs) in metasurfaces, facilitating the design of advanced terahertz (THz) sensing functionalities by accidental q-BICs. The framework mainly includes three parts: (1) How Do Accidental BICs Happen? (2) Molecular Identification; (3) Quantitative Detection. 

<p align="center">
  <img width="800" height="640" alt="image" src="https://github.com/user-attachments/assets/37165f83-544b-4574-a08d-7e17959f0c49" />
</p>

* For Part (1), the core code is located in the folders `physical_mining_feature_prediction` and `physical_mining_shap_analysis`.
* For Part (2), the core code is located in the folders `bidirectional_forward_design_network` and `bidirectional_inverse_design_network`.
* For Part (3), the core code is located in the folders `spectrum_unloaded_prediction_network` and `spectrum_loaded_prediction_network`.
* The `prediction_tools` folder contains the UI interaction tools developed using PyQt5.
* The `auxiliary_tools` folder contains auxiliary utilities used during the project development, particularly those for data preprocessing.
* The `data_space` folder contains the dataset and related experimental results of this project. Due to their large size, they can be downloaded from [Google Drive](https://drive.google.com/file/d/10HxPnLU55VS_4dcNoTHQ1NRpp_2NSZ3a/view?usp=sharing).

##  Model Training 

The AI models in this project were trained locally using **Python 3.9**, the open-source deep learning framework **PyTorch 1.12.1**, and **CUDA 11.6**. The training and execution were conducted on a computer with the following hardware and system specifications:

- **Operating System**: Windows 11  
- **GPU**: NVIDIA GeForce RTX 3060Ti  
- **CPU**: Intel(R) Core(TM) i7-10700K @ 2.90 GHz  
- **RAM**: 16 GB  

##  Running the Project
* It is recommended to run this project using the same or more advanced libraries, tools, and environment as mentioned above.
* To launch the GUI tool located in the `prediction_tools` folder, please install the required dependency `pyqt5`.
* To avoid potential path-related issues, it is strongly recommended to place the entire project directory under `D:\DL\` .


