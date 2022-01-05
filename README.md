# HSI-LIDAR-fusion-using-DNN


The goal of this research project is to develop a deep learning-based architecture to fuse Hyperspectral and LiDAR data for achieving accurate classification maps. 

The project contains three following stages: 
- A short report is provided which reviewing the literature of the fusion of Hyperspectral and LiDAR and briefly discusses both conventional and deep learning-based methodologies.
- A new fusion architecture was implemented for Hyperspectral and LiDAR fusion. 
- The performance of the developed method is compared with the state-of-the-art in terms of classification accuracies. Additionally, sensitivity analysis concerning the selection of parameters/ Hyperparameters is provided.


System-specific notes

The data were generated and preprocessed by Matlab R2016a or higher versions, and the codes of various networks were tested on PyTorch 1.6.0 version in Python 3.7 on Windows 10 machines.

The orginal HS-Lidar data used in this project can be downloaded from

http://hyperspectral.ee.uh.edu/2egf4tg8hial13gt/2013_DFTC.zip

❗ Note: If you would like to use these data, you have to cite the related works, otherwise, you will be trapped into the copyright issues. Please pay more attention to it.

How to use it?

Here an example experiment is given by using Houston2013 hyperspectral and LiDAR data. Directly run main.py to produce the results. 

If you want to run the code in your own data, you can accordingly change the input (e.g., data, labels) and tune the parameters.

If you encounter the bugs while using this code, please do not hesitate to contact us.
