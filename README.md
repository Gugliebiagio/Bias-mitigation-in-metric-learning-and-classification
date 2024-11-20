# Bias Mitigation in Deep Learning Models
### Project for Ethics in AI course - A.A. 2023/2024

The aim of this project is to study and assess fairness in the prediction of deep learning models, with the focus on two particular use cases:
Skin disease prediction and Face Recognition. 

Our work will address the following tasks:
1. Identification of preferences of the models towards specific categories, like gender and ethnicity.
2. Fine tuning of the models using bias mitigation adjusting algorithms to ensure fairness and impartiality. In particular, we will explore  different losses and the usage of a more balanced dataset for transfer learning.
3. Analysis and comment of the results.


Dataset
1. skin-disease-dataset available at https://dvcs.apice.unibo.it/aequitas/skin-desease-dataset, which contains images of skin lesions grouped into different diseases.
By inspecting the dataset, we identified 2 main problems to be tackled: unbalanced class distribution and unfair skin tones distribution among images.
2.  Fair Face Recognition Challenge dataset at ECCV 2020 available at https://chalearnlap.cvc.uab.cat/dataset/36/description/

All the training logs and the test results are in the relative notebooks (SkinDiseasesPrediction.ipynb and FaceRecogntion.ipynb). The repository contains also all the weights to reproduce our experiments and improve them.



