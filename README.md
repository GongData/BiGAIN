# Bidirectional_f-Divergence-Based_GAN_Imputation  <br>
Paper : Bidirectional f-Divergence-Based Deep Generative Method for
Imputing Missing Values in Time Series Data  <br>
environment: tensorflow version 1.7 python 2.7

(https://www.mdpi.com/2571-905X/8/1/7)


### Physionet Dataset <br>

__Stage1:__ <br>
Go to Gan_Imputation folder:  <br>
Execute the `Physionet_main.py` file and include the parameter, then we will get 4 folders named as "checkpoint" (the saved models), G_results (the generated samples), imputation_test_results (the imputed test dataset) and imputation_train_results (the imputed train dataset).  <br>

For example, <br>
if you want to use Rverse KL as loss function of the GAN, <br>
please run `Physionet_main.py --loss-type Reverse_KL` . <br>
If you want to use Jensen-Shannon(JS) as loss function of the GAN, <br>
please run `Physionet_main.py --loss-type JS` .<br> 


__Stage2:__ <br>
Go to GRUI floder: <br>
Execute the `Run_GAN_imputed.py` file, then one folder-"checkpoint_physionet_imputed" will be created, go to the "checkpoint_physionet_imputed/30_8_128_64_0.001_400_True_True_True_0.15_0.5" folder, find "result" file, the "result" file stands for the mortality prediction results by The RNN classifier trained on the GAN imputed dataset. The first column is epoch, the second column is accuracy and the last column is the AUC score. <br>




### KDD Dataset <br>

__Stage1:__ <br>
Go to Gan_Imputation folder:  <br>
Execute the `kdd_main.py` file and include the parameter, then we will get 3 folders named as "checkpoint" (the saved models),  imputation_test_results (the imputed test dataset) and imputation_train_results (the imputed train dataset).  <br>

For example, <br> 
if you want to use use Reverse KL as loss function of the GAN, <br>
please run `kdd_main.py --loss-type Reverse_KL` . <br>
If you want to use Jensen-Shannon(JS) as loss function of the GAN, <br>
please run `kdd_main.py --loss-type JS` .<br> 


__Stage2:__ <br>
Go to GRUI floder: <br>
Execute the `Run_GAN_imputed.py` file, then one folder-"checkpoint_physionet_imputed" will be created, go to the "checkpoint_physionet_imputed/30_8_128_64_0.001_400_True_True_True_0.15_0.5" folder, find "result" file, the "result" file stands for the mortality prediction results by The RNN classifier trained on the GAN imputed dataset. The first column is epoch, the second column is accuracy and the last column is the AUC score. <br>




### Original Work <br>
This code is adopted and modified from Luo et al.’s original implementation in their paper, *“Multivariate Time Series Imputation with Generative Adversarial Networks”* . Our modifications include integrating an f-divergence-based loss function and extending the architecture to a bidirectional GRUI framework.

Paper https://papers.nips.cc/paper_files/paper/2018/hash/96b9bff013acedfb1d140579e2fbeb63-Abstract.html <br>
Github https://github.com/Luoyonghong/Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks <br>

