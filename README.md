# data_augementation
    this hub is about how to make data augementation in seq2seqs model, when OCR model processes digital sequences, a very import 
issue is digital sequences not satisfied uniform distribution, the problem will be caused is OCR model cannot recognize partial digital sequences. for example, training the ten digital sequence 0000012345, the first five digits are beginning with 00000. when prediction data appears 1234500000, the training model can not correctly predict the sequence data. At this time, the best practice is to evenly distribute the data. The main functions are as follows:

![Image text](https://github.com/Qunstores/data_augementation/blob/master/date_trimed_data/01.jpg)

  a. Independent and identical distribution enhancement of ten-digit sequence data.
  
  b. Independent and identical distribution enhancement of date sequence data.
  
  c. the data format uses a normal distribution, such as image pixel resolution of 250*50, the augementation data with 250 as the mean, 10 for the variance of the length obey the normal distribution.
  
  d. in order to improve the robustness of OCR model, Gaussian noise and salt-and-pepper noise are added in the process of data enhancement.
  
  e. Randomly rotate 50% of the generated data to improve the generalization ability of the model.
  
  

Usage is as follows:

includes:

    --code_images  if not noe generate code data images
    
    --date_images  if not noe generate date data images
    
    --image_years  data deadline specified by date data images (2018-image_years)
    
    --image_nums   code data images numbers
    
    --noise_sigma  
    
    --use_rotate 
    
    --normal_mean 
    
	
usage:

   code data images generate:
   
   python data_augementation.py --code_images --image_nums 100
   
   date data images generate:
   
   python data_augementation.py --date_images --image_years 2019
   

