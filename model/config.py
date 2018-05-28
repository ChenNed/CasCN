import math
DATA_PATHA = "../data"


train_pkl = DATA_PATHA+"/data_train.pkl"
val_pkl = DATA_PATHA+"/data_val.pkl"
test_pkl = DATA_PATHA+"/data_test.pkl"
information = DATA_PATHA+"/information.pkl"



#parameters
observation = 3*60*60-1
print ("observation time",observation)
n_time_interval = 6
print ("the number of time interval:",n_time_interval)
time_interval = math.ceil((observation+1)*1.0/n_time_interval)#向上取整
print ("time interval:",time_interval)
lmax = 2
