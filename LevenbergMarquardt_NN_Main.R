##########################################################
##
##
## Written by D. Nguyen Kien, 
##    Civil Engineering College
##    Tongji University
##    Shanghai, China
##    E-mail: nkdung@hotmail.com 
##
## Hanoi, May 2020
##
##
##########################################################




##########################################################
##         INITIALIZATION
##########################################################

# layer=====
input_nodes = 1   # n
hidden_nodes = 3  # k
out_nodes = 1     # m

layer = c(input_nodes, hidden_nodes, out_nodes)

# sigmoid=====
af1 = "sigmoid"
af2 = "linear"
af = c(af1, af2)



# sgd method=====
learning_rate = 0.1

#lm method=====
OptMethod = "lm"   # LM is very unstable method --> WHY ?????????????
mu = 100
beta = 10
error_min = 0.005
criteria_lm = c(1e-3, 1e-3, 1e-1, 1e-1)


epochs = 50


check_query = "yes"


# initialization in weight=====
wih = matrix(rnorm(layer[1]*layer[2], 0, layer[2]^(-0.5)), nrow=layer[1])
who = matrix(rnorm(layer[2]*layer[3], 0, layer[3]^(-0.5)), nrow=layer[2])




##########################################################
##         LOAD DATA
##########################################################
x = seq(-2,2, by =0.1)   # vector
y = 1+sin(pi*x/4)        # vector with the same size with x
x_train = x
y_train = y



##########################################################
##         TESTING DATA
##########################################################
x_test = seq(-1.95, 1.98, by=0.12)
y_test = 1+sin(pi*x_test/4)




##########################################################
##         TRAIN
##########################################################

# define array
MSE_train = numeric(epochs)
sumMSE_train_LM = numeric(epochs* length(x_train) )

mu_array = numeric(epochs)
mu_array[1] = mu

sumMSE_train_error = 0

flag = 0    #
i0 = 0


for(e in 1:epochs){
  for (i in 1: length(x_train)){
    input_data = x_train[i]
    target_data = y_train[i]
    
    
    # choose Optimization Method ===================================================
    if ((OptMethod=="sgd") || (OptMethod =="Sgd") || (OptMethod =="SGD")){
      met = gradient_descent (input_data, target_data, learning_rate, wih, who, af)
    } else if ((OptMethod=="lm") || (OptMethod =="Lm") || (OptMethod =="LM")){
      met = LevenbergMarquardt(input_data, target_data, mu, wih, who, af)
    }
    
    wih = met$wih_update
    who = met$who_update
    sumMSE_train_error = sumMSE_train_error + (met$out_error)^2

    
    
    # When to stop LM Opitmization method ===============================
    if ((OptMethod=="lm") || (OptMethod =="Lm") || (OptMethod =="LM")){
      
      # # Convergence checking-------------------  
      if (met$criteria_1 < criteria_lm[1]) {
        flag = 1
        i0 = i
        break
      }
      
    } ##### end of LM method
    
    
  }  ##### end for (i in 1: length(x_train))
  
  
  MSE_train[e] = 1/ (length(x_train) * length(x_train[i])) * sumMSE_train_error #CORRECT ????????
  
  
  if (e>1) {    # CORRECT ??????????????????
    if (MSE_train[e] < MSE_train[e-1]) {
      mu = mu/beta
    }
    else if (MSE_train[e] > MSE_train[e-1]){
      mu = mu*beta
    }
    else if (MSE_train[e] == MSE_train[e-1]){
      mu = mu
    }
    mu_array[e] = mu
  }
  
  
  # # in case of convergence check met----> Do we NEED ????????????????????  # &&&&&&&&&&&&&&&&
  if (i0 >0){
    MSE_train[e] = 1/ (i0 * length(x_train[i])) * sumMSE_train_error
  }
  
  sumMSE_train_error = 0
  
  
  # To STOP the code if the error meets the error_min
  # However it is not efficient. Sometime mu gets Inf or wih, who get Inf or NaN ?????????
  epochs_end = e
  if ((MSE_train[epochs_end] < error_min) || (flag==1)){
    MSE_train_plot = MSE_train[1:epochs_end]
    break
  }
  else{
    MSE_train_plot = MSE_train[1:epochs_end]
  }
  
  
}  ##### end for(e in 1:epochs)





##########################################################
##         PLOT RESULT
##########################################################
par(mar=c(1,1,1,1))
par(mfrow=c(3,1))



##########################################################
##         TEST
##########################################################
if (check_query =="yes"){
  y_query = numeric(length(x_test))
  MSE_test = numeric(length(x_test))
  
  for (i in 1: length(x_test)){
    y_query[i] = query(x_test[i], wih, who, af)
  }
  
  plot(x_train, y_train, type= "l", col = "black" )
  # par(new=TRUE)
  lines(x_test, y_query, type = "p", col = "magenta")
  
  

  y_query_train = numeric(length(x_train))
  for (i in 1: length(x_train)){
    y_query_train[i] = query(x_train[i], wih, who, af)
  }
  lines(x_train, y_query_train, type = "p", col = "blue")
  
}



##########################################################
##         PLOT RESULT
##########################################################

# # MSE  plot
# plot(1:epochs_end, MSE_train_plot, type = "b", frame = TRUE, pch = 20,
#      col = "purple", xlab = "epoch", ylab = "log10(MSE)")

# Log10 MSE plot
plot(1:epochs_end, log10(MSE_train_plot), type = "b", frame = TRUE, pch = 20,
     col = "red", xlab = "epoch", ylab = "log10(MSE)")

# mu plot
# mu_array_plot = mu_array[1:epochs_end]
plot(1:epochs_end, mu_array[1:epochs_end], col = "green")
