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





#=================================================================
rm(list = ls())  # clear variables
graphics.off()  # clear graphics
cat("\f") # clear screen
#=================================================================



# ====== PACKAGES ================================================
#
# install.packages("pracma") ---> get size function
# install.packages("matlib") ---> get inverse of matrix quickly
library("pracma")
#=================================================================




# Diag function definition ===================================
diag_D = function(mat_){ #matrix's dimension is [1xn]
  mat2 = diag(length(mat_))
  diag(mat2) = mat_
  
  return (mat2)
}



# Activation Function ========================================
activation = function(af, int){
  if ((af=="sigmoid") || (af=="Sigmoid")){
    output = 1/ (1+exp(-int))
    der_out = (output * (1-output))
  }else if((af=="tanh") || (af=="Tanh")){
    ouput = (exp(int)-exp(-int))/(exp(int)+exp(-int))
    der_out = (1-output*output)    # production is correct ????
  } else if ((af=="linear") || (af=="Linear")){
    output = int
    der_out = diag(length(int))  # ONE matrix, same size with "int"
  }
  
  return (list(activ=output, der_activ=der_out))
}




# Gradient Descent ===========================================
gradient_descent = function(inp, target, lr, wih, who, af){
  input_hidden = inp %*% wih      #[1xn x nxk] = [1xk]
  output_hidden = activation(af[1], input_hidden)$activ    #[1xk] 
  der_hidden = activation(af[1], input_hidden)$der_activ   #[1xk]
  
  input_out = output_hidden %*% who     #[1xk x kxm] = [1xm]
  output_out = activation(af[2], input_out)$activ        #[1xm] 
  der_out = activation(af[2], input_out)$der_activ       #[1xm]
  
  D2 = diag_D(der_out)      #[m x m]
  D1 = diag_D(der_hidden)   #[k x k]
  
  error = t(output_out - target)   #[m x 1]
  
  delta2 = D2 %*% error;     #[m x 1]
  delta1 = (D1 %*% who) %*% delta2;    #[k x 1] = [kxk  x  kxm  x mx1]
  
  who_ = who - lr * (t(output_hidden) %*% t(delta2))  #[kx1  x 1xm] = [kxm]
  wih_ = wih - lr * (t(inp) %*% t(delta1))    #[nx1 x 1xk] = [nxk]
  
  return (list(wih_update = wih_, who_update = who_, out_error = error ))
}





# Levenberg-Marquardt ==========================================
# LM training - Tariq, page 110-111    & Hagan paper
LevenbergMarquardt = function(inp, target, mu, wih, who, af){
  
  input_hidden = inp %*% wih      #[1xn x nxk] = [1xk]
  output_hidden = activation(af[1], input_hidden)$activ    #[1xk] 
  der_hidden = activation(af[1], input_hidden)$der_activ   #[1xk]
  
  input_out = output_hidden %*% who     #[1xk x kxm] = [1xm]
  output_out = activation(af[2], input_out)$activ        #[1xm] 
  der_out = activation(af[2], input_out)$der_activ       #[1xm]
  
  D2 = diag_D(der_out)      #[m x m]
  D1 = diag_D(der_hidden)   #[k x k]
  
  error = t(output_out - target)   #[m x 1]
  
  # gradient = Je ---------------------------------------------------
  delta2 = D2 %*% error;     #[m x 1]
  delta1 = (D1 %*% who) %*% delta2;    #[k x 1] = [kxk  x  kxm  x mx1]
  
  gradient_who = (t(output_hidden) %*% t(delta2))  #[kx1  x 1xm] = [kxm]
  gradient_wih = (t(inp) %*% t(delta1))    #[nx1 x 1xk] = [nxk]
  
  
  # J calculation ---------------------------------------------------  
  delta2_LM = D2 ;     #[m x 1]
  delta1_LM = (D1 %*% who) %*% delta2_LM;    #[k x 1] = [kxk  x  kxm  x mx1]
  
  # Look at the paper of Hagan and Menhaj 1994
  J_who = t((t(output_hidden) %*% t(delta2_LM)))   # or equals (delta2_LM %*% output_hidden), [m x k]
  J_wih = t((t(inp) %*% t(delta1_LM)))   # or equals (delta1_LM %*% inp), [k x n]
  
  
  # Update values --------------------------------------------------- 
  # # Look at the paper of Yu and Wilamowski, 2011
  #           ( [kxk]  =[kxm] * [mxk] + [kxk] ) * [kxm]     = [kxm]
  who_p = inv(t(J_who)%*%J_who + mu*diag(length(output_hidden)) ) %*% gradient_who
  who_ = who - who_p
  
  
  #           ( [nxn]  =[nxk] * [kxn] + [nxn] ) * [nxk]     = [nxk] ???
  wih_p = inv(t(J_wih)%*%J_wih + mu*diag(length(inp)) ) %*% gradient_wih
  wih_ = wih - wih_p
  

  # Convergence check ------------------------------------------------
  criteria_1 = max(norm(gradient_who), norm(gradient_wih))  
  
  
  return (list(wih_update = wih_, who_update = who_, out_error = error, out_update = output_out,
               criteria_1_update = criteria_1))
}





# QUERY ========================================================
query = function(inp_test, wih, who, af){
  input_hidden = inp_test %*% wih      #[1xn x nxk] = [1xk]
  output_hidden = activation(af[1], input_hidden)$activ    #[1xk] 
  
  input_out = output_hidden %*% who     #[1xk x kxm] = [1xm]
  output_out = activation(af[2], input_out)$activ        #[1xm] 
  
  return (output_out)
}

