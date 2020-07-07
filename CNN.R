###############################################################################
################################### PREPARO I DATI#############################
###############################################################################

source("functions.R")
library(keras)
library(imager)


dati <- list()
directory <- "./Images"
slices.tot <- 835
for(pat in list.files(directory)){
  cat(pat)
  cat("\n")
  count <- 1
  n_imgs <- length(list.files(paste(directory,pat,sep = "/",collapse = "")))
  dati[[pat]] <- array(data = 0,dim = c(640,480,n_imgs,3))
  for(slice in list.files(paste(directory,pat,sep = "/",collapse = ""))){
    cat(slice)
    cat("\n")
    tmp <- load.image(paste(paste(directory,pat,sep="/"),slice,sep = "/"))
    dati[[pat]][,,count,] <- resize(tmp,size_x = 640,size_y = 480,size_z = 1,size_c = 3)
    count <- count + 1
    slices.tot <- slices.tot + 1
  }
}

data.r <- prepare.data.for.cnn(file.dataset.clinico="./ROP.csv",
                               getROIvoxels=dati,slices.tot,resize_rows=640,resize_cols=480,
                               pat.id = "Eye_ID", clinical.outcome = "Treatment",
                               train.split = 0.9,
                               number.of.classes = 2)

############### COSTRUISCO L'ARCHITETTURA DELLA RETE###########################
########################, strides = c(2,2)#######################################################


use_session_with_seed(49,disable_parallel_cpu = FALSE)
#a linear stack of layers
model<-keras_model_sequential()
#configuring the Model
model %>%  
  layer_conv_2d(filters = 32, kernel_size = c(5,5), activation = 'relu', strides = c(2,2),
                input_shape=c(dim(data.r$train_x)[2],dim(data.r$train_x)[3],1)) %>% 
  layer_conv_2d(filters = 32, kernel_size = c(5,5), activation = 'relu', strides = c(2,2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(5,5), activation = 'relu', strides = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(5,5), activation = 'relu', strides = c(2,2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = 2, activation = 'softmax')

#Model's Optimizer
#defining the type of optimizer-ADAM-Adaptive Momentum Estimation
opt<-optimizer_adam( lr= 0.0001 , decay = 1e-6 )
#lr-learning rate , decay - learning rate decay over each update


###############################################################################
############### COMPILO IL MODELLO DELL RETE ##################################
###############################################################################
model %>%
  compile(loss="categorical_crossentropy",
          optimizer=opt,metrics = "accuracy")
#Summary of the Model and its Architecture
summary(model)
##

###############################################################################
############### ADDESTRO LA RETE ##############################################
###############################################################################
require(doParallel)
cl<-makeCluster(detectCores()-1)
registerDoParallel(cl)

history <- model %>% fit( data.r$train_x,data.r$train_y ,batch_size=25,
                          epochs=23,validation_data = list(data.r$test_x, data.r$test_y),
                          shuffle=TRUE,callbacks = list(callback_tensorboard("./logs/modelRPO_2"),
                                                        callback_model_checkpoint("modelRPO.h5")))

stopCluster(cl)
model$save("model-RPO.hdf5", overwrite = TRUE, include_optimizer = TRUE)
#save(data.r, file = "data.r.RData")

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################

library(keras)
library(caret)
model <- load_model_hdf5("model-RPO.hdf5")

model %>% evaluate(data.r$train_x,data.r$train_y)
model %>% evaluate(data.r$test_x,data.r$test_y)

preds <- model %>% predict_classes(data.r$test_x)

true <- array(data = 0, dim(data.r$test_y)[1])
true[which(data.r$test_y[,2]==1)] <- 1

table(true,preds)
confusionMatrix(as.factor(true),as.factor(preds),positive = "1")

ddd <- cbind(preds_prob,true)
set.seed(1)
rand <- sample(nrow(ddd))
ddd <- ddd[rand,]
ddd <- as.data.frame(ddd)
names(ddd) <- c("zero","one","Actual_class")
ddd <- ddd[order(ddd$Actual_class),]
ddd$Actual_class <- as.factor(ddd$Actual_class)

ggplot(ddd, aes(x=c(1:dim(ddd)[1]) ,y=one, color=Actual_class)) +
  geom_point(size = 3) + coord_cartesian(ylim = c(0,1)) + geom_hline(yintercept=.5) + 
  labs(title = "Predictions on testing and actual values",
       x = "Image instance", y = "Predicted probability of belonging to class 1")


#generate ROC
library(pROC)
library(caret)
model <- load_model_hdf5(filepath = "./model-RPO.hdf5")
preds_prob <- model %>% predict_proba(data.r$test_x)
preds_prob_1 <- preds_prob[,2]
true <- array(data = 0, dim(data.r$test_y)[1])
true[which(data.r$test_y[,2]==1)] <- 1
plot(roc(true,preds_prob_1),main = paste("ROC on test set (83 images). AUC:",round(auc(roc(true,preds_prob_1)),3)))
confusionMatrix(true,preds_prob_1)