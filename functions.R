############### DA QUI #############

###############################################################################
################################### FUNZIONI #################################
###############################################################################

library(keras)


#funzione per preparare i dati da inserire nella rete
prepare.data.for.cnn <- function(file.dataset.clinico, getROIvoxels,pat.id, clinical.outcome,all_slices,
                                 train.split, number.of.classes, resize_cols = NA, resize_rows= NA){

  #carico il dataset con l'outcome
  dataset_clinico <- read.csv(file.dataset.clinico, stringsAsFactors=FALSE)
  dataset_clinico[,pat.id] <- as.character(dataset_clinico[,pat.id])
  
  #carico i getROIVoxels e li associo all'outcome
  # load(rdata.getROIvoxels)
  gtv <- getROIvoxels
  gtv <- gtv[which(as.numeric(names(gtv)) %in% as.numeric(dataset_clinico[,pat.id]))]



  #sistemo il dataset nella forma giusta (tensor[batch,nrow,ncol,channels])
  gtv_resized_tens <- array( dim = c(all_slices,resize_rows,resize_cols,1))
  outcome <- array(dim = c(all_slices))
  count <- 1
  ddd <- list()
  fff <- 1
  pat_cs_bySlice <- array()
  for(pat in names(gtv)){
    oo <- dataset_clinico[,clinical.outcome][which(as.numeric(dataset_clinico[,pat.id]) == as.numeric(pat))]
    for(j in 1:dim(gtv[[pat]])[3]){
      gtv_resized_tens[count,1:resize_rows,1:resize_cols,1] <- gtv[[pat]][,,j,1]
      pat_cs_bySlice[count] <- pat
      outcome[count] <- oo
      count <- count + 1
    }
    fff <- fff +1
  }


  #le classi devono cominciare con "0"
  #if(length(outcome_bigroi[which(outcome_bigroi==0)])==0){outcome_bigroi <- outcome_bigroi-1}

  #divido in test e train
  dim_train <- round(dim(gtv_resized_tens)[1]*train.split)
  dim_test <- round(dim(gtv_resized_tens)[1]*(1-train.split))
  train_x <- array( dim = c(dim_train,resize_rows,resize_cols,1))
  test_x <- array( dim = c(dim_test,resize_rows,resize_cols,1))
  #index_train <- sample(size = dim_train, x = 1:dim(gtv_resized_tens)[1])
  index_train <- c(1:dim_train)
  
  train_x[,,,1] <- gtv_resized_tens[index_train,,,]
  train_y <- outcome[index_train]
  train_y <- train_y[which(!is.na(train_y))]
  pat_cs_bySlice <- pat_cs_bySlice[index_train]
  test_x[,,,1]  <- gtv_resized_tens[-index_train,,,]
  test_y <- outcome[-index_train]
  test_y <- test_y[which(!is.na(test_y))]

  train_y<-to_categorical(train_y,num_classes = number.of.classes)
  test_y<-to_categorical(test_y,num_classes = number.of.classes)
  
  data.ready <- list("train_x"=train_x,"train_y"=train_y, "test_x"=test_x, "test_y"=test_y, "resize_x"=resize_rows,"resize_y"=resize_cols,"index_train"=index_train, "pat_cs_bySlice"=pat_cs_bySlice)
  return(data.ready)
  
}

###### CLASSIFICARE PER PAZIENTI (da wrappare in una funzione)

# classification <- c()
# for(m in names(table(data.test[,1]))) {
#   sub.m <- subset.matrix(x = data.test, subset = data.test[,1] == m)
#   P <- sum(as.numeric(sub.m[,2] == sub.m[,3])/nrow(sub.m))
#   if (P > .5) classification <- c(classification, 1)
#   if (P <= .5) classification <- c(classification, 0)
# }
