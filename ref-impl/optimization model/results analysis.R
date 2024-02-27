

####1 Open the GA results
####2 The number of planned promotions for each SKU
####3 The own price elasticiteis
####4 The cross-SKU price elasticities
####5 The cross-periods price ealsticities
####6 Visaliziation the results







####1 Open the GA results
###################  Inputs for the codes runngins

setwd("G:/skydrive/SkyDrive/projects/retail promotional causal net analysis/EJOR/code raw/") ### set the working directory
source("sku-functions.R")

source("./optimization model/ga-functions.R")

##### Input the store data location and file name
store.name<-c("./data/SKU data store 236117.Rda","./data/SKU data store 657979.Rda","./data/SKU data store 649405.Rda","./data/SKU data store 234212.Rda")



##### Input the store demand model parameters                        #################
######################################################################################

store.para.name<-c("./demand models with regularization/Store 236117 parameters.Rda","./demand models with regularization/Store 657979 parameters.Rda",
"./demand models with regularization/Store 649405 parameters.Rda","./demand models with regularization/Store 234212 parameters.Rda")
store.result.name<-c("./optimization model/results/result-store-236117.Rda","./optimization model/results/result-store-657979.Rda"
                    ,"./optimization model/results/result-store-649405.Rda","./optimization model/results/result-store-234212.Rda")



i.store=1

load(store.result.name[i.store])    #### save the opt results for current store
