
######## This program is to implement market share model for SKU sales forecasting


library(glmnet)
library(MASS)
setwd("G:/skydrive/SkyDrive/projects/retail promotional causal net analysis/EJOR/code raw/") ### First need to set the working directory of the codes

######## Inputs for running the programe

store.id=649405 ##store1: 236117 s2:657979 s3:649405 s4:234212    ####  input the store Id for forecasting

start<-11
ref=1                   ### which SKU in the category is used as reference
time.tr0<-c(11,251)     ### time window for training 
time.vd0<-c(252,331)    ### time window for validation

rolling.step<-8         ### the week cycle for planning and forecasting
rollings=10             ### rolling numbers in the experiments
rollingwindow=200       ### length of the rolling window
store.data<-paste0("./data/SKU data store ",store.id,".Rda") #### the location of the store data file


###########################################



load(store.data)
source("./demand model with top-down/attraction-functions.R")  ### some self-defined functions 
Cate_N<-length(SKU.data)-2;                                    ### number of categories in the store data

set.seed(0)
calib.time<-c(start,length(SKU.data[[1]][,1])-5)                 ### the start week and the lengh of total time 
t.steps<-seq(1,(time.vd0[2]-time.vd0[1]+1),by=rolling.step)    ### for rolling time steps

holiday<-c("Thanksgiving","Christmas", "NewYear", "X4thJuly","Labour")   ### holidays used in forecast the category level sales
X.time<-as.matrix(SKU.data$time)[,holiday]                               ### extract holiday indicators
x.time<-cbind(X.time[calib.time[1]:calib.time[2],],X.time[(calib.time[1]+1):(calib.time[2]+1),])
################################################
results.pre<-c()
subcate<-c()


for (i in 1:Cate_N){                               ### loop for each category

results.pre[[i]]<-matrix(0,SKU.data$Num[i],3)      ### saving the results         

Nsku<-SKU.data$Num[i]                              ### number of SKUs in the category

data<-get.datafull(i,subcate=1:Nsku,calib.time=calib.time)           ### get the predictors for all SKU
data.lag1<-get.datafull(i,subcate=1:Nsku,calib.time=(calib.time-1))  ### get the lag 1 

sku.pre<-c()
it<-1

for (t in t.steps){

time.tr<-time.tr0+t-1                               ### update the training window
time.vd<-c(time.tr[2]+1,time.tr[2]+rolling.step+1)  ### udate the validation window


data.train<-data.split(data,time.tr)                ### extract data for training
data.train.lag1<-data.split(data.lag1,time.tr)      ### extract data in lag one week for training


data.valid<-data.split(data,time.vd)                ### extract data for validation
data.valid.lag1<-data.split(data.lag1,time.vd)      ### extract data in lag one week for validation


x1.tr<-list(log(data.train.lag1$Y/rowSums(data.train.lag1$Y)),data.train$P,data.train.lag1$P)  ### predictors whose coefficients wanted to be heterogenous among SKUs
x2.tr<-list(data.train$D,data.train$F,data.train.lag1$D,data.train.lag1$F)                     ### predictors whose coefficients wanted to be homogeneous among SKUs


x1.vd<-list(log(data.valid.lag1$Y/rowSums(data.valid.lag1$Y)),data.valid$P,data.valid.lag1$P)
x2.vd<-list(data.valid$D,data.valid$F,data.valid.lag1$D,data.valid.lag1$F)


model.lm<-attraction.model(y=data.train$Y,x1=x1.tr,x2=x2.tr,ref=1)           ### estimate the  market share/ attraction model
para<-get.para(model.lm,Nsku,ref=ref,nx1=length(x1.tr),nx2=length(x2.tr))    ### get coefficients from the estimated market share/ attraction model

para$beta[is.na(para$beta)]<-0;para$alpha[is.na(para$alpha)]<-0;             
share.pre<-attraction.pre(x1=x1.vd,x2=x2.vd,para$y0,para$beta,para$alpha)    ### generate the forecasts for market share of each SKU in the category


data.agg.tr<-combine(cate.aggregate(data.train),cate.aggregate(data.train.lag1))    ### category aggregate level data for training
data.agg.vd<-combine(cate.aggregate(data.valid),cate.aggregate(data.valid.lag1))    ### category aggregate level data for validation


data.agg.tr$X<-cbind(data.agg.tr$X,x.time[time.tr[1]:time.tr[2],])           ### prepare the aggregate level predictors for training
data.agg.vd$X<-cbind(data.agg.vd$X,x.time[time.vd[1]:time.vd[2],])           ### prepare the aggregate level predictors for validation


model.sales<-cate.sales.model(y=data.agg.tr$Y,own=data.agg.tr$X)            ### category level aggregate sales forecasting model

sale.pre<-cate.sales.pre(model.sales,own=data.agg.vd$X)                     ### category level aggregate sales forecasts

sku.pre<-rbind(sku.pre,(as.vector(sale.pre)*share.pre)[1:rolling.step,])    ### top-down decomposition: sales of each SKU = category aggregate sales * market share of each SKU

}


data.full<-get.datafull(i,subcate=c(1:SKU.data$Num[i]),calib.time)           ### get the real sales in validation periods
data.full.valid<-data.split(data.full,time.vd0)

for (j in 1:SKU.data$Num[i]){
         
results.pre[[i]][j,]<-pre.accuracy(data.full.valid$Y[,j],sku.pre[,j])        ### calculate forecasting accuracy
}


cat(paste0("Forecasting for ","store: ",store.id, ", category: ",i),'\n') 
flush.console() 

}



pre.acu<-matrix(0,4,3)    ### the results analysis for Table 3 in the paper
k<-1
for (i in 1:4){
pre.acu[k,]<-colMeans(results.pre[[i]])
k<-k+1
}

write.table(pre.acu,"clipboard",sep="\t")

pre.acu


save(results.pre,file=paste0("./demand model with top-down/Store ",store.id, " predictions.Rda") )   ### save the forecasting results




