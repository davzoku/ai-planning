



setwd("G:/skydrive/SkyDrive/projects/retail promotional causal net analysis/EJOR/code raw/")  ### First need to set the working directory of the codes

######## Inputs for running the programe

store.id=649405 ##store1: 236117 s2:657979 s3:649405 s4:234212    ####  input the store Id for forecasting
start<-11   ### the time window of the data used in the experiments,IRI has 365 weeks (7 years)of the data, 
                        ### so we use week 11 to 365 in our experments, weeks before 11 used by some lags variables,e.g. sales moving average.
valid.time<-251         ### start of the week for validation
forward=8               ### the week cycle for planning and forecasting
rollings=10             ### rolling numbers in the experiments
rollingwindow=200       ### length of the rolling window
store.data<-paste0("./data/SKU data store ",store.id,".Rda") #### the location of the store data file

############################################################

source("sku-functions.R")

load(store.data)

library(glmnet)
library(TTR)


Cate_N<-length(SKU.data)-2;  #### the number of categories in the data

set.seed(0)

timetest=seq(1,forward*rollings,by=forward)      ### steps for rolling loops
calib.time<-c(start,length(SKU.data[[1]][,1]))   ### the start week and the lengh of total time 

##################################################################################################################################

################################################
results.pre<-list()     ### for store the prediction results
results.para<-list()    ### for store the model parameters which will be used for further optimization


for (i in 1:length(SKU.data$Num)){   #### start the loop for categories


pre.comp=matrix(0,SKU.data$Num[i],3)     #### for tempary storing the prediction comparision results

para10.own=list()                    #### for tempary storing the parameters of SKU own parameters
para10.other=list()                  #### for tempary storing the parameters of SKU cross parameters

coef.limt<-10                        #### the experiential value to avoid obtain inormal parameter estimations
                   

for (j in 1:SKU.data$Num[i]){      #### start the loop for each SKUs in the category

Gset.prod<-getpart(i,j)            ### get the competive set for current SKU
X5<-cbind(get.index(i,j,calib.time),get.index(i,j,calib.time-1))      ### get the promotion intense indexes from the competive set


Y <- as.matrix(log(SKU.data[[i]][,j][calib.time[1]:calib.time[2]]))    ### the units of sales
colnames(Y)<-colnames(SKU.data[[i]])[j]


Y.mov<-log(SMA(SKU.data[[i]][,j],8)[(calib.time[1]-1):(calib.time[2]-1)])  #### moving average of Y in last 8 weeks

X.pc<-(SKU.data[[i]][,j+SKU.data$Num[i]*2][calib.time[1]:calib.time[2]])   ### price of the focal SKU
X.d1<-SKU.data[[i]][,j+SKU.data$Num[i]*3][calib.time[1]:calib.time[2]]     ### display 1 of the focal SKU
X.d2<-SKU.data[[i]][,j+SKU.data$Num[i]*4][calib.time[1]:calib.time[2]]     ### display 2 of the focal SKU
X.f11<-SKU.data[[i]][,j+SKU.data$Num[i]*5][calib.time[1]:calib.time[2]]    ### feature type 1 of the focal SKU
X.f12<-SKU.data[[i]][,j+SKU.data$Num[i]*6][calib.time[1]:calib.time[2]]    ### feature type 2 of the focal SKU
X.f21<-SKU.data[[i]][,j+SKU.data$Num[i]*7][calib.time[1]:calib.time[2]]    ### feature type 3 of the focal SKU
X.f22<-SKU.data[[i]][,j+SKU.data$Num[i]*8][calib.time[1]:calib.time[2]]    ### feature type 4 of the focal SKU

X.time<-as.matrix(SKU.data$time)[calib.time[1]:calib.time[2],]             ### calender events

Y.lag1<-as.matrix(log(SKU.data[[i]][,j][(calib.time[1]-1):(calib.time[2]-1)])) ### the units of sales in last week
colnames(Y.lag1)<-paste0(colnames(SKU.data[[i]])[j],"lag_",1)

X.pc.d<-(1/price.discount(X.pc,calib.time))                       ## get the relative price with regular price
X.pc.d[scale(X.pc.d)>=3]<-max(X.pc.d[-which(scale(X.pc.d)>=3)])  ### filter extra odinary values which are larger than 3 sigmas

X1<-cbind(X.pc.d,X.d1+X.d2,X.f11+X.f12+X.f21+X.f22)          

colnames(X1)<-c(paste0(colnames(SKU.data[[i]])[j+SKU.data$Num[i]*2],"regular"),
                colnames(SKU.data[[i]])[j+SKU.data$Num[i]*3],
                colnames(SKU.data[[i]])[j+SKU.data$Num[i]*5])

X.pc.lag<-(SKU.data[[i]][,j+SKU.data$Num[i]*2][(calib.time[1]-1):(calib.time[2]-1)]) ### price in last week
X.d1.lag<-SKU.data[[i]][,j+SKU.data$Num[i]*3][(calib.time[1]-1):(calib.time[2]-1)]
X.d2.lag<-SKU.data[[i]][,j+SKU.data$Num[i]*4][(calib.time[1]-1):(calib.time[2]-1)]
X.f11.lag<-SKU.data[[i]][,j+SKU.data$Num[i]*5][(calib.time[1]-1):(calib.time[2]-1)]
X.f12.lag<-SKU.data[[i]][,j+SKU.data$Num[i]*6][(calib.time[1]-1):(calib.time[2]-1)]
X.f21.lag<-SKU.data[[i]][,j+SKU.data$Num[i]*7][(calib.time[1]-1):(calib.time[2]-1)]
X.f22.lag<-SKU.data[[i]][,j+SKU.data$Num[i]*8][(calib.time[1]-1):(calib.time[2]-1)]

X.pc.lag.d<-1/price.discount(X.pc.lag,calib.time-1)    ## get the regular price which is defined as median of price in each year

X1.lag<-cbind(X.pc.lag.d,X.d1.lag+X.d2.lag,X.f11.lag+X.f12.lag+X.f21.lag+X.f22.lag)
colnames(X1.lag)<-c(paste0(colnames(SKU.data[[i]])[j+SKU.data$Num[i]*2],"lag"),
                  paste0(colnames(SKU.data[[i]])[j+SKU.data$Num[i]*3],"lag"),
                  paste0(colnames(SKU.data[[i]])[j+SKU.data$Num[i]*5],"lag")
                )

Y.v<-Y[(valid.time+1):(valid.time[1]+rollings*forward),]              ### the Y for forecasting validation

X.full<-cbind(X1,-X1.lag,-Y.lag1,X.time[,c(13:29)],Y.mov,X5[,-5])                      ### full set of predictors

#####################

Y.pre.full.v0=rep(0,forward);                                ### for store rolling forecasts of the full model

t.index<-0                                                   ### indicator of the rolling step
########################################################################################

for (tt in timetest){                                        ### rolling loop

t.index<-t.index+1                                           ### update the rolling time indicator

#######################################################################################################


                                   

train.window<-(valid.time-rollingwindow-1+tt):(valid.time-1+tt)  #### update the rolling window for training
test.window<- (valid.time+tt):(valid.time+tt-1+forward)          #### update the rolling window for testing
                                   
X.full.c<-X.full[train.window,]                          
X.full.v<-X.full[test.window,]                          

Y.c<-Y[train.window,]                                             


cvfit.full<- cv.glmnet(X.full.c, Y.c)  ### 

coef.test<-abs(as.vector(coef(cvfit.full,s="lambda.min")[-1]))>=coef.limt           ### inormal coefficients test

if(sum(coef.test)>=1) {
coef.exclude<-which(coef.test==TRUE)
cvfit.full<- cv.glmnet(X.full.c, Y.c, exclude=coef.exclude)
}

Y.pre.full.v<-predict(cvfit.full, newx=X.full.v,s="lambda.min")

Y.pre.full.v0[tt:(tt+forward-1)]<-Y.pre.full.v    ### store the forecasts
Y.pre.rsd.1<-Y.c-predict(cvfit.full, newx=X.full.c,s="lambda.min")

}            ### end of rolling loop                                                          

adj.full<-mean(exp(Y.pre.rsd.1))                                     ### used for log back tranfrom adjustments

pre.comp[j,]<-pre.accuracy(exp(Y.v),exp(Y.pre.full.v0)*adj.full) ### forcasting accuracy comparisions

cat(paste0("Forecasting for ","store: ",store.id, ", category: ",i,", SKU: ",j),'\n') 
flush.console() 
}

results.pre[[i]]<-pre.comp                                           ### store forcasting accuracy comparisions

}


save(results.pre,file=paste0("./demand model with promotion indense index/Store ",store.id, " predictions.Rda") )   ### save the forecasting results

pre.acu<-matrix(0,length(SKU.data$Num),3)                            ### the results analysis for Table 3 in the paper
for (i in 1:length(SKU.data$Num)){
pre.acu[i,]<-colMeans(results.pre[[i]])
}
write.table(pre.acu,"clipboard",sep="\t")


pre.acu



