

setwd("G:/skydrive/SkyDrive/projects/retail promotional causal net analysis/EJOR/code raw/")  ### First need to set the working directory of the codes

######## Inputs for running the programe

store.id=236117 ##store1: 236117 s2:657979 s3:649405 s4:234212    ####  input the store Id for forecasting
start<-11   ### the time window of the data used in the experiments,IRI has 365 weeks (7 years)of the data, 
                        ### so we use week 11 to 365 in our experments, weeks before 11 used by some lags variables,e.g. sales moving average.
valid.time<-251         ### start of the week for validation
forward=8               ### the week cycle for planning and forecasting
rollings=10             ### rolling numbers in the experiments
rollingwindow=200       ### length of the rolling window
store.data<-paste0("./data/SKU data store ",store.id,".Rda") #### the location of the store data file

############################################################
#############################################################

source("sku-functions.R")   ### some self-defined funcitons used in the codes

load(store.data)            ### the loading of store data which stored in list named SKU.data

library(glmnet)
library(TTR)


Cate_N<-length(SKU.data)-2;  #### the number of categories in the store data

set.seed(0)

timetest=seq(1,forward*rollings,by=forward)      ### steps for rolling loops
calib.time<-c(start,length(SKU.data[[1]][,1]))   ### the start week and the lengh of total time 

results.pre<-list()     ### for store the prediction accuracy results
results.para<-list()    ### for store the model parameters which will be used for further optimization


for (i in 1:Cate_N){   #### start the main loop for each category

pre.comp=matrix(0,SKU.data$Num[i],6)     #### for tempary storing the prediction comparision results

para10.own=list()                    #### for tempary storing the parameters of SKU own parameters
para10.other=list()                  #### for tempary storing the parameters of SKU cross parameters

coef.limt<-10                        #### the experiential value to avoid obtain inormal parameter estimations
nsku<-SKU.data$Num[i]                #### number of the SKUs in the data

for (j in 1:nsku){                   #### start the loop for each SKUs in the category

para.own.t=list()                  #### store the SKU own parameters for each rolling steps
para.other.t=list()                #### store the SKU cross parameters for each rolling steps

Gset.prod<-getpart(i,j)            ### get the competive set for current SKU
X5<-get.set.full.P(Gset.prod)      ### get the predictors from the competive set


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

X.pc.d<-(1/price.discount(X.pc,calib.time))                      ### calculate the  relative price reduction, which is defined as price.median/price
X.pc.d[scale(X.pc.d)>=3]<-max(X.pc.d[-which(scale(X.pc.d)>=3)])  ### filter extra odinary values which are larger than 3 sigmas

X1<-cbind(X.pc.d,X.d1+X.d2,X.f11+X.f12+X.f21+X.f22)              

colnames(X1)<-c(paste0(colnames(SKU.data[[i]])[j+SKU.data$Num[i]*2],"discount"),
                colnames(SKU.data[[i]])[j+SKU.data$Num[i]*3],
                colnames(SKU.data[[i]])[j+SKU.data$Num[i]*5])

X.pc.lag<-(SKU.data[[i]][,j+SKU.data$Num[i]*2][(calib.time[1]-1):(calib.time[2]-1)]) ### price lag 1 week
X.d1.lag<-SKU.data[[i]][,j+SKU.data$Num[i]*3][(calib.time[1]-1):(calib.time[2]-1)]   ### display lag 1 week
X.d2.lag<-SKU.data[[i]][,j+SKU.data$Num[i]*4][(calib.time[1]-1):(calib.time[2]-1)]
X.f11.lag<-SKU.data[[i]][,j+SKU.data$Num[i]*5][(calib.time[1]-1):(calib.time[2]-1)]  ### feature lag 1 week
X.f12.lag<-SKU.data[[i]][,j+SKU.data$Num[i]*6][(calib.time[1]-1):(calib.time[2]-1)]
X.f21.lag<-SKU.data[[i]][,j+SKU.data$Num[i]*7][(calib.time[1]-1):(calib.time[2]-1)]
X.f22.lag<-SKU.data[[i]][,j+SKU.data$Num[i]*8][(calib.time[1]-1):(calib.time[2]-1)]

X.pc.lag.d<-1/price.discount(X.pc.lag,calib.time-1)    ## get the relative price reduction lag 1 week 

X1.lag<-cbind(X.pc.lag.d,X.d1.lag+X.d2.lag,X.f11.lag+X.f12.lag+X.f21.lag+X.f22.lag)
colnames(X1.lag)<-c(paste0(colnames(SKU.data[[i]])[j+SKU.data$Num[i]*2],"lag"),
                  paste0(colnames(SKU.data[[i]])[j+SKU.data$Num[i]*3],"lag"),
                  paste0(colnames(SKU.data[[i]])[j+SKU.data$Num[i]*5],"lag")
                )


X.part<-(cbind(X1,-X1.lag,-Y.lag1,X.time[,c(13:29)],Y.mov))        ### the set of used own predictors, X.time[,c(13:29)] includes all the holidays
Y.v<-Y[(valid.time+1):(valid.time+rollings*forward),]              ### the Y for forecasting validation

X.full<-cbind(X.part,X5)                                     ### full set of predictors

#####################

Y.pre.full.v0=rep(0,forward);                                ### for store rolling forecasts of the full model
Y.pre.lm.v0=rep(0,forward);                                  ### for store rolling forecasts of the lm model with only own predictors

t.index<-0                                                   ### indicator of the rolling step
########################################################################################

for (tt in timetest){                                        ### rolling loop

t.index<-t.index+1                                           ### update the rolling step indicator

#######################################################################################################

train.window<-(valid.time-rollingwindow-1+tt):(valid.time-1+tt)  #### update the rolling window for training
test.window<- (valid.time+tt):(valid.time+tt-1+forward)          #### update the rolling window for testing
                                   
X.full.c<-X.full[train.window,]                          
X.full.v<-X.full[test.window,]                          

X.part.c<-X.part[train.window,]                          
X.part.v<-X.part[test.window,]  

Y.c<-Y[train.window,]                                                


lower.part<-c(0,0,0,0,0,0,rep(-Inf,ncol(X.part)-6))                  ### define the lower limits for own predictors, prepared for sign constrained regression
cvfit.part <- cv.glmnet(X.part.c, (Y.c),lower=lower.part,alpha = 1)  ### first stage estimation

coef.test<-abs(as.vector(coef(cvfit.part,s="lambda.min")[-1]))>=coef.limt  ### test for inormal coeffiences

if(sum(coef.test)>=1) {                                              ### if existing of inormal coeffiences, then exclude it in the model,another solution could be prepare the data carefully
coef.exclude<-which(coef.test==TRUE)                                 ### inormal coeffiences is small probility event to happen, but very harmful for forecasts 
cvfit.part <- cv.glmnet(X.part.c, (Y.c),lower=lower.part,alpha = 1,exclude=coef.exclude)  ## if existing of inormal coeffiences, reestimate the model
}


Y.pre.rsd<- Y.c-predict(cvfit.part, newx=(X.part.c), s = "lambda.min")  ### in-sample error

lower=rep(c(0,0,0,-Inf),ncol(X.full.c[,-c(1:ncol(X.part))])/4)          ### define the lower limits for competitive predictors 

cvfit.full<- cv.glmnet(X.full.c[,-c(1:ncol(X.part))], Y.pre.rsd, lower=lower,alpha = 1)  ### second stage estimation

coef.test<-abs(as.vector(coef(cvfit.full,s="lambda.min")[-1]))>=coef.limt           ### inormal coefficients test

if(sum(coef.test)>=1) {
coef.exclude<-which(coef.test==TRUE)
cvfit.full<- cv.glmnet(X.full.c[,-c(1:ncol(X.part))], Y.pre.rsd, lower=lower,exclude=coef.exclude,alpha = 1)
}


Y.pre.full.v<-predict(cvfit.full, newx=X.full.v[,-c(1:ncol(X.part))], s ="lambda.min")+predict(cvfit.part, newx=X.part.v, s ="lambda.min")  ## generate forecasts of the rolling step
Y.pre.rsd.1<-(Y.pre.rsd)-predict(cvfit.full, newx=(X.full.c[,-c(1:ncol(X.part))]), s = "lambda.min")                                        ### in sample error for second stage


para.own.t[[t.index]]<-coef(cvfit.part,s="lambda.min")              ### coefficients of own predictors 
para.other.t[[t.index]]<-coef(cvfit.full,s="lambda.min")            ### coefficients of cross predictors 

Y.pre.full.v0[tt:(tt+forward-1)]<-Y.pre.full.v    ### store the forecasts


part.lm <-glmnet(X.part.c, (Y.c),alpha = 1,lambda=0)                ### estimated with OLS using own predictor, used as benchmark in the paper.

coef.test<-abs(as.vector(coef(part.lm)[-1]))>=coef.limt             ### inormal coefficients test, same with previous ones
if(sum(coef.test)>=1) {
coef.exclude<-which(coef.test==TRUE)
part.lm <- glmnet(X.part.c, (Y.c),alpha = 1,exclude=coef.exclude,lambda=0)
}

adj.lm<-mean(exp(Y.c-predict(part.lm, newx=(X.part.c))))             ### used for log back tranfrom adjustments

Y.pre.lm.v0[tt:(tt+forward-1)]<-predict(part.lm, newx=(X.part.v))    ### store the forecasts for OLS
}            ### end of rolling loop                                                          

adj.full<-mean(exp(Y.pre.rsd.1))                                     ### used for log back tranfrom adjustments
  
para10.own[[j]]=para.own.t                                           ### store the own coefficients of one SKU
para10.other[[j]]=para.other.t                                       ### store the cross coefficients of one SKU

pre.comp[j,]<-c(pre.accuracy(exp(Y.v),exp(Y.pre.full.v0)*adj.full),pre.accuracy(exp(Y.v),exp(Y.pre.lm.v0)*adj.lm)) ### forcasting accuracy comparisions

cat(paste0("Forecasting for ","store: ",store.id, ", category: ",i,", SKU: ",j),'\n') 
flush.console() 
}            ### end of the SKU loop

results.pre[[i]]<-pre.comp                                           ### store forcasting accuracy comparisions

results.para[[i]]<-list(para10.own,para10.other)                     ## store all the coefficients of all SKU

}            ### end of the category loop


save(results.pre,file=paste0("./demand models with regularization/Store ",store.id, " predictions.Rda") )   ### save the forecasting results
save(results.para,file=paste0("./demand models with regularization/Store ",store.id, " parameters.Rda") )   ### save the coefficients 


pre.acu<-matrix(0,length(SKU.data$Num),6)                            ### the results analysis for Table 3 in the paper
for (i in 1:length(SKU.data$Num)){
pre.acu[i,]<-colMeans(results.pre[[i]])
}
write.table(pre.acu,"clipboard",sep="\t")








