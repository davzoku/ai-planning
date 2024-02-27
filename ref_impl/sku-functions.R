

pre.accuracy<-function(Y,Y.pre){    #### function for forecasting accuracy evaluation
  
  MAE<-mean(abs(Y-Y.pre))
  MSE<-((sum((Y-Y.pre)^2))/length(Y))
  MASE<-MAE/mean(abs(Y[2:length(Y)]-Y[1:(length(Y)-1)]))
  MPE<-sum(Y-Y.pre)/sum(Y)
  return(cbind(MAE,MASE,MPE))
  
}


get.price.list<-function(cate){     ####  given a category, output the prices in(0.2,0.35,0.5,0.65,0.8) quantile for each SKU in the category


price=(SKU.data[[cate]][calib.time[1]:calib.time[2],(SKU.data$Num[cate]*2+1):(SKU.data$Num[cate]*3)])

price.list<-matrix(0,ncol(price),5)
for (i in 1:ncol(price)){  
price.list[i,]<-quantile(unique(price[,i]),c(0.2,0.35,0.5,0.65,0.8))
}

price.list
}


getpart<-function(i,j){
getpart=c()
 prod.n=c(1:SKU.data$Num[i])[-j]
  for(m in prod.n){
   getpart<-c(getpart,i,m)
}

return(getpart)
}



get.price.avr<-function(cate){    ### obtain the average price in the data for each SKU in the data

  price=(SKU.data[[cate]][calib.time[1]:calib.time[2],(SKU.data$Num[cate]*2+1):(SKU.data$Num[cate]*3)])
  
 (colMeans((price)))
 
 }


get.set.full.P<-function(Gset.prod){    ### give the set of the SKUs, output the predictors of all these SKUs
  
  full<-c()
 
  for (i in 1:(length(Gset.prod)/2)){
    
    k<-Gset.prod[2*i-1]
    m<-Gset.prod[2*i]
    
   
    X.pc<-(SKU.data[[k]][,m+SKU.data$Num[k]*2][calib.time[1]:calib.time[2]])
    X.d1<-SKU.data[[k]][,m+SKU.data$Num[k]*3][calib.time[1]:calib.time[2]]
    X.d2<-SKU.data[[k]][,m+SKU.data$Num[k]*4][calib.time[1]:calib.time[2]]
    X.f11<-SKU.data[[k]][,m+SKU.data$Num[k]*5][calib.time[1]:calib.time[2]]
    X.f12<-SKU.data[[k]][,m+SKU.data$Num[k]*6][calib.time[1]:calib.time[2]]
    X.f21<-SKU.data[[k]][,m+SKU.data$Num[k]*7][calib.time[1]:calib.time[2]]
    X.f22<-SKU.data[[k]][,m+SKU.data$Num[k]*8][calib.time[1]:calib.time[2]]
    X.f1=X.f11+X.f12;X.f2=X.f21+X.f22
    X.sale.lag<-log(SKU.data[[k]][,m][(calib.time[1]-1):(calib.time[2]-1)])
    X.sale.lag2<-log(SKU.data[[k]][,m][(calib.time[1]-2):(calib.time[2]-2)])
    X.pc.d<--1/price.discount(X.pc,calib.time)
    X.pc.d[abs(scale(X.pc.d))>=3]<-max(X.pc.d[-which(abs(scale(X.pc.d))>=3)])
    X.pc.lag<-1/(SKU.data[[k]][,m+SKU.data$Num[k]*2][(calib.time[1]-1):(calib.time[2]-1)])
    X.d1.lag<-SKU.data[[k]][,m+SKU.data$Num[k]*3][(calib.time[1]-1):(calib.time[2]-1)]
    X.d2.lag<-SKU.data[[k]][,m+SKU.data$Num[k]*4][(calib.time[1]-1):(calib.time[2]-1)]
    X.f11.lag<-SKU.data[[k]][,m+SKU.data$Num[k]*5][(calib.time[1]-1):(calib.time[2]-1)]
    X.f12.lag<-SKU.data[[k]][,m+SKU.data$Num[k]*6][(calib.time[1]-1):(calib.time[2]-1)]
    X.f21.lag<-SKU.data[[k]][,m+SKU.data$Num[k]*7][(calib.time[1]-1):(calib.time[2]-1)]
    X.f22.lag<-SKU.data[[k]][,m+SKU.data$Num[k]*8][(calib.time[1]-1):(calib.time[2]-1)]
    X.f1.lag=X.f11.lag+X.f12.lag;X.f2.lag=X.f21.lag+X.f22.lag



    X1<-cbind(X.pc.d,-(X.d1+X.d2),-(X.f1+X.f2), -X.sale.lag)
    colnames(X1)<-c(colnames(SKU.data[[k]])[m+SKU.data$Num[k]*2],
                    colnames(SKU.data[[k]])[m+SKU.data$Num[k]*3],
                    colnames(SKU.data[[k]])[m+SKU.data$Num[k]*5],
                    colnames(SKU.data[[k]])[m])


    X1.lag<-cbind(X.pc.lag,(X.d1.lag+X.d2.lag),(X.f1.lag+X.f2.lag),X.sale.lag)
    colnames(X1.lag)<-c(paste0(colnames(SKU.data[[k]])[m+SKU.data$Num[k]*2],".lag"),
                   paste0(colnames(SKU.data[[k]])[m+SKU.data$Num[k]*3],".lag"),
                   paste0(colnames(SKU.data[[k]])[m+SKU.data$Num[k]*5],".lag"),
                   paste0(colnames(SKU.data[[k]])[m],".lag"))
                   
   full<-cbind(full,X1)
                                   }  
  
  return(full)
  
  
    }


######################################


price.discount<-function(pc,calib.time){  ### given the prices of a SKU, output relative discount defined by price/price.median
time.years<-c(rep(1:7,each=52),7)[calib.time[1]:calib.time[2]]

price.median<-c()

for (tt in 1:7){

lower<-max(pc[time.years==tt])*0.95
price.median<-c(price.median,rep(median(pc[time.years==tt][pc[time.years==tt]>=lower]),sum(time.years==tt)))
}

pc.disc<-pc/price.median

return(cbind(pc.disc))
}



#####



get.index<-function(cate,j,calib.time){    ####### given a category cate, SKU j, time window calib.time, output the promotional indexes exclude the focal SKU j.
  
  sales=(SKU.data[[cate]][(calib.time[1]):(calib.time[2]),1:SKU.data$Num[cate]])[,-j]
  sales.ind=(SKU.data[[cate]][1:250,1:SKU.data$Num[cate]])[,-j]
  weight=colMeans(sales.ind)/sum(colMeans(sales.ind))
  weight=weight/sum(weight)
  price=(SKU.data[[cate]][calib.time[1]:calib.time[2],(SKU.data$Num[cate]*2+1):(SKU.data$Num[cate]*3)])[,-j]
  price.index=log(colSums(t(price)*weight))

  sales.lag=(SKU.data[[cate]][(calib.time[1]-1):(calib.time[2]-1),1:SKU.data$Num[cate]])[,-j]
  sales.index=log(rowMeans(sales.lag))
  
 
  display.1=SKU.data[[cate]][calib.time[1]:calib.time[2],(SKU.data$Num[cate]*3+1):(SKU.data$Num[cate]*4)][,-j]
  display.2=SKU.data[[cate]][calib.time[1]:calib.time[2],(SKU.data$Num[cate]*4+1):(SKU.data$Num[cate]*5)][,-j]
  display.index=(colSums(t(display.1+2*display.2)*weight))

  feature.1=SKU.data[[cate]][calib.time[1]:calib.time[2],(SKU.data$Num[cate]*5+1):(SKU.data$Num[cate]*6)][,-j]
  feature.2=SKU.data[[cate]][calib.time[1]:calib.time[2],(SKU.data$Num[cate]*6+1):(SKU.data$Num[cate]*7)][,-j]
  feature.3=SKU.data[[cate]][calib.time[1]:calib.time[2],(SKU.data$Num[cate]*7+1):(SKU.data$Num[cate]*8)][,-j]
  feature.4=SKU.data[[cate]][calib.time[1]:calib.time[2],(SKU.data$Num[cate]*8+1):(SKU.data$Num[cate]*9)][,-j]
  feature.index=(colSums(t(feature.1+1*feature.2+2*feature.3+2*feature.4)*weight))
 
return(cbind(sales.index,price.index, display.index, feature.index))
 }




get.price.regular<-function(pc,calib.time){   ####### given a category cate, SKU j, time window calib.time, output the regular prices which defined as median price which large than 0.95*max(price) in a year.

time.years<-c(rep(1:7,each=52),7)[calib.time[1]:calib.time[2]]

price.median<-c()

for (tt in 1:7){
lower<-max(pc[time.years==tt])*0.95
price.median<-c(price.median,rep(median(pc[time.years==tt][pc[time.years==tt]>=lower]),sum(time.years==tt)))
}

return(price.median)
}




