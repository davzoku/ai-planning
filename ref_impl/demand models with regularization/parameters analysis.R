

source("sku-functions.R")
setwd("G:/skydrive/SkyDrive/projects/retail promotional causal net analysis/EJOR/code raw/")
store.id=234212 ##store1: 236117 s2:657979 s3:649405 s4:234212

load(paste0("./demand models with regularization/Store ",store.id, " parameters.Rda") )   ### save the coefficients
store.data<-paste0("./data/SKU data store ",store.id,".Rda")
load(store.data)




cate=1  #########
nsku=length(para10.own1)
rollings=10
# parameter analysis

para10.own<-list()
para10.other<-list()

for (j in 1:nsku){

para10.own[[j]]<-results.para[[cate]][[1]][[j]][[1]]
para10.other[[j]]<-results.para[[cate]][[2]][[j]][[1]]
for (t in 2:rollings){
para10.own[[j]]<-para10.own[[j]]+(para10.own1[[j]][[t]])
para10.other[[j]]<-para10.other[[j]]+para10.other1[[j]][[t]]

}

para10.own[[j]]<-para10.own[[j]]/10  #### get the rolling average
para10.other[[j]]<-para10.other[[j]]/10    #### get the rolling average

}




price.para<-c();
display.para<-c();
feature.para<-c();

for (j in 1:nsku){

parameter.own<-para10.own[[j]]
price.para<-c(price.para,parameter.own[2])
display.para<-c(display.para,parameter.own[3])
feature.para<-c(feature.para,parameter.own[4]);

}

price.avg<-get.price.avr(cate)
price.elastics.own<-  -price.para/price.avg                 ## own price elastic at average price

price.index<-c(FALSE,rep(c(TRUE,FALSE,FALSE,FALSE),nsku-1)) ### indicator of price coefficients
feature.index<-c(FALSE,rep(c(FALSE,FALSE,TRUE,FALSE),nsku-1)) ### indicator of feature coefficients

price.elastics.other<-matrix(0,nsku,nsku)                     ##### price elastics matrix
feature.elastics.other<-matrix(0,nsku,nsku)                   ##### feature elastics matrix



for (j in 1:nsku){

price.elastics.other[j,j]<-price.elastics.own[j]
feature.elastics.other[j,j]<-feature.para[j]
parameter.other<-as.vector(para10.other[[j]])

if (j==1){
price.elastics.other[j,(j+1):nsku]<-parameter.other[price.index]/price.avg[(j+1):nsku]
feature.elastics.other[j,(j+1):nsku]<--parameter.other[feature.index]
}else{
if (j==nsku){
price.elastics.other[j,1:(nsku-1)]<-parameter.other[price.index]/price.avg[1:(nsku-1)]
feature.elastics.other[j,1:(nsku-1)]<--parameter.other[feature.index]
}else{
price.elastics.other[j,1:(j-1)]<-parameter.other[price.index][1:(j-1)]/price.avg[1:(j-1)]
price.elastics.other[j,(j+1):nsku]<-parameter.other[price.index][(j):(nsku-1)]/price.avg[(j+1):nsku]
feature.elastics.other[j,1:(j-1)]<--parameter.other[feature.index][1:(j-1)]
feature.elastics.other[j,(j+1):nsku]<--parameter.other[feature.index][(j):(nsku-1)]

}

} ### end of if else


}  ### end of j loop




library(corrplot)
corrplot(price.elastics.other, method = "square",is.corr = FALSE,tl.col="black",outline=TRUE,tl.cex=0.8,cl.cex= 0.8)
corrplot(feature.elastics.other, method = "square",is.corr = FALSE,tl.col="black",outline=TRUE,tl.cex=0.8,cl.cex= 0.8)









