


Zero.mean<-function(sku){                      ### function for manipulating zero for data tranformed

sku.num<-ncol(sku)/features

for (k in 1:sku.num){                         
  zero.index<-sku[,k]==0
  sku[zero.index,k]=mean(sku[,k])             ### for sales, zeros are replaced with mean, this is very rare in the IRI data

}

for (k in (sku.num*2+1):(sku.num*3)){
  zero.index<-sku[,k]==0
  sku[zero.index,k]=mean(sku[,k])             ### for price, zeros are replaced with mean, this is very rare in the IRI data

}

return(sku)

}



NA.mean<-function(sku){                        ### function for manipulating NAs for data tranformed

sku.num<-ncol(sku)/features

for (k in 1:sku.num){
  na.index<-which(is.na(sku[,k]))
  sku[na.index,k]=mean(sku[,k],na.rm=TRUE)      ### for sales, NAs are replaced by mean

}

for (k in (sku.num*2+1):(sku.num*3)){
  na.index<-which(is.na(sku[,k]))
  sku[na.index,k]=mean(sku[,k],na.rm=TRUE)      ### for prices, NAs are replaced by mean


}


for (k in (sku.num*1+1):(sku.num*2)){           ### for pr indicators, NA are replaced with zero
  na.index<-which(is.na(sku[,k]))
  sku[na.index,k]=0

}

for (k in (sku.num*3+1):(ncol(sku))){            ### for display and feature indicators, NA are replaced with zero
  na.index<-which(is.na(sku[,k]))
  sku[na.index,k]=0

}

return(sku)

}





