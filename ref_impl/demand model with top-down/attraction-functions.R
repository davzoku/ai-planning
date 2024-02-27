
get.datafull<-function(icate,subcate,calib.time){   ### given the category id, set of SKUs, time period, output the all the data including sales, price, display and features

 Nsku<-SKU.data$Num[icate]
 Y<- as.matrix((SKU.data[[icate]][(calib.time[1]):(calib.time[2]),subcate])) 
 X.pc<-as.matrix(log((SKU.data[[icate]][calib.time[1]:calib.time[2],(Nsku*2+subcate)])))
 X.D1<-((SKU.data[[icate]][calib.time[1]:calib.time[2],(Nsku*3+subcate)]))
 X.D2<-((SKU.data[[icate]][calib.time[1]:calib.time[2],(Nsku*4+subcate)]))
 X.F1<-((SKU.data[[icate]][calib.time[1]:calib.time[2],(Nsku*5+subcate)]))
 X.F2<-((SKU.data[[icate]][calib.time[1]:calib.time[2],(Nsku*6+subcate)]))
 X.F3<-((SKU.data[[icate]][calib.time[1]:calib.time[2],(Nsku*7+subcate)]))
 X.F4<-((SKU.data[[icate]][calib.time[1]:calib.time[2],(Nsku*8+subcate)]))
 X.D<-as.matrix(X.D1+X.D2)
 X.D[X.D>1]=1
 X.F<-as.matrix(X.F1+X.F2+X.F3+X.F4)
 X.F[X.F>1]=1

 list(Y=Y,P=X.pc,D=X.D,F=X.F)

}

####################################

data.split<-function(data,window){                       #### given data, and time window, return the data in the time window
window<-c(window[1]:window[2])
 list(Y=data$Y[window,],P=data$P[window,],D=data$D[window,],F=data$F[window,])

}

#####################################

singe2mix<-function(x){                                ### 
Nsku<-nrow(x);Nt<-ncol(x)
x.mix<-matrix(0,Nsku*Nt,Nsku)
index<-c(1,Nt)
for (i in 1:Nsku){
x.mix[index[1]:index[2],i]<-c(x[i,])
index<-index+Nt
}
x.mix
}


#################################################

get.para<-function(model,Nsku,ref,nx1,nx2){             ### get the estimated parameters from a market share model/attraction model

y0<-rep(0,Nsku)
coefs<-(model)
y0[-ref]<-coefs[1:(Nsku-1)]

beta<-matrix(0,Nsku,nx1)
for( ix in 1:nx1){
beta[-ref,ix]<-coefs[(Nsku*ix):(Nsku*(ix+1)-1)][-c(Nsku)]
beta[ref,ix]<--coefs[(Nsku*ix):(Nsku*(ix+1)-1)][Nsku]
}

alpha<-0
if(nx2>0){
alpha<-coefs[(Nsku*(ix+1)):(Nsku*(ix+1)+nx2-1)]

}

list(y0=y0,beta=beta,alpha=alpha)



}


##########################################

attraction.pre<-function(x1,x2,incept,beta,alpha){     #### generate forecasts with attraction model given model coefficients



Nsku<-nrow(beta)
nx1<-length(x1) ## to be item specific
nx2<-length(x2) ## to be homogenous
Nt<-nrow(x1[[1]])


Y<-matrix(0,Nt,Nsku)
for (i in 1:Nsku){
X1<-c()
for( ix in 1:nx1){
X1<-cbind(X1,x1[[ix]][,i])
}


if (!is.null(x2)){
X2<-c()
for( ix in 1:nx2){
X2<-cbind(X2,x2[[ix]][,i])

}
}else{
alpha<-0
X2<-0
}

if (is.null(x2)){
Y[,i]<-exp(incept[i]+beta[i,]%*%t(X1))
}else{
Y[,i]<-exp(incept[i]+beta[i,]%*%t(X1)+alpha%*%t(X2))
}
        }
Y/rowSums(Y)


}


#############################


attraction.model<-function(y,x1,x2,ref=1){           #### estimate a market share model      

##input: y:sales at t,T*S sku; x: a list including k matrix, each a predictor
##output: a model
###the dimension of y should be the same with xi in x
## at least two skus

Nsku<-ncol(y)-1
Nt<-nrow(y)
nx1<-length(x1) ## to be item specific
nx2<-length(x2) ## to be homogenous

y.cate<-rowSums(y)
y.share<-y/y.cate
y.est<-as.vector(log(y.share[,-c(ref)])-log(y.share[,ref]))


incept<-singe2mix(matrix(1,Nsku,Nt))


x1.beta<-c()

if (nx1==0){
for( ix in 1:nx1){
x1.beta<-cbind(x1.beta,singe2mix(t(x1[[ix]][,-c(ref)])))
}
}

if (nx1>0){
for( ix in 1:nx1){
x1.beta<-cbind(x1.beta,singe2mix(t(x1[[ix]][,-c(ref)])))
x1.beta<-cbind(x1.beta,rep(x1[[ix]][,ref],Nsku))
}
}

x2.beta<-c()
if (nx2>0){
for( ix in 1:nx2){
x2.beta<-cbind(x2.beta,as.vector(x2[[ix]][,-c(ref)]-x2[[ix]][,ref]))
}
}


xx<-cbind(incept,x1.beta,x2.beta)

model.lm<-lm(y.est~xx+0)
coefs<-coef(model.lm)

coefs
} 


##########################################

cate.aggregate<-function(data){            ### SKU data aggregate to category data
y<-log(rowSums(data$Y))
#y.lag<-log(rowSums(data$Y.lag))
share<-colMeans(data$Y)/sum(colMeans(data$Y))
p<-log(colSums(share*t(exp(data$P))))
d<-colSums(share*t(data$D))
f<-colSums(share*t(data$F))

list(Y=y,X=cbind(p,d,f),share=share)
}

##############################################

cate.sales.model<-function(y,own,other=c()){   ### category level sales model
x<-cbind(own,other)
require(glmnet)
model<-glmnet(x,y,lambda=0)

}

###################################

cate.sales.pre<-function(model,own,other=c(),adj=1){    ### generate category level forecasts
x<-as.data.frame(cbind(own,other))

coefs<-as.vector(coef(model))
coefs[is.na(coefs)]<-0
exp(coefs%*% t(cbind(1,x)))*adj
}


#####################################

combine<-function(data,...){
arg<-list(...)
narg<-length(arg)
for (i in 1:narg){
data$X<-cbind(data$X,arg[[i]]$Y,arg[[i]]$X)
}

data

}






