
#### profit.fun2 is the fitness function used in GA, x.ga is a gene with length (price.ga+ndf)*h*nc. Each x.ga represents a promotional schedule of the category,
#### for each SKU each week in the category, price, display and feature are set    

profit.fun2<-function(x.ga){       

##### The key here is to decode x.ga to x.value, so that we can evaluate sales and profit

prod<-matrix(0,nc,Npredictors)                           #### temp variable to store value for a sku, prepare for assemble of x.value
profit<-rep(0,lags+h)


x.value<-matrix(0,nc*Npredictors+Nholidays+1,lags+h)                ######   initialize x.value

x.sku<-matrix(x.ga,nc,(price.ga+ndf)*h,byrow=TRUE)                  ######   transform a chain to a matrix, each row is the decision values for a sku
num.d<-rep(0,lags+h);num.f=rep(0,lags+h);                           ######   store the number of promotions in t
num.dt<-rep(0,lags+h);num.ft=rep(0,lags+h);num.prt=rep(0,lags+h)    ######   store the number of promotions across t
num.p<-matrix(0,nc,lags+h); num.pr=num.p;



for (t in (lags+1):(lags+h)){

tl<-t-lags
x.display[,t]<-(x.sku[,(nvariable*tl-ndf+1)])                       ##### decoding display to x.value
x.feature[,t]<-x.sku[,(nvariable*tl-ndf+2)]                         ##### decoding feature to x.value
num.d[t]<-sum(x.display[,t])                                        ##### total number of the displays in the category at week t                   
num.f[t]<-sum(x.feature[,t])                                        ##### total number of the features in the category at week t
num.p[,t]<-(rowSums(x.sku[,(nvariable*(tl-1)+1):(nvariable*(tl-1)+price.ga)])-1)  #### number of prices are selected, when more than one price is selected, this would be larger than one which will be punished
num.pr[,t]<-(rowSums(x.sku[,(nvariable*(tl-1)+1):(nvariable*(tl-1)+price.ga)]))   #### when a price is selected, this means a price reduction
num.p[num.p[,t]<0,t]<-0                                                ############## smaller than zero means no one price is selected, price will be set to the highest in price.list

x.price[,t]<-price.regular/(rowSums(x.sku[,(nvariable*(tl-1)+1):(nvariable*(tl-1)+price.ga)] * (price.list[,1:(nprice-1)]))+   ### decoding price to x.value
                              (price.list[,nprice])*abs(1-rowSums(x.sku[,(nvariable*(tl-1)+1):(nvariable*(tl-1)+price.ga)])) ) ### if all is 0, then use the last price 

prod<-cbind(1,x.price[,t],x.display[,t],x.feature[,t],x.price[,t-1],x.display[,t-1],x.feature[,t-1],sale[,t-1],Y.mov[,t-1])    
x.value[,t]<-c(matrix(t(prod),nrow(prod)*ncol(prod),1,byrow=TRUE),1,x.time[t,])                                                ###  x.value 



sale[,t]=para %*% x.value[,t]                                                ####  estimate sales
Y.mov[,t]<-log(exp(sale[,t])/movs+exp(Y.mov[,t-1])-sale.mov[,t-1]/movs)      ####  update sales for the calculation of next week
sale.mov[,t+movs-1]<-sale[,t]                                                ####  updata the sales moving average
profit[t]<-((price.regular)*p.rate/x.price[,t])%*% exp(sale[,t])-C.d*num.d[t]-C.f*num.f[t]         ### calculate the profit

if (tl>2){
labelchanging.cost<-sum(1-rowSums(x.sku[,(nvariable*(tl-1)+1):(nvariable*(tl-1)+price.ga)] * x.sku[,(nvariable*(tl-2)+1):(nvariable*(tl-2)+price.ga)]))*C.label  ### price lable changing cost
profit[t]<-profit[t]-labelchanging.cost                                      #### update the profit by minus retagging cost
}


                             }


num.d[((num.d-Lim.d)<=0)]<-0                       ### check whether displays are over the constrain     
num.f[((num.f-Lim.f)<=0)]<-0                       ### check whether features are over the constrain
num.pr.s<-colSums(num.pr)
num.pr.s.low<-colSums(num.pr)
num.pr.s[(num.pr.s-Lim.pr)<=0]<-0                  ### check whether price reduction are over the upper bound
num.pr.s.low[(num.pr.s.low-Lim.pr.low)>=0]<-0      ### check whether price reduction are below the lower bound


num.dt<-rowSums(x.display[,(lags+1):(lags+h)])     ### displays over planning horizon
num.ft<-rowSums(x.feature[,(lags+1):(lags+h)])     ### features over planning horizon
num.prt<-rowSums(num.pr)                           ### discount over planning horizon

num.dt[(num.dt-Lim.dr.t)<=0]<-0                    ### check whether displays over planning horizon are over the upper bound
num.ft[(num.ft-Lim.fr.t)<=0]<-0                    ### check whether features over planning horizon are over the upper bound
num.prt[(num.prt-Lim.pr.t)<=0]<-0                  ### check whether discount over planning horizon are over the upper bound


total.profit<-sum(profit)                          ### total profit of the category over planning horizon

panelty<-max(100000,total.profit)                  ### set the value of penalty
panelty.price<- abs(sum(panelty*num.p))            ### penalty for price constrains
panelty.num.d<-abs(sum(num.d[(lags+1):(lags+h)])*panelty)   ### penalty for display constrains
panelty.num.f<-abs(sum(num.f[(lags+1):(lags+h)])*panelty)   ### penalty for feature constrains
panelty.num.pr<-abs((sum(num.pr.s[(lags+1):(lags+h)])+sum(num.pr.s.low[(lags+1):(lags+h)]))*panelty)  ### penalty for discount constrains

panelty.num.dt<-abs(sum(num.dt)*panelty)          ### penalty for display over horizon constrains
panelty.num.ft<-abs(sum(num.ft)*panelty)          ### penalty for feature over horizon constrains
panelty.num.prt<-abs(sum(num.prt)*panelty)        ### penalty for discount over horizon constrains

total.profit-panelty.price-(panelty.num.d+panelty.num.f+panelty.num.pr)-          ### total proft with penalty, only when all the constrains are met, profit can be positive
          (panelty.num.dt+panelty.num.ft+panelty.num.prt)

}

##########################################################


##################initial population#######################
#############################################################

sku.population<-function(GAobj){                   #### this function is used to generate initial population

pop.size<-GAobj@popSize
pop.nBits<-GAobj@nBits
nct<-nc*h
sku.pop<-matrix(0,pop.size,pop.nBits)
for (s in 1:pop.size){

x.sku.pop<-matrix(0,nc,(price.ga+ndf)*h)

d.nt<-rep(0,nc);f.nt<-rep(0,nc);p.nt<-rep(0,nc)

for (pop.t in 1:h){

t.index.start<-(pop.t-1)*6
t.index.end<-(pop.t)*6-1

rnum.pr<-sample(1:nc,ceiling(Lim.pr.low))

p.nt[rnum.pr]<-p.nt[rnum.pr]+1

rnum.d<-rep(0,nc)
rnum.d[sample(1:nc,floor(Lim.d))]<-1
if ( any(d.nt[rnum.d==1]>=(Lim.dr.t-1)) ) rnum.d[which((d.nt>=(Lim.dr.t-1)))]<-0
d.nt[rnum.d==1]<-d.nt[rnum.d==1]+1


rnum.f<-rep(0,nc)
rnum.f[sample(1:nc,floor(Lim.f))]<-1
if (any(f.nt[rnum.f==1]>=(Lim.fr.t-1)) ) rnum.f[ which(f.nt>=(Lim.fr.t-1)) ]<-0
f.nt[rnum.f==1]<-f.nt[rnum.f==1]+1

if (length(rnum.pr)>0 ) {
for (pr.i in rnum.pr){
x.sku.pop[pr.i,(t.index.start+1):(t.index.start+3)]<- t(rmultinom(1, size = 1, prob = c(0.2,0.2,0.2)))
}
                        }
x.sku.pop[,t.index.start+5]<-rnum.d
x.sku.pop[,t.index.start+6]<-rnum.f

}

sku.pop[s,]<-matrix(t(x.sku.pop),1,)

}
sku.pop
}



################this function is for results analysis

profit.fun.analysis<-function(x.ga){

prod<-matrix(0,nc,Npredictors)                     # temp variable to store value for a sku, prepare for assemble of x.value
profit<-matrix(0,nc,lags+h)
x.value<-matrix(0,nc*Npredictors+Nholidays+1,lags+h)  

x.price2<-x.price 

x.sku<-matrix(x.ga,nc,(price.ga+ndf)*h,byrow=TRUE)  #####transform a chain to a matrix, each row is the decision values for a sku
num.d<-c();num.f=c()
num.p<-matrix(0,nc,lags+h)

for (t in (lags+1):(lags+h)){

tl<-t-lags
x.display[,t]<-(x.sku[,(nvariable*tl-ndf+1)])
x.feature[,t]<-x.sku[,(nvariable*tl-ndf+2)]
num.d[t]<-sum(x.display[,t])
num.f[t]<-sum(x.feature[,t])
num.p[,t]<-(rowSums(x.sku[,(nvariable*(tl-1)+1):(nvariable*(tl-1)+price.ga)])-1)
num.p[num.p[,t]<0,t]<-0

for (i in 1:nc){
x.price[i,t]<-price.regular[i]/(x.sku[i,(nvariable*(tl-1)+1):(nvariable*(tl-1)+price.ga)] %*% (price.list[i,1:(nprice-1)])+
                              (price.list[i,nprice])*abs(1-sum(x.sku[i,(nvariable*(tl-1)+1):(nvariable*(tl-1)+price.ga)])) ) ### if all is 0, then use the last price 
x.price2[i,t]<-x.sku[i,(nvariable*(tl-1)+1):(nvariable*(tl-1)+price.ga)] %*% c(1,2,3,4)+
                              (5*abs(1-sum(x.sku[i,(nvariable*(tl-1)+1):(nvariable*(tl-1)+price.ga)])) ) ### if all is 0, then use the last price 
prod[i,]<-c(1,x.price[i,t],x.display[i,t],x.feature[i,t],x.price[i,t-1],x.display[i,t-1],x.feature[i,t-1],sale[i,t-1],Y.mov[i,t-1])
}

x.value[,t]<-c(matrix(t(prod),nrow(prod)*ncol(prod),1,byrow=TRUE),1,x.time[t,])
sale[,t]=para %*% x.value[,t]   ##x.value should be a vector including all the value needed to calculate sale, para is a matrix corresponding to x.value
Y.mov[,t]<-log(exp(sale[,t])/movs+exp(Y.mov[,t-1])-sale.mov[,t-1]/movs)
sale.mov[,t+movs-1]<-sale[,t]

profit[,t]<-((price.regular)*p.rate/x.price[,t])* exp(sale[,t])-C.d*x.display[,t]-C.f*x.feature[,t]

if (tl>2){
labelchanging.cost<-(1-rowSums(x.sku[,(nvariable*(tl-1)+1):(nvariable*(tl-1)+price.ga)] * x.sku[,(nvariable*(tl-2)+1):(nvariable*(tl-2)+price.ga)]))*C.label
profit[,t]<-profit[,t]-labelchanging.cost
}

             }

return(list(profit=profit,display=x.display,feature=x.feature,price=x.price2,price.v=price.regular/x.price,sale=exp(sale)))

}


###################









