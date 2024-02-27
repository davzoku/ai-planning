



################# The Main programe for promotional optimization by GA.GA is a time consuming algorithm, on average the promotional planning for one SKU 
#################  over 8 weeks horizon needs half a minute on a i7 4 cores computer, so in our experiments we have 912 SKU in 4 stores, for one experiment
################# this needs about 4~5 hours in total. As we need conduct about 20 experiments in this project, all the experiments are conducted on Amazon EC2.
################# When one only want to test the program, we would suggest to select one store one category to run so that to save the time.

library(GA)
library(TTR)
library(glmnet)


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


#######  
start<-11               ### the time window of the data used in the experiments,IRI has 365 weeks (7 years)of the data, 
                        ### so we use week 11 to 365 in our experments, weeks before 11 used by some lags variables,e.g. sales moving average.
valid.time<-251         ### start of the week for validation
h=8                     ### the week cycle for planning optimization
rollings=10             ### rolling numbers in the experiments
t.index<-1              ### the period for opt, as we define 10 rolling periods in the experiments, we need to indicate which period want to be the planning period for optimization

ndf<-2;                        #### number bits for coding the display and feature
lags<-1;                       #### number of the lags used in the model
h<-8;                          #### the length of the horizen for optimization
holidays<-c(9:25)              #### the position of the holiday indicators in the demand model
movs<-8                        #### for calculating the moving average sales
Npredictors<-9               ###### number of own predictors used in the demand models:(intercept,price[t],display[t],feature[t],price[t-1],display[t-1],feature[t-1],sale[t-1],mov[t-1])
p.rate<-0.20                 ##### profit rate
C.d<-20;C.f<-20;C.label<-1   ##### cost of display , feature,retagging

### In practice, retailer also need to set the values for constrains,including
### Lim.pr.t   :the maximum number of weeks a SKU is allowed to have a price reduction
### Lim.pr     :upper bounds on the number of SKUs in the category allowed to have a price reduction in a week
### Lim.pr.low :lower bounds on the number of SKUs in the category allowed to have a price reduction in a week
### Lim.dr.t   :the maximum number of weeks a SKU is allowed to have a display
### Lim.d      :upper bounds on the number of SKUs in the category allowed to have a display in a week
### Lim.fr.t   :the maximum number of weeks a SKU is allowed to have a feature advertising
### Lim.f      :upper bounds on the number of SKUs in the category allowed to have a feature in a week
####### In our simulative optimization experiments, we obtain above constrains from history data, so we didn't set them arbitrarily



###########################################

timetest=seq(1,h*rollings,by=h)      ### steps for rolling loops
Nholidays<-length(holidays)          ### number of holiday indicators in the demand model


for (i.store in c(1:4)){             ###### Main loop, optimize promotinal planning store by store


load(store.name[i.store])                        ### loading the store data, which stored in a list named "SKU.data"             
load(file=store.para.name[i.store])              ### loading the coefficients from demand model, which stored in a list named "results.para"
calib.time<-c(start,length(SKU.data[[1]][,1]))   ### the start week and the lengh of total time 

tt=timetest[t.index]


opt.results<-c()                    ### for storing the optimization results

for (cate in 1:4){                  ### start the category loop, optimization category by category

para10.own<-list()                  ### for reading the coefficients of the own predictors in the demand model
para10.other<-list()                ### for reading the coefficients of the cross predictors in the demand model
nc<-SKU.data$Num[cate]              ########## number of skus in the category


for (j in 1:nc){

para10.own[[j]]<-results.para[[cate]][[1]][[j]][[t.index]]   #### reading the coefficients of the own   predictors in the demand model
para10.other[[j]]<-results.para[[cate]][[2]][[j]][[t.index]] #### reading the coefficients of the cross predictors in the demand model

}


## the key for GA is to build a fitness function, given a solution could evaluate the objective value
## we need a matrix of cost for every SKU C
## price list for every SKU, price.x is better to transform to a vector, equal number of prices for each SKU will be easier
## x is a vector of solution: sku1 solution time 1| solution time 2 | solution time3 
## for one SKU at time t is x1,x2...xk(k price);xk+1(display or feature); 

############para prepare, transform the estimated parameters to opt needed form
### para has nc row and should be same as x.value in each row
####prod[i]=c(intercept,price[t],display[t],feature[t],price[t-1],display[t-1],feature[t-1],sale[t-1],mov[t-1])


#### para=c(intercept,price[t],display[t],feature[t],price[t-1],display[t-1],feature[t-1],sale[t-1],mov[t-1])
#### last other.intercept, holidays


time.p<-calib.time[1]-1+valid.time+tt   ##### begin of the time of opt
price.list<-get.price.list(cate)        ##### get history price 5 levels represent 5 quantile of price in history,as the price options
nprice<-ncol(price.list);               ##### number of price options
price.ga<-ncol(price.list)-1            ##### num of dicision variable for price of one sku  
nvariable<-nprice+ndf-1                 ##### number of decision variables for one sku

num.var<-(price.ga+ndf)*h*nc            ##### total decision variables for one gene, the length of a gene


########################## Following is to prepare parameters matrix

para<-matrix(0,nc,nc*Npredictors+Nholidays+1)  #### the parameters matrix, each row represents a SKU, including both own predictors and cross predictors
                                               ### the sequence is (intercept,price[t],display[t],feature[t],price[t-1],display[t-1],feature[t-1],sale[t-1],mov[t-1])
                                               ### of  each SKU in the category, and then intercept from cross model (second stage in demand model), and at last holidays
                                               ### the coefficients which are not used in cross model are set to zero.
for (i in 1:nc){                        #### prepare coefficients matrix, one row one SKU, para=c(SKU[1],SKU[2],....,intercept.cross,holidays)
                                        #### where SKU[i]=c(intercept,price[t],display[t],feature[t],price[t-1],display[t-1],feature[t-1],sale[t-1],mov[t-1])

para[i,((i-1)*Npredictors+1):((i-1)*Npredictors+4)]<-para10.own[[i]][1:4]
para[i,((i-1)*Npredictors+5):(i*Npredictors-1)]<- -para10.own[[i]][5:8]
para[i,(i*Npredictors)]<- para10.own[[i]][26]

para[i,nc*Npredictors+1]<-para10.other[[i]][1]
para[i,(nc*Npredictors+2):(nc*Npredictors+Nholidays+1)]<-para10.own[[i]][holidays]

for (j in 1:(nc-1)){
if (j<i){
para[i,((j-1)*Npredictors+2):((j-1)*Npredictors+4)]<-(-para10.other[[i]][((j-1)*4+2):((j-1)*4+4)])
para[i,((j-1)*Npredictors+Npredictors-1)]<- -para10.other[[i]][((j-1)*4+5)]  ## sales lags
         }
if (j>=i){
para[i,((j)*Npredictors+2):(j*Npredictors+4)]<-(-para10.other[[i]][((j-1)*4+2):((j-1)*4+4)])
para[i,((j)*Npredictors+Npredictors-1)]<- -para10.other[[i]][((j-1)*4+5)]
          }  
                   }                     #### end of j loop


              }  #### end of i,sku loop

############################# End of the preparing parameters matrix




###########Following is to prepare the x.value conrespondence to parameters,that is log(Y)= para*x.value to estimate sales for each x.value solution.
###########x.value is used to calculate the sales, x.value multiplied by parameters matrix to obtain the estimated sales for each SKU,so it should be exactly the same sequence conrespondence to parameters matrix
### x.value=c(SKU[1],SKU[2],....,intercept.cross,holidays), where SKU[i]=c(intercept,price[t],display[t],feature[t],price[t-1],display[t-1],feature[t-1],sale[t-1],mov[t-1])
#### the value a gene represents,


sale<-matrix(0,nc,lags+h)      ### for calculating sales
sale.mov<-matrix(0,nc,movs+h)  ### for calcualting the moving average
Y.mov<-matrix(0,nc,lags+h)     ### store the sales moving average
x.price<-matrix(0,nc,lags+h)   
x.display<-matrix(0,nc,lags+h)
x.feature<-matrix(0,nc,lags+h)
price.regular<-rep(0,nc)

######################### known values, needed at the beginning of the horizen planning because there are lags in the demand model

x.display[,1:lags]<-t((SKU.data[[cate]][(time.p-lags):(time.p-1),(nc*3+1):(nc*4)])+
                    (SKU.data[[cate]][(time.p-lags):(time.p-1),(nc*4+1):(nc*5)]))
x.feature[,1:lags]<-t((SKU.data[[cate]][(time.p-lags):(time.p-1),(nc*5+1):(nc*6)])+
                    (SKU.data[[cate]][(time.p-lags):(time.p-1),(nc*6+1):(nc*7)])+
                    (SKU.data[[cate]][(time.p-lags):(time.p-1),(nc*7+1):(nc*8)])+
                    (SKU.data[[cate]][(time.p-lags):(time.p-1),(nc*8+1):(nc*9)]))

x.time<-as.matrix(SKU.data$time)[(time.p-lags):(time.p+h-1),13:29]
sale[,1:lags]<-t(log(SKU.data[[cate]][(time.p-lags):(time.p-1),(1):(nc*1)]))
sale.mov[,1:movs]<-t((SKU.data[[cate]][(time.p-movs):(time.p-1),(1):(nc*1)]))
for (j in 1:nc){
Y.mov[j,1:lags]<-log(SMA(SKU.data[[cate]][,j],movs)[(time.p-lags):(time.p-1)])
X.pc<-(SKU.data[[cate]][,j+nc*2])
price.regular[j]<-get.price.regular(X.pc,calib.time=c(1,length(SKU.data[[1]][,1])))[(time.p-1)]
               }
x.price[,1:lags]<-t(price.regular/(SKU.data[[cate]][(time.p-lags):(time.p-1),(nc*2+1):(nc*3)]))

##################################################################################################3

################################################################################

#### setting the constrains on promotions which are summarized from historical data, retailer could of course set their own values, here is only for a relatively fair comparison between opt and real profit
### Lim.pr.t   :the maximum number of weeks a SKU is allowed to have a price reduction
### Lim.pr     :upper bounds on the number of SKUs in the category allowed to have a price reduction in a week
### Lim.pr.low :lower bounds on the number of SKUs in the category allowed to have a price reduction in a week
### Lim.dr.t   :the maximum number of weeks a SKU is allowed to have a display
### Lim.d      :upper bounds on the number of SKUs in the category allowed to have a display in a week
### Lim.fr.t   :the maximum number of weeks a SKU is allowed to have a feature advertising
### Lim.f      :upper bounds on the number of SKUs in the category allowed to have a feature in a week
####### In our simulative optimization experiments, we obtain above constrains from history data, so we didn't set them arbitrarily

h.discount<-SKU.data[[cate]][,(nc+1):(nc*2)]
h.display<-SKU.data[[cate]][,(nc*3+1):(nc*4)]+SKU.data[[cate]][,(nc*4+1):(nc*5)]
h.feature<-SKU.data[[cate]][ ,(nc*5+1):(nc*6)]+SKU.data[[cate]][ ,(nc*6+1):(nc*7)]+SKU.data[[cate]][ ,(nc*7+1):(nc*8)]+SKU.data[[cate]][ ,(nc*8+1):(nc*9)]

Lim.pr.t<-h*round(max(colSums(h.discount>0))/nrow(h.discount)*100)/100
Lim.pr<-nc*round(max(rowSums(h.discount>0))/ncol(h.discount)*100)/100
Lim.pr.low<-nc*round(mean(h.discount>0)*100)/100

Lim.dr.t<-h*round(max(colSums(h.display>0))/nrow(h.display)*100)/100
Lim.d<-nc*round(max(rowSums(h.display>0))/ncol(h.display)*100)/100

Lim.fr.t<-h*round(max(colSums(h.feature>0))/nrow(h.feature)*100)/100
Lim.f<-nc*round(max(rowSums(h.feature>0))/ncol(h.feature)*100)/100

###########################################################################


GA <- ga(type = "binary", fitness = profit.fun2, nBits = num.var,pmutation = 0.25,           ##### GA optimization, profit.fun2 is the fitness function
 maxiter = 200, run = 100,population=sku.population, popSize = 10,parallel = TRUE)           ##### sku.population is a function to generate initial poplulation

x.ga<-GA@solution[nrow(GA@solution),]                                                        ##### obtain the best solution


#######results analysis##############################################

##########    calculate real profits for comparision
sale.real<-t(log(SKU.data[[cate]][(time.p):(time.p+h-1),(1):(nc*1)]))
price.real<-t(1/(SKU.data[[cate]][(time.p):(time.p+h-1),(nc*2+1):(nc*3)]))
display.real<-t((SKU.data[[cate]][(time.p):(time.p+h-1),(nc*3+1):(nc*4)])+
                    (SKU.data[[cate]][(time.p):(time.p+h-1),(nc*4+1):(nc*5)]))
feature.real<-t((SKU.data[[cate]][(time.p):(time.p+h-1),(nc*5+1):(nc*6)])+
                    (SKU.data[[cate]][(time.p):(time.p+h-1),(nc*6+1):(nc*7)])+
                    (SKU.data[[cate]][(time.p):(time.p+h-1),(nc*7+1):(nc*8)])+
                    (SKU.data[[cate]][(time.p):(time.p+h-1),(nc*8+1):(nc*9)]))


profit.real<-matrix(0,nc,h)

	for (t in 1:h){  ### calculate the real profit week by week
	profit.real[,t]<-(p.rate/price.real[,t])* exp(sale.real[,t])-C.d*(display.real[,t])-C.f*(feature.real[,t])   ###10percent as the profit

	if (t >2){      #### calculate the retagging cost
	labelchanging.cost<-(abs(price.real[,t]-price.real[,t-1])>0)*C.label
	profit.real[,t]<-profit.real[,t]-labelchanging.cost
		   }


	             }  ## end of the t loop

real.summary<-list(exp(sale.real),1/price.real,feature.real,profit.real)   ### real summary

################# calculate opt profits for comparision

results.opt<-profit.fun.analysis(x.ga)

####################################################

##### following is for generating the results in Table 6 & Table 7, Fig. 4 & Fig. 5 reported in the paper

result.comp<-data.frame( rowMeans(results.opt$profit[,-c(1:lags)]), rowMeans(real.summary[[4]]), rowMeans(results.opt$price.v[,-c(1:lags)]),
				 rowMeans(real.summary[[2]]),  rowMeans(results.opt$feature[,-c(1:lags)])+rowMeans(results.opt$display[,-c(1:lags)]),
				 rowMeans(real.summary[[3]]),  rowSums(results.opt$sale[,-c(1:lags)])/sum(results.opt$sale[,-c(1:lags)]),
                         rowSums(real.summary[[1]])/sum(real.summary[[1]]), rowMeans(results.opt$sale[,-c(1:lags)]),rowMeans(real.summary[[1]])  )
names(result.comp)<-c("opt profit","real profit","opt price","real price","opt feature","real feature","opt share","real share","opt sales","real sales")


opt.results[[cate]]<-list(GA,result.comp)  ### store the results

}          #############  end of the category loop

save(opt.results,file=store.result.name[i.store])    #### save the opt results for current store

}          ########## end of the store loop














