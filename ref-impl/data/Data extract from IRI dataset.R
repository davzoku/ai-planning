

setwd("G:/skydrive/SkyDrive/projects/retail promotional causal net analysis/EJOR/code raw/")  ### First need to set the working directory of the codes

######################## Inputs

store.id=236117                             ### First need to input the store ID to be extrated, in this paper, store1: 236117 store2:657979 store3:649405 store4:234212

categorys<-c("milk","beer","mayo","yogurt")  ### The category names to be extracted 

#############################
library(reshape2)
source("./data/transform_functions.R")   ### some self-defined function to extract data from IRI

SKU.data.na<-list()
SKU.data<-list()
Num.SKU<-c()
ncate<-0;
features<-9                           


for (catename in categorys){  

ncate<-ncate+1;

year.adj1<-0;year.adj2<-0

storedata<-c()

for (year in 1:7){                        #### we have 7 years of IRI data

if (year==6) year.adj2<-1;
if (year==7) year.adj1<-1;

data.file<-paste0("G:/Êý¾Ý/IRI/Academic Dataset External/","Year",year,"/External/",catename,                ##### need to set your own path to the directory of IRI data
                           "/",catename,"_groc_",1062+year*52+year.adj1,"_",1113+year*52+year.adj2)          

part1<-read.table(data.file,header=TRUE)                                                                     ##### read data one year by one year
storedata<-rbind(storedata,part1[part1$IRI_KEY==store.id,])                                                  ##### extract data according to store ID

}

part1<-c()

storedata$INDEX<-paste0(as.character(storedata$IRI_KEY),"_",as.character(storedata$SY),"_",as.character(storedata$GE)
                 ,"_",as.character(storedata$VEND),"_",as.character(storedata$ITEM))                         ####  used as unique index for each SKU

storedata$price<-storedata$DOLLARS/storedata$UNITS                                                            ### price
storedata$D1<-ifelse(storedata$D==1,1,0)                                                                      ### minor display
storedata$D2<-ifelse(storedata$D==2,1,0)                                                                      ### major display
storedata$F1<-ifelse((storedata$F=="C") |(storedata$F=="FS-C"),1,0)                                           ### Feature C
storedata$F2<-ifelse((storedata$F=="B") |(storedata$F=="FS-B"),1,0)                                           ### Feature B						
storedata$F3<-ifelse((storedata$F=="A") |(storedata$F=="FS-A"),1,0)                                           ### Feature A
storedata$F4<-ifelse((storedata$F=="A+") |(storedata$F=="FS-A+"),1,0)                                         ### Feature A+


storedata.ext<-data.frame(storedata$INDEX,storedata$WEEK,storedata$UNITS,storedata$PR,storedata$price,        ### reorganize data extracted 
   storedata$D1,storedata$D2,storedata$F1,storedata$F2,storedata$F3,storedata$F4)
names(storedata.ext)<-c("INDEX","WEEK","UNITS","PR","price","D1","D2","F1","F2","F3","F4")                    ### names of the variables


sales<-dcast(storedata.ext,WEEK~INDEX,value.var="UNITS")
names(sales)<-paste0(names(sales),"_UNITS")

PR<-dcast(storedata.ext,WEEK~INDEX,value.var="PR")
names(PR)<-paste0(names(PR),"_PR")

price<-dcast(storedata.ext,WEEK~INDEX,value.var="price")
names(price)<-paste0(names(price),"_price")

D1<-dcast(storedata.ext,WEEK~INDEX,value.var="D1")
names(D1)<-paste0(names(D1),"_D1")


D2<-dcast(storedata.ext,WEEK~INDEX,value.var="D2")
names(D2)<-paste0(names(D2),"_D2")

F1<-dcast(storedata.ext,WEEK~INDEX,value.var="F1")
names(F1)<-paste0(names(F1),"_F1")

F2<-dcast(storedata.ext,WEEK~INDEX,value.var="F2")
names(F2)<-paste0(names(F2),"_F2")

F3<-dcast(storedata.ext,WEEK~INDEX,value.var="F3")
names(F3)<-paste0(names(F3),"_F3")

F4<-dcast(storedata.ext,WEEK~INDEX,value.var="F4")
names(F4)<-paste0(names(F4),"_F4")

selected<-which(colSums(is.na(sales))<=68)               ## SKUs being selected which need have sales obervations more than 300 weeks                                                          
## length(selected)

storedata.t<-as.matrix(cbind(sales[,selected[-1]],PR[,selected[-1]],price[,selected[-1]],D1[,selected[-1]],D2[,selected[-1]],F1[,selected[-1]],
                   F2[,selected[-1]],F3[,selected[-1]],F4[,selected[-1]]) )
rownames(storedata.t)<-sales[,1]

storedata.t<-NA.mean(storedata.t);	  ### manipulating NAs
storedata.t<-Zero.mean(storedata.t)   ### manipulating Zeros
SKU.data[[ncate]]<-storedata.t        
Num.SKU<-c(Num.SKU,ncol(storedata.t)/features)

}


time<-read.csv("./time.csv")               ### month and holidays defined for IRI data
time[is.na(time)]<-0

SKU.data$time=time[match(rownames(storedata.t),time$IRI.Week),-c(1:3)]; 
SKU.data$Num=Num.SKU
names(SKU.data)<-c("milk","beer","mayo","yogurt","time","Num")

save(SKU.data,file=paste0("./data/SKU data store ",store.id,".Rda") )                         #### save the data extracted for the store








