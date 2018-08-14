library(dplyr)
setwd("E:\\Getting and Cleaning data")
data<-read.csv("first.csv",stringsAsFactors =  FALSE)
tbl_data <- tbl_df(data)
rm(data)
filterData<-filter(tbl_data,!is.na(VAL))
head(filterData)
select(filterData,VAL)
total<-filter(filterData,VAL >= 24)
library(xlsx)
dat<-read.xlsx("getdata%2Fdata%2FDATA.gov_NGAP.xlsx",sheetIndex = 1,header = T,rowIndex = 18:23,colIndex = 7:15)
?read.xlsx
dat
sum(dat$Zip*dat$Ext,na.rm=T)
dat$Zip
dat$X7
library(XML)
url <- "https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2Frestaurants.xml"
download.file(url , destfile = "restaurents.xml")
dat <- xmlTreeParse("rest.xml")
head(dat)
xmlName(dat)
getNode(dat,"//zipcode")
dat[[1]][[1]]
library(data.table)
DT <- fread("getdata%2Fdata%2Fss06pid.csv")
?data.table
head(data)
rowMeans(data)[data$SEX == 1];rowMeans(data)[data$SEX == 2]
rowMeans(DT)[DT$SEX==1]: rowMeans(DT)[DT$SEX==2]
DT[,mean(pwgtp15),by=SEX]
sapply(split(DT$pwgtp15,DT$SEX),mean)
mean(DT$pwgtp15,by=DT$SEX)
mean(DT[DT$SEX==1,]$pwgtp15); mean(DT[DT$SEX==2,]$pwgtp15)
tapply(DT$pwgtp15,DT$SEX,mean)
