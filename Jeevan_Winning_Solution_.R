train = read.csv("train_FBFog7d.csv")
test = read.csv("Test_L4P23N3.csv")
alch_csv = read.csv("NewVariable_Alcohol.csv") 

test = merge(test, alch_csv, "ID")
train = merge(train,alch_csv,"ID")

train = train[,c(-1)]
ID = test$ID
test = test[,c(-1)]

library(gbm)

treeLength = 750
soln = matrix(data = 0,nrow=3387,ncol = 10)
colNo = 1

while(treeLength<1200){
  
  gbmModel = gbm(Happy ~ .,data=train,distribution="multinomial",n.trees=treeLength,interaction.depth=3)
  
  PredictGbm = predict(gbmModel,n.trees=treeLength, newdata=test,type='response')
  p.predBST <- apply(PredictGbm, 1, which.max)
  
  soln[,colNo] = p.predBST
  
  treeLength = treeLength+50
  colNo = colNo+1
  
}

for(i in 1:3387){
  soln1[i,10] = names(which.max(table(soln1[i,1:9])))
}
ans = soln1[,10]
  
ans[ans=="1"]="Not Happy"
ans[ans=="2"]="Pretty Happy"
ans[ans=="3"]="Very Happy"

submission = data.frame(ID = ID,Happy = ans) 
write.csv(submission,"submissionFinal.csv",row.names = FALSE)
