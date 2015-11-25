
#Model1
#Model1
#Model1

import graphlab as gl
cd C:\\Users\\Nalin\\Documents\\R\\Black Friday
train=gl.SFrame('train.csv')
test=gl.SFrame('test.csv')
train.column_names()
test.column_names()
train['Purchase']=train['Purchase'].astype(float)
trainBasic=gl.SFrame({'user_id':train['User_ID'],'item_id':train['Product_ID'],'Purchase':train['Purchase']})
trainUser=gl.SFrame({'user_id':train['User_ID'],'Gender':train['Gender'],'Age':train['Age'],'Occupation':train['Occupation'],'City_Category':train['City_Category'],'Stay_In_Current_City_Years':train['Stay_In_Current_City_Years'],'Marital_Status':train[ 'Marital_Status']})
trainProduct=gl.SFrame({'item_id':train['Product_ID'],'Product_Category_1':train['Product_Category_1']})
model1= gl.factorization_recommender.create(trainBasic, target='Purchase',
                                                user_data=trainUser,
                                               item_data=trainProduct,num_factors=70,side_data_factorization=True,random_seed=50)
testBasic=gl.SFrame({'user_id':test['User_ID'],'item_id':test['Product_ID']})
testUser=gl.SFrame({'user_id':test['User_ID'],'Gender':test['Gender'],'Age':test['Age'],'Occupation':test['Occupation'],'City_Category':test['City_Category'],'Stay_In_Current_City_Years':test['Stay_In_Current_City_Years'],'Marital_Status':test[ 'Marital_Status']})
testProduct=gl.SFrame({'item_id':test['Product_ID'],'Product_Category_1':test['Product_Category_1']})
predictions=model1.predict(testBasic, new_user_data=testUser,new_item_data=testProduct)
Purchase=gl.SArray(predictions)
User_ID=gl.SArray(test['User_ID'])
Product_ID=gl.SArray(test['Product_ID'])
Submission=gl.SFrame({'User_ID':User_ID,'Product_ID':Product_ID,'Purchase':Purchase})
Submission.save('Sub13',format='csv')

#Model2
#Model2
#Model2

import graphlab as gl
cd C:\\Users\\Nalin\\Documents\\R\\Black Friday
train=gl.SFrame('train.csv')
test=gl.SFrame('test.csv')
train.column_names()
test.column_names()
train['Purchase']=train['Purchase'].astype(float)
trainBasic=gl.SFrame({'user_id':train['User_ID'],'item_id':train['Product_ID'],'Purchase':train['Purchase']})
trainUser=gl.SFrame({'user_id':train['User_ID'],'Gender':train['Gender'],'Age':train['Age'],'Occupation':train['Occupation'],'City_Category':train['City_Category'],'Stay_In_Current_City_Years':train['Stay_In_Current_City_Years'],'Marital_Status':train[ 'Marital_Status']})
trainProduct=gl.SFrame({'item_id':train['Product_ID'],'Product_Category_1':train['Product_Category_1']})
model1= gl.factorization_recommender.create(trainBasic, target='Purchase',
                                                user_data=trainUser,
                                               item_data=trainProduct,num_factors=65,side_data_factorization=True,random_seed=50)
testBasic=gl.SFrame({'user_id':test['User_ID'],'item_id':test['Product_ID']})
testUser=gl.SFrame({'user_id':test['User_ID'],'Gender':test['Gender'],'Age':test['Age'],'Occupation':test['Occupation'],'City_Category':test['City_Category'],'Stay_In_Current_City_Years':test['Stay_In_Current_City_Years'],'Marital_Status':test[ 'Marital_Status']})
testProduct=gl.SFrame({'item_id':test['Product_ID'],'Product_Category_1':test['Product_Category_1']})
predictions=model1.predict(testBasic, new_user_data=testUser,new_item_data=testProduct)
Purchase=gl.SArray(predictions)
User_ID=gl.SArray(test['User_ID'])
Product_ID=gl.SArray(test['Product_ID'])
Submission=gl.SFrame({'User_ID':User_ID,'Product_ID':Product_ID,'Purchase':Purchase})
Submission.save('Sub21',format='csv')

#Model3
#Model3
#Model3

import graphlab as gl
cd C:\\Users\\Nalin\\Documents\\R\\Black Friday
train=gl.SFrame('train.csv')
test=gl.SFrame('test.csv')
train.column_names()
test.column_names()
train['Purchase']=train['Purchase'].astype(float)
trainBasic=gl.SFrame({'user_id':train['User_ID'],'item_id':train['Product_ID'],'Purchase':train['Purchase']})
trainUser=gl.SFrame({'user_id':train['User_ID'],'Gender':train['Gender'],'Age':train['Age'],'Occupation':train['Occupation'],'City_Category':train['City_Category'],'Stay_In_Current_City_Years':train['Stay_In_Current_City_Years'],'Marital_Status':train[ 'Marital_Status']})
trainProduct=gl.SFrame({'item_id':train['Product_ID'],'Product_Category_1':train['Product_Category_1']})
model1= gl.factorization_recommender.create(trainBasic, target='Purchase',
                                                user_data=trainUser,
                                               item_data=trainProduct,num_factors=65,max_iterations=60,side_data_factorization=True,random_seed=50)
testBasic=gl.SFrame({'user_id':test['User_ID'],'item_id':test['Product_ID']})
testUser=gl.SFrame({'user_id':test['User_ID'],'Gender':test['Gender'],'Age':test['Age'],'Occupation':test['Occupation'],'City_Category':test['City_Category'],'Stay_In_Current_City_Years':test['Stay_In_Current_City_Years'],'Marital_Status':test[ 'Marital_Status']})
testProduct=gl.SFrame({'item_id':test['Product_ID'],'Product_Category_1':test['Product_Category_1']})
predictions=model1.predict(testBasic, new_user_data=testUser,new_item_data=testProduct)
Purchase=gl.SArray(predictions)
User_ID=gl.SArray(test['User_ID'])
Product_ID=gl.SArray(test['Product_ID'])
Submission=gl.SFrame({'User_ID':User_ID,'Product_ID':Product_ID,'Purchase':Purchase})
Submission.save('Sub29',format='csv')

#Finally, ensemble all three prediction file with same weightage 
