import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize ,scale

#set param
iteration = 2

def offset_func(x,O,features_weight):
    row,clm= x.shape
    offset = np.zeros(row)
    for sample in range (row):
        tmp = np.subtract(O , x[sample])
        tmp = np.multiply(tmp,features_weight)
        offset[sample] = np.sum(np.power(tmp,2))
    return offset

def find_threshold(GS_t):
    time = 0
    th = list()
    for i in range(len(GS_t)):
        if (GS_t[i][2] - time) > 0.01 :
            tmp = (time + GS_t[i][2])/2
            th.append(tmp)
            time = GS_t[i][2]
    return th

def CPE_computation(threshold,GS_t,y,row):
    CPE = list()
    classifier_at_0 = np.array([x[0:2] for x in GS_t[0:row]],dtype=np.int32).reshape(-1,2)
    cost_at_0 = np.zeros(row)
    tmp = 0
    for index in range(row):
        index_query = classifier_at_0[index][0]
        index_nearest = classifier_at_0[index][1]
        if y[index_query] != y [index_nearest]:
            cost_at_0[index]=1 
            tmp += 1
    CPE.append((0,tmp))
    #print(CPE)
    th_lower= 0
    counter = 0
    for th in threshold:
        while GS_t[counter][2] <th :
            if GS_t[counter][2] > th_lower:
                index_query = GS_t[counter][0]
                index_nearest = int(GS_t[counter][1])
                if y[index_query] == y[index_nearest] and cost_at_0[index_query] == 1: 
                    tmp -= 1
                    cost_at_0[index_query] = 0
                if y[index_query] != y[index_nearest] and cost_at_0[index_query] == 0:
                    tmp += 1
            counter += 1
        th_lower = th
        #print(tmp)
        CPE.append((th , tmp))
    return CPE

def critical_time_computation(runners):
    L = list()
    runners.pop(0)
    L.append(runners[0])
    L_counter = 0
    #removeing first runners
    for runner_index , runner_offset , runner_velocity in runners:
        if runner_velocity < L[L_counter][2]:
            L_counter += 1
            L.append((runner_index,runner_offset,runner_velocity))
    #first obj in S's tuple is tuple that include:
    #runner's index , offset and velocity
    #second obj detrmined the crtitical_time(weights)
    S =list() #return list 
    S.append((runners[0], 0))
    S_counter = 0
    L.pop(0) #remove L_first from L
    #removeing lucky runners
    for runner_index , runner_offset , runner_velocity in L:
        #compute critical_times(weight) for each runner
        critical_time  = runner_offset - S[S_counter][0][1]
        critical_time /= (S[S_counter][0][2] - runner_velocity)
        while S_counter > 0 and critical_time <= S[S_counter][1]:
            S.pop(S_counter)
            S_counter -= 1
            critical_time  = runner_offset - S[S_counter][0][1]
            critical_time /= (S[S_counter][0][2] - runner_velocity)
        S.append(((runner_index,runner_offset,runner_velocity),critical_time))
        S_counter += 1
    return S

def feature_weighting(x,y,features_weight):
    flag = 0
    for itr in range (iteration):
        #final_weights = list()
        row,clm= x.shape
        for feature in range(clm):
            if flag == 0:
                features_weight[feature] = 0
                GS_t = list()
                instance_number = 0
                for instance in x:
                    offset = np.zeros(row)
                    offset = offset_func(x,instance,features_weight)
                    offset = offset.reshape(-1,1)
                    L = np.hstack((np.arange(row).reshape(-1,1), offset))
                    L = L[np.argsort(L[:,1])] #sort instances by sorthing their distance from X(O)
                    list_ =list()
                    for index , distance in L:
                        velocity = np.subtract(instance[feature],x[int(index)][feature])
                        velocity = np.power(velocity,2)
                        list_.append((index,distance,velocity))
                    S_t = critical_time_computation(list_)
                    for runner , time in S_t:
                        GS_t.append((instance_number, runner[0] ,time))
                    instance_number += 1
                #sorted GS_t by critical_time
                GS_t = sorted(GS_t, key = lambda x: x[2])
                #check 
                """cont = 1
                for obj in GS_t:
                    print(cont , obj)
                    cont += 1 """
                threshold = find_threshold(GS_t)
                #print(len(threshold))
                CPE = CPE_computation(threshold,GS_t,y,row)
                #first element of CPE is th_time 
                #and second element is CPE for this threshold
                #sort in-place from highest to lowest
                CPE_sorted_by_CPE= sorted (CPE,key = lambda x: x[1])
                #count= 0
                cost = CPE_sorted_by_CPE[0][1] 
                if cost < 2:
                    flag = 1
                """while CPE_sorted_by_CPE[count][1] == cost:
                    count+=1"""
                print(CPE_sorted_by_CPE[0])
                #final_weights.append(CPE_sorted_by_CPE[0][0])
                features_weight[feature] = CPE_sorted_by_CPE[0][0]
        print(features_weight)
        #features_weight = final_weights
    return features_weight

# Data_Set-------------------------------------------
Banknote = pd.read_csv ('C://Users//Parisan.Sh//Desktop//pattern//banknote.dataset.csv', encoding='ansi')
#print(Banknote)
y = Banknote.y
y = y.values
y = y.reshape(-1,1)
#print(y)
Banknote = Banknote.drop('y', axis = 1)
Banknote_Normalize = scale(Banknote)
x = Banknote_Normalize
x_train , x_test , y_train , y_test = train_test_split (x , y ,test_size = .2 , random_state = 42)
#----------------------------------------------------
knn = KNeighborsClassifier(n_neighbors= 5 , metric= 'minkowski',p=2)
row , clm = x.shape

#knn.fit(x_train,y_train)
#y_predict1 = knn.predict(x_test)
#print(confusion_matrix(y_test,y_predict1))  
#print(classification_report(y_test,y_predict1))

features_weight = np.ones(clm)
features_weight = feature_weighting(x_train , y_train ,features_weight)
x_train = np.multiply(x_train , features_weight)
x_test = np.multiply(x_test,features_weight)
knn.fit(x_train,y_train)
y_predict2 = knn.predict(x_test)
print(confusion_matrix(y_test,y_predict2))  
print(classification_report(y_test,y_predict2))