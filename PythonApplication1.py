import matplotlib.pyplot as plt
from minisom import MiniSom
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from numpy import zeros #genfromtxt,array,linalg,zeros,apply_along_axis
from pylab import plot,axis,show,pcolor,colorbar,bone

dataAge = np.genfromtxt('CouncilNonNorm6.csv', delimiter=',', skip_header=1, usecols=(2, 3, 4, 5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20))
dataQualifications = np.genfromtxt('CouncilNonNorm6.csv', delimiter=',', skip_header=1, usecols=(32, 33, 34, 35, 36, 37, 38))

sc = MinMaxScaler(feature_range = (0,1))
sc2 = MinMaxScaler(feature_range = (0,1))

sc2.fit(dataAge)
dataAgeNorm = sc2.transform(dataAge)

sc.fit(dataQualifications)
dataQualificationsNorm = sc.transform(dataQualifications)

labels = np.genfromtxt('CouncilNonNorm6.csv', delimiter=',', usecols=(46), skip_header=1, dtype=str)

dataQualificationsNorm[:-1]
labels[:-1]
labels[20] = 'NOC'


som = MiniSom(15, 15, dataQualifications.shape[1], sigma=.9, learning_rate=.2,
              neighborhood_function='gaussian', random_seed=0)


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

X_train, X_test, y_train, y_test = train_test_split(dataQualificationsNorm, labels, stratify=labels)

X_train2, X_test2, y_train2, y_test2 = train_test_split(dataAgeNorm, labels, stratify=labels)


#som.train_random(X_train, 5000, verbose=True) #50000


#som.train_random(X_train, 5000, verbose=True)


"""
with open('som.p', 'wb') as outfile:  #save
    pickle.dump(som, outfile)
"""
"""

#Will save a file called som.p
with open('som.p', 'wb') as outfile:
    pickle.dump(som, outfile)
# Will load a file called som.p and "som" variable
#can be called and used to retrived the train SOM.
with open('som.p', 'rb') as infile:
    som = pickle.load(infile)
    

with open('somR.p', 'rb') as infile: #load Qualf
    som2 = pickle.load(infile)
"""
with open('som.p', 'rb') as infile:
    som = pickle.load(infile)
    

with open('somR.p', 'rb') as infile: #load Qualf
    som2 = pickle.load(infile)



def FindNeuronWinner(county):
    mappings = som.win_map(dataQualificationsNorm)

    for x in range(0, 44):
        for y in range (0, 44):
            for winningValues in mappings[x,y]:
                #winningValues  = winningValues.reshape(1, -1) #reverting min scaler fit function
                #winningValues  = sc.inverse_transform(winningValues) #reverting min scaler transform from 0,1 - data normalisation
                #winningValues  = winningValues.flatten() #turning 2d array into 1d array
                 #winningValues = np.concatenate((winningValues, winLoc))
                #currentLoc = np.around(winningValues , 0).astype(int) #saving the values to an array and rounding the values to whole numbers.
                #county #31 no qual
                winningValues = np.array(winningValues)
                county = np.array(county)
                
                if np.array_equal(winningValues, county) == True:
                    ans = [x , y]
                    #print(winningValues)
                    #print(county)
                    #print(x)
                    #print(y)
                    return ans
                """
                if (winningValues == county).all():
                    ans = [x , y]
                    return ans
                """


def FindNeuronWinnerA(county):
    mappings = som2.win_map(dataAgeNorm)

    for x in range(0, 70):
        for y in range (0, 70):
            for winningValues in mappings[x,y]:
                #winningValues  = winningValues.reshape(1, -1) #reverting min scaler fit function
                #winningValues  = sc.inverse_transform(winningValues) #reverting min scaler transform from 0,1 - data normalisation
                #winningValues  = winningValues.flatten() #turning 2d array into 1d array
                 #winningValues = np.concatenate((winningValues, winLoc))
                #currentLoc = np.around(winningValues , 0).astype(int) #saving the values to an array and rounding the values to whole numbers.
                #county #31 no qual
                winningValues = np.array(winningValues)
                county = np.array(county)
                
                if np.array_equal(winningValues, county) == True:
                    ans = [x , y]
                    #print(winningValues)
                    #print(county)
                    #print(x)
                    #print(y)
                    return ans
                """
                if (winningValues == county).all():
                    ans = [x , y]
                    return ans
                """


def GetNeuronWinner(x, y):
    mappings = som.win_map(dataQualificationsNorm)
    results = []
    winningData = pd.DataFrame()



    #columns=['council_name','age4', 'age9', 'age14','age19', 'age24', 'age29', 'age34', 'age39', 'age44', 'age49', 'age54', 'age59','age64', 'age69', 'age74', 'age79', 'age84', 'age90', 'con', 'lab', 'libdem', 'green', 'other', 'vacant', 'majority']
    columns=['council_name', 'all_person', 'age4', 'age9', 'age14','age19', 'age24', 'age29', 'age34', 'age39', 'age44', 'age49', 'age54', 'age59','age64', 'age69', 'age74', 'age79', 'age84', 'age89', 'age90', 'density', 'male', 'female', 'con', 'lab', 'libdem', 'green', 'other', 'vacant', 'total', 'tq', 'nq', 'l1', 'l2', 'apprenticeship', 'l3', 'l4', 'oq', 'majority']
    allData = pd.read_csv('CouncilNonNorm3.csv', names=columns, sep=',', engine='python')


    for winningValues in mappings[x,y]:
       winningValues  = winningValues.reshape(1, -1) #reverting min scaler fit function
       winningValues  = sc.inverse_transform(winningValues) #reverting min scaler transform from 0,1 - data normalisation
       winningValues  = winningValues.flatten() #turning 2d array into 1d array
       results.append(np.around(winningValues , 0).astype(int)) #saving the values to an array and rounding the values to whole numbers.

    #for i in range(0,len(results)):
    for i in range(0,len(results)):
        #'tq', 'nq', 'l1', 'l2', 'apprenticeship', 'l3', 'l4', 'oq', 'majority']
        #answer = data.where((data['tq'] == str(int(results[i][0]))) & (data['nq'] == str(int(results[i][1]))) & (data['l1'] == str(int(results[i][2]))) & (data['l2'] == str(int(results[i][3]))) & (data['apprenticeship'] == str(int(results[i][4])))  & (data['l3'] == str(int(results[i][5]))) & (data['l4'] == str(int(results[i][6]))) & (data['oq'] == str(int(results[i][7]))))
        #answer = allData.where((allData['age4'] == str(int(results[i][0])))) #& (allData['age9'] == str(int(results[i][1]))) & (allData['age14'] == str(int(results[i][2]))) & (allData['age19'] == str(int(results[i][3]))))
        answer = allData.where((allData['nq'] == str(int(results[i][0]))) & (allData['l1'] == str(int(results[i][1]))) & (allData['l2'] == str(int(results[i][2]))))
        answer = answer.dropna(how='all')
        winningData = pd.concat([winningData, answer], ignore_index=True)

    return winningData

def GetNeuronWinnerNorm(x, y):
    mappings = som.win_map(dataQualificationsNorm)
    results = []

    for winningValues in mappings[x,y]:
        results.append(winningValues) #saving the values to an array and rounding the values to whole numbers.

    dfResults = pd.DataFrame(results, columns = ['No Qual','Level 1 Qual','Level 2 Qual','Apprenticeship','Level 3 Qual','Level 4 Qual','Other Qual'])
    return dfResults


bone() #select white, gray and black colour scheme for background.
pcolor(som.distance_map().T)
#euclidean distance between the neuron and its neighbour.
colorbar()
#Colours background based on distacne bewtween the neuron and its neighbour.
t = zeros(len(labels),dtype=int) #labels for each political party.
t[labels == 'CON'] = 0
t[labels == 'LAB'] = 1
t[labels == 'NOC'] = 2
markers = ['o','s','D',]
#Circle for Conservative, Square for Labour & Rhombus for NOC
colors = ['b','r','g']
#Blue for Conservative, Red for Labour & Green for No overall Control
for cnt,xx in enumerate(dataQualificationsNorm):
 w = som.winner(xx) #gives the data for the current winning neuron.
 plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',
   markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
 #Plots the winning data on the graph
axis([0,som._weights.shape[0],0,som._weights.shape[1]])
 #Shows a bar of neuron weights.
plt.title("Qualification SOM")
show()



d_x, d_y = zip(*[som.winner(d) for d in dataQualificationsNorm])
#packed into a list of iteratable tuples
d_x = np.array(d_x)
d_y = np.array(d_y) # converted into array

plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
plt.colorbar()

label_names = {'CON':0, 'LAB':1, 'NOC':2}
#Dictonary to label a political party to a number.
colors = ['b','r','g']

for c in np.unique(labels):
    targetRand = labels==c
    plt.scatter(d_x[targetRand]+.5+(np.random.rand(np.sum(targetRand))-.5)*.8,
                d_y[targetRand]+.5+(np.random.rand(np.sum(targetRand))-.5)*.8, 
                s=50, c=colors[label_names[c]], label=c)
    #Scatters the points within a neuron with random varibles for easier
    #way to analyse the data.
plt.legend(loc='upper right')
plt.grid()
plt.title("Qualification SOM")
plt.show()

#plt.figure(figsize=(7, 7))
frequency = som.activation_response(dataQualificationsNorm)
#Activation sequence of SOM in other words, frequency of data points
#within one neuron.
plt.pcolor(frequency.T, cmap='Blues') 
plt.colorbar() #Just plotting the frequency on the graph
plt.title("Qualification SOM")
plt.show()


#Age SOM displaying now
w_x, w_y = zip(*[som2.winner(d) for d in dataAgeNorm])
w_x = np.array(w_x)
w_y = np.array(w_y)

plt.figure(figsize=(10, 9))
plt.pcolor(som2.distance_map().T, cmap='bone_r', alpha=.2)
plt.colorbar()

label_names = {'CON':0, 'LAB':1, 'NOC':2}
colors = ['b','r','g']

for c in np.unique(labels):
    idx_target = labels==c
    plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                s=50, c=colors[label_names[c]], label=c)
plt.legend(loc='upper right')
plt.title("Age SOM")
plt.grid()
plt.show()






def classify2(som, data):
    winmap = som.labels_map(X_train2, y_train2)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

def classify(som, data):
    winmap = som.labels_map(X_train, y_train)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

def findDataFromQualToAgeSom(county, multiple):
    mappings = som.win_map(dataQualificationsNorm)
    ind_pos = [20, 21, 22, 23, 24, 25, 26]
    for x in range(0, 44):
        for y in range (0, 44):
            for winningValues in mappings[x,y]:
                winningValues = np.array(winningValues)
                county = np.array(county)
                county = county.flatten()
                county2 = county[ind_pos]
                
                if np.array_equal(winningValues, county2) == True:
                    ans = [x , y]
                    return ans

def findDataFromAgeToQualSom(county, multiple):
    mappings = som2.win_map(dataAgeNorm)
    ind_pos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    for x in range(0, 70):
        for y in range (0, 70):
            for winningValues in mappings[x,y]:
                winningValues = np.array(winningValues)
                county = np.array(county)
                county = county.flatten()
                county2 = county[ind_pos]
                
                if np.array_equal(winningValues, county2) == True:
                    ans = [x , y]
                    return ans

def getCountyNormData(county, multiple):
    answerMulti = pd.DataFrame()
    fullDataDf = pd.read_csv('CouncilNonNorm7.csv')
    nameCol = fullDataDf["Area name"]
    ageCol = pd.DataFrame(dataAgeNorm)
    qualCol = pd.DataFrame(dataQualificationsNorm, columns=["NQ", "L1", "L2", "Ap", "L3", "L4", "OQ"])

    AllCountyDataNorm = pd.concat([nameCol, ageCol, qualCol], axis=1)

    if multiple == True:
        for counties in county.iterrows():
            answer = AllCountyDataNorm.where(AllCountyDataNorm['Area name'] == counties[1]['council_name']) #county['council_name'][counties])
            answer = answer.dropna(how='all')
            answerMulti = pd.concat([answerMulti, answer])
    else:
        answer = AllCountyDataNorm.where(AllCountyDataNorm['Area name'] == county['council_name'][0])
        answer = answer.dropna(how='all')
        answerMulti = answer
        print("hi")

    return answerMulti

    #place in county
        #test = fullDataDf.where()
    #return "hi"

def GetNeuronWinner2(x, y):
    mappings = som2.win_map(dataAgeNorm)
    results = []
    winningData = pd.DataFrame()



    #columns=['council_name','age4', 'age9', 'age14','age19', 'age24', 'age29', 'age34', 'age39', 'age44', 'age49', 'age54', 'age59','age64', 'age69', 'age74', 'age79', 'age84', 'age90', 'con', 'lab', 'libdem', 'green', 'other', 'vacant', 'majority']
    columns=['council_name', 'all_person', 'age4', 'age9', 'age14','age19', 'age24', 'age29', 'age34', 'age39', 'age44', 'age49', 'age54', 'age59','age64', 'age69', 'age74', 'age79', 'age84', 'age89', 'age90', 'density', 'male', 'female', 'con', 'lab', 'libdem', 'green', 'other', 'vacant', 'total', 'tq', 'nq', 'l1', 'l2', 'apprenticeship', 'l3', 'l4', 'oq', 'majority']
    allData = pd.read_csv('CouncilNonNorm3.csv', names=columns, sep=',', engine='python')


    for winningValues in mappings[x,y]:
       winningValues  = winningValues.reshape(1, -1) #reverting min scaler fit function
       winningValues  = sc2.inverse_transform(winningValues) #reverting min scaler transform from 0,1 - data normalisation
       winningValues  = winningValues.flatten() #turning 2d array into 1d array
       results.append(np.around(winningValues , 0).astype(int)) #saving the values to an array and rounding the values to whole numbers.

    #for i in range(0,len(results)):
    for i in range(0,len(results)):
        #'tq', 'nq', 'l1', 'l2', 'apprenticeship', 'l3', 'l4', 'oq', 'majority']
        #answer = data.where((data['tq'] == str(int(results[i][0]))) & (data['nq'] == str(int(results[i][1]))) & (data['l1'] == str(int(results[i][2]))) & (data['l2'] == str(int(results[i][3]))) & (data['apprenticeship'] == str(int(results[i][4])))  & (data['l3'] == str(int(results[i][5]))) & (data['l4'] == str(int(results[i][6]))) & (data['oq'] == str(int(results[i][7]))))
        answer = allData.where((allData['age4'] == str(int(results[i][0])))) #& (allData['age9'] == str(int(results[i][1]))) & (allData['age14'] == str(int(results[i][2]))) & (allData['age19'] == str(int(results[i][3]))))
        #answer = allData.where((allData['nq'] == str(int(results[i][0]))) & (allData['l1'] == str(int(results[i][1]))) & (allData['l2'] == str(int(results[i][2]))))
        answer = answer.dropna(how='all')
        winningData = pd.concat([winningData, answer], ignore_index=True)

    return winningData



print("F1 Scores for Qualifications Data")
print(classification_report(y_test, classify(som, X_test)))
print("F1 Scores for Age Data")
print(classification_report(y_test2, classify2(som2, X_test2)))

QualNeuron = GetNeuronWinner(41,40)
Test = getCountyNormData(QualNeuron, True)
print("------")
print(Test.iloc[:,20:27])
print(findDataFromQualToAgeSom(Test.iloc[2], True))
print(findDataFromAgeToQualSom(Test.iloc[2], True))
print(GetNeuronWinner(41,40))
print(GetNeuronWinner2(59,69))
print("----------------------------")
dataFrameQualNorm = pd.DataFrame(dataQualificationsNorm, columns = ['No Qual','Level 1 Qual','Level 2 Qual','Apprenticeship','Level 3 Qual','Level 4 Qual','Other Qual'])


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


print(Test.iloc[:,20:27])
print(dataFrameQualNorm.describe())

#Combining The Age and Qual Normalized Data for Easier Searching for specific Locations.
fullDataDf = pd.read_csv('CouncilNonNorm7.csv')
nameCol = fullDataDf["Area name"]
ageCol = pd.DataFrame(dataAgeNorm, columns=["<4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", ">90"])
qualCol = pd.DataFrame(dataQualificationsNorm, columns=["NQ", "L1", "L2", "Ap", "L3", "L4", "OQ"])
AllCountyDataNorm = pd.concat([nameCol, ageCol, qualCol], axis=1)



cornwall = AllCountyDataNorm.where(AllCountyDataNorm['Area name'] == "Cornwall")
cornwall = cornwall.dropna(how='all')
print("Age Normalized Data for Cornwall")
print(cornwall)
east_sussex = AllCountyDataNorm.where(AllCountyDataNorm['Area name'] == "East Sussex")
east_sussex = east_sussex.dropna(how='all')
print("Age Normalized Data for East Sussex")
print(east_sussex)
gloucestershire = AllCountyDataNorm.where(AllCountyDataNorm['Area name'] == "Gloucestershire")
gloucestershire = gloucestershire.dropna(how='all')
print("Age Normalized Data for Gloucestershire")
print(gloucestershire)

CurrentCountyData = pd.concat([cornwall, east_sussex, gloucestershire], axis=0)
print("Age Data Norm of Cornwall, East Sussez and Gloucestershire")
print(CurrentCountyData)

print("Cornwall Location on the Age SOM: ")
print(findDataFromAgeToQualSom(cornwall, True))
print("East Sussex Location on the Age SOM: ")
print(findDataFromAgeToQualSom(east_sussex, True))
print("Gloucestershire Location on the Age SOM: ")
print(findDataFromAgeToQualSom(gloucestershire, True))





print(GetNeuronWinner(41,41))
print("41 41 above")
print(GetNeuronWinner2(60,68))
print(GetNeuronWinner2(60,67))
print(GetNeuronWinner2(58,66))
print("40, 40 below")
print(GetNeuronWinner(40,40))



 


#Box Plots for Qualifications
fig = sns.boxplot(data=[qualCol['NQ'],qualCol['L1'],qualCol['L2'],qualCol['Ap'],qualCol['L3'],qualCol['L4'],qualCol['OQ']],showmeans=True)
fig.set_title('Box Plots for Qualification Norm Data', fontsize=14, fontweight='bold')
fig.set_xticklabels(['NQ', 'L1', 'L2', 'Ap', 'L3', 'L4', 'OQ'])
plt.show()
#Box Plots for Age Part 1
fig2 = sns.boxplot(data=[ageCol['<4'], ageCol['5-9'], ageCol['10-14'], ageCol['15-19'], ageCol['20-24'], ageCol['25-29'], ageCol['30-34'], ageCol['35-39'], ageCol['40-44'], ageCol['45-49']],showmeans=True)
fig2.set_title('Box Plots for Age Norm Data Part 1', fontsize=14, fontweight='bold')
fig2.set_xticklabels(['<4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49'])
plt.show()
#Box Plots for Age Part 2
fig2 = sns.boxplot(data=[ageCol['50-54'], ageCol['55-59'], ageCol['60-64'], ageCol['65-69'], ageCol['70-74'], ageCol['75-79'], ageCol['80-84'], ageCol['85-89'], ageCol['>90']],showmeans=True)
fig2.set_title('Box Plots for Age Norm Data Part 2', fontsize=14, fontweight='bold')
fig2.set_xticklabels(['50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '>90'])
plt.show()



xvar = input("Write the X-axis: ")
yvar = input("Write the Y-axis: ")
print(GetNeuronWinner(int(xvar),int(yvar)))
inploc = input("Write the loc: ")
#print(dataQualifications[int(inploc)])
print(dataAge[5])
test = FindNeuronWinner(dataAge[5]) #FindNeuronWinner(dataAgeNorm[int(inploc)])
print(test)
#print(GetNeuronWinner(int(test[0]),int(test[1])))
#print(dataQualificationsNorm[0])
#(FindNeuronWinner(dataQualificationsNorm[0]))
#print(GetNeuronWinner(13,0))


