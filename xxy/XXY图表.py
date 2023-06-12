import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_excel('D:\Desktop\work.xlsx', sheet_name='Sheet1')
data2 = pd.read_excel('D:\Desktop\work.xlsx', sheet_name='×系数')
data3 = pd.read_excel('D:\Desktop\work.xlsx', sheet_name='阈值')

AL3 = 108
AL5 = 14074
AL4 = AL5 - AL3
AN2 = 0.00767372459855052
AN3 = 40

c_list = data2['y'].tolist()
af_list= data2['Xscore'].tolist()

AN4_list = []
AN5_list = []
AL_list = []

for AL in [x / 100 for x in range(-1000, 1001)]:
    ag_list = []
    for value in af_list:
        if value > AL:
            ag_list.append(1)
        else:
            ag_list.append(0)

    ah_list = []
    for i in range(len(c_list)):
        if c_list[i] == ag_list[i]:
            ah_list.append(1)
        else:
            ah_list.append(0)

    AL6 = 0
    for i in range(len(c_list)):
        if c_list[i] == 0 and ah_list[i] == 0:
            AL6 += 1

    AL7 = 0
    for i in range(len(c_list)):
        if c_list[i] == 1 and ah_list[i] == 0:
            AL7 += 1

    AL8 = 0
    for i in range(len(c_list)):
        if c_list[i] == 0 and ah_list[i] == 1:
            AL8 += 1

    AL9 = 0
    for i in range(len(c_list)):
        if c_list[i] == 1 and ah_list[i] == 1:
            AL9 += 1

    AN4 = AL7 / AL3
    AN5 = AL6 / AL4
    AN6 = AN2 * AN3 * AN4 + (1 - AN2) * (AL6 / AL4)

    AN4_list.append(AN4)
    AN5_list.append(AN5)
    AL_list.append(AL)


plt.plot(AL_list, AN4_list, label='AN4')
plt.plot(AL_list, AN5_list, label='AN5')


plt.title('AN4 and AN5 vs AL')
plt.xlabel('AL')
plt.ylabel('AN4 and AN5')

plt.legend()

plt.show()

