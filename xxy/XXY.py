import pandas as pd

data1 = pd.read_excel('D:\Desktop\work.xlsx', sheet_name='Sheet1')
data2 = pd.read_excel('D:\Desktop\work.xlsx', sheet_name='×系数')
data3 = pd.read_excel('D:\Desktop\work.xlsx', sheet_name='阈值')
"""
AL6 = len(data2[(data2['y'] == 0) & (data2['是否准确'] == 0)])
AL7 = len(data2[(data2['y'] == 1) & (data2['是否准确'] == 0)])
AL8 = len(data2[(data2['y'] == 0) & (data2['是否准确'] == 1)])
AL9 = len(data2[(data2['y'] == 1) & (data2['是否准确'] == 1)])
"""
AL3 = 108
AL5 = 14074
AL4 = AL5 - AL3

AN2 = 0.00767372459855052
AN3 = 40

c_list = data2['y'].tolist()
af_list= data2['Xscore'].tolist()

min_AN6=10
for AL in [x / 100 for x in range(-1000, 1001)]:
    print(f"AL:{AL}")

    ag_list = []
    for value in af_list:
        if value > AL:
            ag_list.append(1)
        else:
            ag_list.append(0)
    #print("ag_list",len(ag_list))
    ah_list = []
    for i in range(len(c_list)):
        if c_list[i] == ag_list[i]:
            ah_list.append(1)
        else:
            ah_list.append(0)
    #print("ah_list",len(ah_list))

    AL6 = 0
    for i in range(len(c_list)):
        if c_list[i] == 0 and ah_list[i] == 0:
            AL6 += 1
    print(f"AL6:{AL6}")

    AL7 = 0
    for i in range(len(c_list)):
        if c_list[i] == 1 and ah_list[i] == 0:
            AL7 += 1
    print(f"AL7:{AL7}")

    AL8 = 0
    for i in range(len(c_list)):
        if c_list[i] == 0 and ah_list[i] == 1:
            AL8 += 1
    print(f"AL8:{AL8}")

    AL9 = 0
    for i in range(len(c_list)):
        if c_list[i] == 1 and ah_list[i] == 1:
            AL9 += 1
    print(f"AL9:{AL9}")
    AN4 = AL7 / AL3
    print(f"AN4:{AN4}")
    AN5 = AL6 / AL4
    print(f"AN5:{AN5}")
    AN6 = AN2 * AN3 * AN4 + (1 - AN2) * (AL6 / AL4)
    print(f"AN6:{AN6}","\n")
    if AN6 < min_AN6:
        min_AN6 = AN6
        min_AL=AL
print("最小AN6值为：", min_AN6)
print("最小AN6值对应的阈值为：", min_AL)