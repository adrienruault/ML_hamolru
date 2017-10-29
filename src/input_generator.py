




def main():
    
    min_deg = np.zeros((27,30))+1
    max_deg = np.zeros((27,30))+3

    for k in range(3):
        for j in range(3):
            for i in range(3):
                print(i + 3*j + 9*k)
                min_deg[i + 3*j + 9*k][0] = (i+1)
                min_deg[i + 3*j + 9*k][1] = (j+1)
                min_deg[i + 3*j + 9*k][2] = (k+1)
                
                max_deg[i + 3*j + 9*k][0] = (i+1)
                max_deg[i + 3*j + 9*k][1] = (j+1)
                max_deg[i + 3*j + 9*k][2] = (k+1)
                
    min_deg = min_deg.astype(int).tolist()
    max_deg = max_deg.astype(int).tolist()
    result=[]
    for i in range(27):
        result += [min_deg[i], max_deg[i]]
    
    return result 





