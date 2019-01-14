import numpy as np

def conv_to_csv(dataset, filename):
    path = '../../embed/'+dataset
    fpath = path+'/'+filename
    print fpath
    
    with open(fpath,'r') as f:
        data = f.read().split('\n')
        if len(data[-1])==0:
            data.pop()

        l=data[0]
        l=l.split(' ')
        node_count = l[0]
        print('Node count in %s dataset ::' %dataset, node_count)

        mat2 = np.zeros((int(node_count),int(l[1])))

        for line in data[1:]:
            l = line.split(' ')
            l = [float(x) for x in l]
            ind = int(l[0])
            mat2[ind,:] = l[1:]    

        print mat2.shape
        np.savetxt(fpath+'.csv',mat2,fmt='%.6e')



