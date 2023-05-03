
# import required module
import os
import numpy as np
import matplotlib.pyplot as plt

# assign directory
directory = 'C:/Users/czyji/Downloads/673-marl-20230503T165646Z-001/673-marl/IQN_results_different_agents'

def plot_data(fdata,data_color):
# read test episode series
    data_X = fdata['episodes']
    
    data_Y_agent = {}
    mean_Y_agent = {}
    
    for k in fdata.keys():
        if k == 'episodes':
            continue
    
        data_Y = fdata[k]
        mean_Y = None
    

        window=1000
        pfx_Y = [data_Y[0]] * (window // 2)
        sfx_Y = [data_Y[-1]] * (window // 2)

        cmb_Y = np.concatenate((pfx_Y, data_Y, sfx_Y))
        mean_Y = []

        for i in range(len(data_Y)):
            lb = data_Y[0]
            rb = data_Y[-1]

            start = i - window // 2
            total = 0
            count = 0

            for j in range(start, start + window):
                if j < 0:
                    total += lb
                elif j >= len(data_Y):
                    total += rb
                else:
                    total += data_Y[j]

                count += 1

            mean_Y.append(total / count)

    data_Y_agent[k] = data_Y
    mean_Y_agent[k] = mean_Y

    fig, ax = plt.subplots()
    
    
    
    ax.set_title('result_of_DRLs')
    ax.set_ylabel('episode')
    ax.set_xlabel('reward')
    
    lines = {}
    mean_lines = {}
    
    for k in data_Y_agent.keys():
        data_Y, mean_Y = data_Y_agent[k], mean_Y_agent[k]
        
        lines[k], = ax.plot(data_X,
                data_Y, 
                data_color + '-',
                alpha=0.25)

        mean_lines[k], = ax.plot(data_X,
                mean_Y,
                data_color + '-', alpha=0.7,
                label=k)
    
    ax.legend()
    
    while True:
        for k in data_Y_agent.keys():
            data_Y, mean_Y = data_Y_agent[k], mean_Y_agent[k]
            
            lines[k].set_data(data_X, data_Y)
    
            plt.ylim((min(data_Y), max(plt.ylim()[1], max(data_Y))))
            plt.xlim((min(data_X), max(plt.xlim()[1], max(data_X))))
    
           
            mean_lines[k].set_data(data_X, mean_Y)
    

            if not plt.get_fignums():
                break
        else:
            plt.show()
            break
    

        #ax.set_size_inches(args.width, args.height)
        plt.savefig('results.png')

# iterate over files in
# that directory

# that directory
data_Y={}
mean_Y={}
for filename in os.listdir(directory):
    loc = os.path.join(directory, filename)
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k',]
    color=colors.pop(0)
    
    with open(loc, 'r') as f:
        fdata = eval(f.read())
        for k in fdata.keys():
            if k == 'episodes':
                continue  
            data_Y[filename[6:-5]] = np.array(fdata[k])
            mean_Y[filename[6:-5]]=[]
for ele in data_Y.keys():
    if len(data_Y[ele])<10000:
        temp=data_Y[ele]
        data_Y[ele]=np.ones(10000)*-10000
        data_Y[ele][0:temp.size]=temp[0:temp.size]

    window=50
    for i in range(len(data_Y[ele])):
        lb = data_Y[ele][0]
        rb = data_Y[ele][-1]

        start = i - window // 2
        total = 0
        count = 0

        for j in range(start, start + window):
            if j < 0:
                total += lb
            elif j >= len(data_Y[ele]):
                total += rb
            else:
                total += data_Y[ele][j]

            count += 1

        mean_Y[ele].append(total / count)
for ele in data_Y.keys():
    plt.plot(mean_Y[ele],label=ele+'mean')
    #plt.plot(data_Y[ele],label=ele+'data')
    plt.legend()
        
        #plot_data(fdata,color)


