import numpy as np
import matplotlib.pyplot as plt
import os



def read_test_acc(filename):
    array=[]
    max_run=0
    with open(filename) as f:
        for line in f:
            if ('Run'in line.strip() )and ( 'Test' in line.strip()):
                # print(type(acc))
                acc=line.split()[-1]
                run=line.split()[1]
                if ',' in run:
                    run=run.strip(',')
                run=int(run)
                max_run = run if run > max_run else max_run
                # print(type(acc))
                if '%' in acc:
                    acc=acc[:-1] 
                    acc=float(acc)
                    acc=float("{0:.4f}".format(acc/100))
                else:
                    acc=float(acc)
                array.append(acc)
    print(array[:10])
    print(len(array))
    return array, max_run+1


def draw(DATASET,  model, my_full, pseudo_mini_batch, path, n_run, fan_out=None):
    
    fig,ax=plt.subplots(figsize=(24,6))
    # x=range(len(bench_full))
    length_full=len(my_full)
    len_pseudo=len(pseudo_mini_batch)
    if n_run>1:
        len_pseudo=int(len_pseudo/n_run)
    if len_pseudo<=100:
        fig,ax=plt.subplots(figsize=(6,6))
    if len_pseudo<=500 and len_pseudo>100:
        fig,ax=plt.subplots(figsize=(12,6))
    len_cut = len_pseudo if len_pseudo < length_full else length_full
    my_full=my_full[:len_cut]
    pseudo_mini_batch=pseudo_mini_batch[:len_cut]
    x1=range(len(my_full))
    x2=range(len(pseudo_mini_batch))
    # ax.plot(x, bench_full, label='benchmark '+DATASET )
    
    ax.plot(x1, my_full, label='my script full graph '+DATASET)
    ax.plot(x2, pseudo_mini_batch, label='pseudo_mini_batch_full_batch '+DATASET + '_fan-out_'+str(fan_out))
    ax.set_title(model+' '+DATASET)
    plt.ylim([0,1])
    plt.xlabel('epoch')
    
    # fig,ax=plt.subplots()
    # ax.autoscale(enable=True,axis='y',tight=False)
    # y_pos= np.arange(0,1000,step=100)
    # labels=np.arange(0,1,step=0.1)
    # print(labels)
    # plt.yticks(y_pos,labels=labels)
    plt.ylabel('Test Accuracy')
    
    plt.legend()
    # plt.savefig('reddit.pdf')
    plt.savefig(path+DATASET+'.png')
    # plt.show()

def get_fan_out(filename):
    fan_out=filename.split('_')[6]
    print(fan_out)
    return fan_out


if __name__=='__main__':
    # bench_path = '../../benchmark_full_graph/logs/'
    # files= ['arxiv']
    files= ['products']
    # files= ['cora', 'pubmed', 'reddit', 'arxiv', 'products']
    my_path = '../../my_full_graph/logs/'
    pseudo_mini_batch_path = '../../pseudo_mini_batch_full_batch/logs/'
    model='sage'
    model_p='sage/'
    # model = 'gat'
    # model_p='gat/'
    my_full=[]
    pseudo_mini_batch=[]

    
    my_path = my_path+model_p+'acc_bak/'
    # pseudo_mini_batch_path = pseudo_mini_batch_path+model_p+'10_runs/'
    pseudo_mini_batch_path = pseudo_mini_batch_path+model_p+'1_runs/'
    for file_in in files:
        n_run=0
        for filename in os.listdir(my_path):
            if filename.endswith(".log"):
                f = os.path.join(my_path, filename)
                if file_in in f:
                    my_full, n_run_full = read_test_acc(f)
        f_i=0
        for filename in os.listdir(pseudo_mini_batch_path):
            if filename.endswith(".log"):
                f = os.path.join(pseudo_mini_batch_path, filename)
                if file_in in f:
                    print(f)
                    f_i+=1
                    pseudo_mini_batch, n_run = read_test_acc(f)
                    fan_out = get_fan_out(filename)
                    draw(file_in,  model, my_full, pseudo_mini_batch, pseudo_mini_batch_path+'convergence_curve/'+str(f_i)+'_', n_run,fan_out)
                    pseudo_mini_batch=[]
        my_full=[]
        
        print()
    