def load_samples():
    # loads pertinent information regarding ECG records (times and durations of balloon inflations)
    # 'timings.txt' and 'balloons.txt' contain info pulled from STAFF III datasheet

    samples=[]
    exclusion_list=[1,4,5,6,82,89] # see paper for exclusin criteria
    f=open('timings.txt','r')
    txt=f.read()
    t_list=txt.split('\n')
    g=open('balloons.txt','r')
    balloon_txt=g.read()
    b_list=balloon_txt.split('\n')
    for i in range(len(t_list)):
        t=t_list[i]
        if t!='X':
            if (i+1 not in exclusion_list):
                t_split=t.split(';')
                if (int(t_split[2]))>90:
                    s=[]
                    s.append(i+1)
                    s.append(t_split[0])
                    s.append(t_split[1])
                    s.append(t_split[2])
                    s.append(b_list[i].split(str(i+1))[-1])
                    samples.append(s)
    return samples