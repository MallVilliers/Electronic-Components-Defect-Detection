import os
def get_filelist(dir, Filelist):
    
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            if s.__contains__(".DS_Store")or s.__contains__('归档良品') or s.__contains__('归档不良品') or s.__contains__("抛弃数据")or s.__contains__("其他"):
                continue
            newDir=os.path.join(dir,s) 
            get_filelist(newDir, Filelist) 
    return Filelist 
if __name__ =='__main__' :
    list = get_filelist('/home/qiqi/fast_repvgg/train_data_body_0216', []) 
    print(len(list)) 
    f=open("/home/qiqi/fast_repvgg/mytxt/train_data_body_0216.txt","w")
    for line in list:
        if line.__contains__('0_pass') and line.__contains__('jpg'):
            f.write(line+"\t"+'0'+'\n')
        elif line.__contains__('1_fail') and line.__contains__('jpg') :
            f.write(line+"\t"+'1'+'\n')
    f.close()
    # for e in list:
    #     print(e)