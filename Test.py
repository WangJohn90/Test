#该程序包含单次的重规划的路径生成算法，但不包含重规划的触发条件
#dl_near,dl_far,idxnear,x_pl[],y_pl[]
# -*- coding: utf-8 -*-
#IP config:192.168.1.93  255.255.255.0
#本机WIN10防火墙关闭！！！
#采集自车相对前车车速以及距离有助于判断驾驶员换道意图
#控制的时候回送进程的SmarteyeC语言程序应当关闭重启，本程序也应当关闭重启
#总结20190626:1.主进程更新频率最高4Hz（ykmPC上），再高则会影响规划路径回送数据的更新 2.预瞄距离需要延长，3.传路径的时候能否在未收到新路径的时候保持原有路径？
#a6opt =-2.8566e-12对应LCD=10s的情形；a6opt =-8.0620e-12对应LCD=8s情形；a6opt=  -1.7319e-11对应LCD=6s的情形
#snr文件：...03为定速，...03_freeACC
"""
Created on Tue Apr 23 22:59:38 2019
@author: Yang Kaiming & Yan Zhanhong
"""
import numpy as np
import matplotlib.pyplot as plt
import socket
import threading
import time
import math as ma 
#from numpy import random
from scipy import interpolate
#from numba import jit

import config
import model
import tensorflow as tf
#import GRU_FF
import bezier
#from numba import jit
#################################socket通信初始化########################################   
#每过3秒切换一次账号
print("Begin...")
#   创建套接字
ip = ""
port =1632
port3=port+6500    #Python发送端口
own_addr = (ip, port)  # 接收方端口信息
own_addr3=('192.168.1.93', port3)
#own_addr3=('127.0.0.1', port3)
byte = 23000

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind(own_addr)  # 绑定端口信息
udp_socket3 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket3.bind(own_addr3)  # 绑定端口信息
print("Waiting for connecting....")
print("New connection")
#########################################################################################

##############CNN/gru model initialization###############################################
try:
    tf.reset_default_graph()
except:
    pass
FLAGS = tf.app.flags.FLAGS
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
sess = tf.Session(config=session_config)
print ('building network...')
classifier = model.MODEL(is_training=False)
saver = tf.train.Saver(max_to_keep=None)     #reloading the trained model
saver.restore(sess,FLAGS.checkpoint_path)     #Open the optimal model after training
print ('restoring from '+FLAGS.checkpoint_path)
print ('reading completed')
input_nlcd=np.zeros((1,1600))


#print ('reading completed')
#input_nlcd=np.zeros((1,1600))

#model_GRU=GRU_FF.GRU_model()
#def GRU_Pre(input_nlcd):
#    data_test=GRU_FF.ReshapeGRU(input_nlcd.shape[0],input_nlcd)
#    pre = model_GRU.predict(data_test)
#    tlc=pre[0,0];tlcc=pre[0,1];
#    return tlc, tlcc

########################## Parameters Initialization ##########################################
f1 = open("Llanelong.csv", "r")
data1 = [line.strip("\n").split(",") for line in f1]
Llane = np.asarray(data1,dtype='float16')
f2 = open("Rlanelong.csv", "r")
data2 = [line.strip("\n").split(",") for line in f2]
Rlane = np.asarray(data2,dtype='float16')

TLC_list=[];LCD_list=[];Td_list=[]
msg=[];data_batch=[]
other_addr=0
interval1=0;interval2=0;interval3=0
time1=0;time2=0;time3=0
plt_pause=0.00000001
n_sam=200;n_x=10    #增加了teye以及tDS
tm=0;td=0;vy=0;dpsi=0;dvf=0;dvr=0;dym=0;Xm=0;Ym=0;is_conflict_YAN=0
X=0
X_plstart=0
Y=Llane[0,2]
LCD=0;TLC=4
xT = 100
yT = Llane[0,2]-Rlane[0,2]
ddx=0.5
X_PL=np.linspace(0,xT,80)
Y_PL=10 * yT*np.power((X_PL/xT),3) - 15 * yT*np.power((X_PL/xT),4) + 6 * yT*np.power((X_PL/xT),5);
vx=16
L_pre=15   #预瞄距离
x_ref=list(Llane[:100,1])
y_ref=list(Llane[:100,2])
count=0
jj=0

xsas=[];  ysas=[]
x_send=[]; y_send=[]

for i in range(len(x_ref)): 
    x_send.append(str(round(x_ref[i],2)))
    x_send.append(',')
    y_send.append(str(round(y_ref[i],3)))           
    y_send.append(',')  
flg_rec=0
flg_send=1
flg_pl=0       #0表示在左车道，1表示换道中，2表示在右车道
flg_lr=0       #0表示左换道，1表示右换道,对应的是按照此flag换道后汽车在哪个车道
flg_ini=1      #0表示已进行路径规划，1表示未进行路径规划
flg_inire=1   #0表示无冲突，1表示需要进行冲突换道重规划，2表示需要进行回原车道保持重规划
islc=0         #0表示车道保持，1表示换道
isconflict=0   #0表示无需重规划，1表示人机冲突超限，需要重规划轨迹
kcnn=0

a11=np.zeros(n_sam-1)
a11=a11.reshape(-1,1)
a12=np.eye(n_sam-1)
a2=np.zeros(n_sam)
a2=a2.reshape(1,-1)
bb=np.zeros((n_sam,1))
bb[n_sam-1]=1
A=np.concatenate((a11,a12),axis=1)
A=np.concatenate((A,a2),axis=0)
input_batch=np.zeros((n_sam,n_x))
b1=np.zeros((n_sam-1,n_x))
data_new=np.zeros((1,n_x))       #用于识别意图的数据
data_float=np.zeros(n_sam+2)   #从smart eye电脑接收的数据

def CNN_FF(input_nlcd):    
    feed_dict={}
    feed_dict[classifier.input_nlcd]=input_nlcd
    feed_dict[classifier.keep_prob]=1.0					
    indiv_prob= sess.run([classifier.indiv_prob],feed_dict) 
    cc=indiv_prob[0]
    tlc_pro=cc[0][0]
    lct_pro=cc[0][1]     
    return tlc_pro,lct_pro


# 初始路径规划
def Ini_planYAN(X,Y,x_ini,y_ini,xT,yT,x_mid):
    x_pl0=np.linspace(X-3,x_ini-0.3,int((x_ini-X+3)/0.3))   #换道决策触发的Xini-Xm后开始换道  ，这里为-3m
    y_pl0=y_ini*np.ones((int((x_ini-X+3)/0.3)))
    
    if x_mid>xT:
        x_mid=xT/2
    
    nodes1 = np.asfortranarray([
            [0    ,  x_mid/2 , x_mid, x_mid , (x_mid+xT)/2, xT],
            [0    ,  0       , 0    , yT    ,  yT         , yT]])

    curve1 = bezier.Curve(nodes1, degree=6)
    s_vals= np.linspace(0,1,int(xT/0.3))            
    path=curve1.evaluate_multi(s_vals)
    X_PL=path[0,:]
    Y_PL=path[1,:]
 
    x_pl1=x_ini+X_PL                            #bezier五阶换道
    y_pl1=y_ini+Y_PL 
    
    x_pl2=np.linspace(x_pl1[-1]+0.3,x_pl1[-1]+150,int(150/0.3))    #换道后段100m
    y_pl2=y_pl1[-1]*np.ones(int(150/0.3))
    
    x_pl=list(x_pl0)+list(x_pl1)+list(x_pl2)
    y_pl=list(y_pl0)+list(y_pl1)+list(y_pl2)
    return x_pl,y_pl
 
################Thread 2: data receving 每隔1/60秒钟执行一次############################
#@jit(nopython=True, nogil=True)
def t2():          # communication & path planning function
    global interval1
    global time1
    global udp_socket
    global other_addr
    global data_batch
    global X,Y,td,tm,vy,dpsi,dvf,dvr,yd_est,Xm,Ym,is_conflict_YAN
    global x_send
    global y_send
    global flg_pl
    global flg_rec
    global msg
    while 1:
        ##################### Data_receving #######################
#        print ('定时器中断,interval1=',interval1)        
        msg, other_addr = udp_socket.recvfrom(byte)
#        print('Receiving thread Receiving from ip:',other_addr,'  received data:',msg)     
#        print("Length of msg:",len(msg))                 
        data_str1=str(msg, encoding = "utf-8")
        data_str=data_str1.split(",")
        
#        print("Length of data_str:",len(data_str))
#        print("data_str=",data_str)       
        X=float(data_str[-12])
        Y=float(data_str[-11])
        td=float(data_str[-10])
        try:
            tm=float(data_str[-9])
        except:
            tm=0;
        try:
            vy=float(data_str[-8])
        except:
            vy=0
        dpsi=float(data_str[-7])
        dvf=float(data_str[-6])
        dvr=float(data_str[-5])
        try:
            yd_est=float(data_str[-4])  
        except:
            yd_est=0
        Xm=float(data_str[-3]) 
        Ym=float(data_str[-2]) 
        is_conflict_YAN=float(data_str[-1]) 
#        is_conflict_YAN=0
        
#        print('Receiving thread Receiving from ip:',other_addr,'  X=',X,'  Y=',Y)
        data_batch=data_str[:2000]    #若包含两个时间戳，则需要到2000
        flg_rec=1

############################## Thread3: data sending ################################
#@jit(nopython=True, nogil=True)
def t3():
    global interval3
    global time3
    global udp_socket3
    global own_addr3
    global flg_send,flg_pl,LCD,TLC
    while 1:
        newtime3 = time.clock()
        interval3 = (newtime3 - time3)
        if interval3 >=1/20:
#            print('good_thread3!')
            msg3, other_addr3 = udp_socket3.recvfrom(byte)
#            if flg_send==1:
#                msg3, other_addr3 = udp_socket3.recvfrom(byte)
#                print('other_addr3=',other_addr3)
#                flg_send=0
            time3=newtime3 
#            print('x_send=',x_send)
#            send_data=''.join(list(str(flg_pl))+list(',')+list('111')+list(',')+list('122')+list(',')+list('\0'))
            if x_send!=[] and y_send!=[]:
                send_data=''.join(list(str(1))+list(',')+list(x_send)+list(y_send)+[str('%.2f' %(TLC))]+list(',')+[str('%.2f' %(LCD))]+list('\0'))
#        send_data=''.join(list(str(1))+list(',')+list(str(10))+list(',')+list(str(11))+list(',')+list(str(12))+list(',')+list('\0'))
            data_send=bytes(send_data, encoding = "utf8") 
            udp_socket3.sendto(data_send, other_addr3)         
#########################################################################################
        
################################### Main function ########################################
if __name__ == '__main__':
#    global interval2
#    global time2
    t = threading.Thread(target=t2)
    tiii=threading.Thread(target=t3)
    t.start()       #启动眼动信息接收进程
    tiii.start()     #启动参考路径发送进程
    while(1):
        newtime2 = time.clock()
        interval2 = (newtime2 - time2)
        if interval2 >=1/20:
            time2=newtime2
###################################驾驶员意图识别模型######################################    
            input_nlcd = np.array(list(map(eval, data_batch)))
#            if flg_rec!=0:
#                print('dteye=',input_nlcd[1799]-input_nlcd[1600],'dtds=',input_nlcd[1999]-input_nlcd[1800])
            input_nlcd =input_nlcd[:1600]
            input_nlcd =input_nlcd.reshape(1,-1) 
            if flg_rec==0:
                input_nlcd=np.zeros((1,1600))           
#            print('gazex=',input_nlcd[0,199],' ax=',input_nlcd[0,399],' head=',input_nlcd[0,599],' sw_deg=',input_nlcd[0,799],' ey=',input_nlcd[0,999],' epsi=',input_nlcd[0,1199],' vx=',input_nlcd[0,1399],' ay=',input_nlcd[0,1599])
            TLC,LCD=CNN_FF(input_nlcd)
#            TLC,LCD=GRU_Pre(input_nlcd)
#            print('TLC=',TLC,'LCD=',LCD)
        
#            plt.figure(1) #get the value of global_step 
#            plt.cla()
#            TLC_list.append(TLC)
#            LCD_list.append(LCD)
##            Td_list.append(td)
#            if len(TLC_list)>100:
#                TLC_list.pop(0)
#                LCD_list.pop(0)                                                                                   
#            plt.plot(TLC_list)
##            plt.plot(LCD_list)
#            plt.pause(plt_pause)
            '''
            plt.figure(2) #get the value of global_step 
            plt.cla()
            LCD_list.append(LCD) 
            if len(LCD_list)>100:
                LCD_list.pop(0)                                                                       
            plt.plot(LCD_list)
            plt.pause(plt_pause)
            
            '''
            
            ##########################################################
            psi=-input_nlcd[0,1199]/57.3  #在线调试
#            print('psi=',psi)
            ############## 离线模式Initial Path planning###############            
#            print('x_send=',x_send,'\n')
#            print('y_send=',y_send,'\n')
            #轨迹规划区间长度为100vx*Ts，Ts为路径规划采样时间            
            #离线TLC设置
            #调整TLC
            
#            X=X+vx/6   #离线测试
#            if (X>=200 and X<=260) or (X>=500 and X<=560):   
#                TLC=3.6;LCD=6
#            else:
#                TLC=0;LCD=0                ##离线测试
#            f1=interpolate.interp1d(x_ref,y_ref,kind='linear')#离线测试
#            try:
#                Y=float(f1(X))         #离线测试，在线时注释掉
#            except:
#                Y=float(f1(X+0.4))
#            psi=ma.atan((f1(X+0.5)-f1(X+0.4))/0.1)
#            TLC=0    #一直车道保持  

###############################人机冲突判断模块######################################



            
######################若远离换道区域，则屏蔽意图识别，避免假阳性的出现####################
            if abs(Y-7.8)<1 and dvf>80:
                TLC=0
#                flg_conflict=0
#                print('不在右换道区域内！','dvf=',dvf)    
            if abs(Y-4.15)<1 and 0<dvr<60:
                TLC=0
#                flg_conflict=0
#                print('不在左换道区域内！','dvr=',dvr) 
                 
            
            
############################无意图识别时的主动换道 for debug#####################################
#            if (X>=200 and X<=260) or (X>=500 and X<=560):   
#                TLC=3;LCD=7
#            else:
#                TLC=0;LCD=0 
##                                                
############################换道逻辑##############################   
        
            if islc==0 and TLC>=3 and X-X_plstart>100:
                X_plstart=X
                print("触发换道!!!!!!,TLC=",TLC)
                islc=1
                
            if X<100:
                TLC=0;X_plstart=0;flg_pl=0;flg_lr=0;flg_ini=1;flg_inire=1;is_conflict_YAN=0;islc=0
            """Lane change sharpness paremeter"""
#            LC_sharpness_time=4      #For SHARP turn
#            LC_sharpness_time=6      #For NORMAL turn
            LC_sharpness_time=8      #For SLOW turn
            """Lane change sharpness paremeter"""
                     
            if is_conflict_YAN==0:
                if islc==1:       #换道                      
                    if flg_lr==1 and flg_ini==1:    #左换道路径规划
                        x_ini=L_pre+X-3
      #                      x_ini=X+(6-TLC-2)*vx        #2是对标签滞后的补偿，开始换道点至车轮压线点的时间长度约2s
                        y_ini=Rlane[0,2]
#                        y_ini=Y                                            
                        xT=vx*(LC_sharpness_time)    #换道持续时间
                        yT=Llane[0,2]-y_ini;
                        x_mid=LC_sharpness_time/2*vx; x_pl,y_pl=Ini_planYAN(X,Y,x_ini,y_ini,xT,yT,x_mid)#当前车辆位置，换道起始位置，换道终点相对于换道起点的位置  
                        flg_ini=0
                        flg_lr=0
                    if flg_lr==0 and flg_ini==1:    #右换道路径规划
                        x_ini=L_pre+X
    #                        x_ini=X+(6-TLC-2)*vx
                        y_ini= Llane[0,2] 
#                        y_ini=Y
                        xT=vx*LC_sharpness_time      #换道持续时间
                        yT=Rlane[0,2]-y_ini;
                        x_mid=LC_sharpness_time/2*vx; x_pl,y_pl=Ini_planYAN(X,Y,x_ini,y_ini,xT,yT,x_mid)#
                        flg_ini=0
                        flg_lr=1
                    
                    dx=list(abs(np.array(x_pl)-X)) 
                    count=dx.index(min(dx)) 
                    x_ref=x_pl[count-60:count+140:2]
                    y_ref=y_pl[count-60:count+140:2]                                                        
                    if (flg_lr==0 and abs(Y-Llane[0,2])<=0.2) :
                        islc=0;print("左换道结束！")                    
                    if (flg_lr==1 and abs(Y-Rlane[0,2])<=0.2):     #换道结束，将islc置零
                        islc=0 ;print("右换道结束！")                                               
                else:     #islc==0车道保持                
                    flg_ini=1    
                    flg_inire=1
                    if flg_lr==0:#左侧车道保持                     
                        dx=list(abs(Llane[:,1]-X)) 
                        count=dx.index(min(dx))
    #                        f2=interpolate.interp1d(Llane[:,1],Llane[:,2],kind='linear')
                        x_ref=Llane[count:count+200:2,1]
                        y_ref=Llane[count:count+200:2,2]
                    else:#右侧车道保持
                        dx=list(abs(Rlane[:,1]-X)) 
                        count=dx.index(min(dx))
    #                        f3=interpolate.interp1d(Rlane[:,1],Rlane[:,2],kind='linear')
                        x_ref=Rlane[count:count+200:2,1]
                        y_ref=Rlane[count:count+200:2,2]  
            ################轨迹重规划模块######################

################Modul from YAN Zhanhong###################### 
            
            elif is_conflict_YAN==1:   #冲突检测，重规划
               print('Conflict Detected！！！！,x= ',X)
               if (abs(Y-Llane[0,2]))<(abs(Y-Rlane[0,2])):
                    print('Go back to Lane keeping mode,to LEFT lane!!!')
                    islc=0;flg_lr=0;   # is_conflict_YAN=0                  
               else:     #换道结束，将islc置零
                    print('Go back to Lane keeping mode,to RIGHT lane!!!')
                    islc=0;flg_lr=1;   # is_conflict_YAN=0 
                    
                    
            x_send=[]
            y_send=[]
            for i in range(len(x_ref)): 
                x_send.append(str('%.2f' %(x_ref[i])))
                x_send.append(',')
                y_send.append(str('%.3f' %(y_ref[i])))           
                y_send.append(',') 
            
################Modul from YAN Zhanhong######################         
            
            
        #            print('main thread: X=',X,'  Y=', Y,' islc=',islc,'  flg_lr=',flg_lr,'   flg_ini=',flg_ini,'  flg_conflic=',flg_conflict,'  flg_inire=',flg_inire,' is_conflic_YAN=',is_conflic_YAN)
#            print('x_send=',x_send)
#            print('X=',X,'islc=',islc)
#            print('-----------------------------------------------------------------------------------')            
#                plt.plot(x_pl,y_pl)                
            
    ##################Plotting the target trajectory##########################           
#            plt.figure(3) #get the value of global_step                                                                        
#            plt.cla()
#            plt.plot(x_ref,y_ref)
#            #plt.plot(X_SAS,Y_SAS)
#            plt.pause(plt_pause)            
   ####################################################################################################                        
                    
    t.join()
    tiii.join()
