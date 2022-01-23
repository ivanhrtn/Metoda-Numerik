import numpy as np
import matplotlib.pyplot as plt

def f(t,H):
    return (-0.000548144*np.sqrt(H)/(3*H-H**2))

h = 500 #pilih lebar selang


#METODE HEUN
t_heun = 0 #waktu awal
H_heun = 2.75 #ketinggian awal air dalam tangki

listt_heun=[]
listh_heun=[]

while H_heun>0:
    H_predictor = H_heun + h*f(t_heun,H_heun)
    H_heun = H_heun + (h/2.0)*(f(t_heun,H_heun) + f(t_heun+h ,H_predictor))
    t_heun=t_heun+h
    listt_heun.append(t_heun)
    listh_heun.append(H_heun)   
plt.plot(listt_heun,listh_heun, '-o' ,color='red',label='Heun')
print ('Menggunakan Metode Heun diperlukan waktu',t_heun,'detik')


#METODE MIDPOINT
t_midpoint = 0 #waktu awal
H_midpoint = 2.75 #ketinggian awal air dalam tangki

listt_midpoint=[]
listh_midpoint=[]

while H_midpoint>0:
    H_midpoint = H_midpoint + h*f(t_midpoint + h/2, H_midpoint + (h/2.0)*f(t_midpoint,H_midpoint))
    t_midpoint=t_midpoint+h
    listt_midpoint.append(t_midpoint)
    listh_midpoint.append(H_midpoint)
    
plt.plot(listt_midpoint,listh_midpoint,'-*', color='green',label='Midpoint')
print ('Menggunakan Metode Midpoint diperlukan waktu',t_midpoint,'detik')


#METODE RUNGKE-KUTTA ORDE 4
t_rk = 0 #waktu awal
H_rk = 2.75 #ketinggian awal air dalam tangki

listt_rk=[]
listh_rk=[]

while H_rk>0:
    k1=f(t_rk,H_rk)
    k2=f(t_rk+0.5*h , H_rk+0.5*k1*h)
    k3=f(t_rk+0.5*h , H_rk+0.5*k2*h)
    k4=f(t_rk+h , H_rk+h*k3)
    
    H_rk=H_rk+h*(k1+2*k2+2*k3+k4)/6.0
    t_rk=t_rk+h
    listt_rk.append(t_rk)
    listh_rk.append(H_rk)
    
plt.plot(listt_rk,listh_rk, '-+',color='blue',label='RK Orde4')
plt.title('Grafik Ketinggian Air dalam Tangki (step size=%s)'%h)
plt.xlabel('Waktu (detik)')
plt.ylabel('Ketinggian Air (meter)')
plt.legend()
plt.show()
print ('Menggunakan Metode Runge-Kutta Orde 4 diperlukan waktu',t_rk,'detik')
