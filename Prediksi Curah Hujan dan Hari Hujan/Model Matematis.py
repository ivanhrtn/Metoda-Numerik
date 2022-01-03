# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel("C:\\Users\\nadya putri\\Downloads\\data hujan pisah tahun.xlsx", sheet_name = r'Sheet2')
curah = np.array(pd.DataFrame(data, columns = ['Curah Hujan']));C1=float(sum(curah))
hari = np.array(pd.DataFrame(data, columns = ['Hari Hujan']));H1=float(sum(hari))
n=len(curah);curah_bar=C1/n; hari_bar=H1/n

#koefisien korelasi
np.corrcoef(curah[:,0],hari[:,0])[1,0]
#PLOT CURAH HUJAN - HARI HUJAN
plt.plot(curah,hari,"o")
plt.ylabel('Hari Hujan')
plt.xlabel('Curah Hujan')
plt.show()
#plot interpolasi
plt.plot(curah,hari)
plt.ylabel('Hari Hujan')
plt.xlabel('Curah Hujan')
plt.show()

curah2=np.power(curah,2);C2=float(sum(curah2))
curah3=np.power(curah,3);C3=float(sum(curah3))
curah4=np.power(curah,4);C4=float(sum(curah4))

hari2=np.power(hari,2);H2=float(sum(hari2))
hari3=np.power(hari,3);H3=float(sum(hari3))
hari4=np.power(hari,4);H4=float(sum(hari4))

curah1hari1=np.multiply(curah,hari);C1H1=float(sum(curah1hari1))
curah2hari1=np.multiply(curah2,hari);C2H1=float(sum(curah2hari1))
curah1hari2=np.multiply(curah,hari2);C1H2=float(sum(curah1hari2))

#diketahui hari hujan, ditanya curah hujan
print('Misal: C=(a0)+(a1)H+(a2)H^2')
#persamaan normal
#n(a0)+(H1)(a1)+(H2)(a2)=C1
#(H1)(a0)+(H2)(a1)+(H3)(a2)=C1H1
#(H2)(a0)+(H3)(a1)+(H4)(a2)=C1H2
A = np.array([ [n,H1,H2], [H1,H2,H3], [H2,H3,H4] ])
B = np.array([C1,C1H1,C1H2])
C = np.linalg.solve(A,B)
a0=float(C[0:1]);a1=float(C[1:2]);a2=float(C[2:3])
print('diperoleh a0=',a0,', a1=',a1,', a2=',a2)
print("sehingga, persamaan regresi:y=",a0,"+",a1,"x+",a2,"x^2.")

plt.figure(figsize = (5,3))
plt.scatter(hari, curah, s = 2)
x=np.array(range(35))
y= -12.45241667702571 + 6.071935332799701*x+ 0.18580383261885566*(x**2)
plt.plot(x,y,color = 'yellow', label = 'y= -12.4524 + 6.07194 x+ 0.1858 x^2.')
plt.title('Garis Regresi Curah Hujan jika diketahui Hari Hujan')
plt.xlabel('Hari Hujan')
plt.ylabel('Curah Hujan')
plt.legend()
plt.show()

curah_regresi= -12.45241667702571 + 6.071935332799701*hari+ 0.18580383261885566*(hari**2)
st_curah=float(sum(np.power(np.subtract(curah,curah_bar),2)))
sr_curah=float(sum(np.power(np.subtract(curah,curah_regresi),2)))
r2_curah=float((st_curah-sr_curah)/st_curah)
print(r2_curah)

#diketahui curah hujan, ditanya hari hujan
print('Misal:H=(b0)+(b1)C+(b2)C^2')
#persamaan normal
#n(b0)+(C1)(b1)+(C2)(b2)=H1
#(C1)(b0)+(C2)(b1)+(C3)(b2)=C1H1
#(C2)(b0)+(C3)(b1)+(C4)(b2)=C2H1
D = np.array([ [n,C1,C2], [C1,C2,C3], [C2,C3,C4] ])
E = np.array([H1,C1H1,C2H1])
F = np.linalg.solve(D,E)
b0=float(F[0:1]);b1=float(F[1:2]);b2=float(F[2:3])
print('diperoleh b0=',b0,', b1=',b1,', b2=',b2)
print("sehingga, persamaan regresi:y=",b0,"+",b1,"x+",b2,"x^2.")

plt.figure(figsize = (5,3))
plt.scatter(curah, hari, s = 2, color = 'green')
s=np.array(range(500))
t=(8.305585988313624)+(0.09477151084750207)*s-0.00012692620157457154*(s**2)
plt.plot(s,t,color = 'r', label = 'y= 8.3056 + 0.0948 x -0.0001 x^2')
plt.title('Garis Regresi Hari Hujan jika diketahui Curah Hujan')
plt.xlabel('Curah Hujan')
plt.ylabel('Hari Hujan')
plt.legend()
plt.show()

hari_regresi= (8.305585988313624)+(0.09477151084750207)*curah-0.00012692620157457154*(curah**2)
st_hari=float(sum(np.power(np.subtract(hari,hari_bar),2)))
sr_hari=float(sum(np.power(np.subtract(hari,hari_regresi),2)))
r2_hari=float((st_hari-sr_hari)/st_hari)
print(r2_hari)

#mencari jumlah hari hujan bulan Oktober, November, Desember 8.305585988313624
x1=84.2
x2=270.7
x3=313.5
print('Hari hujan pada bulan Oktober tahun 2019 adalah',b2*np.square(x1)+b1*x1+b0)
print('Hari hujan pada bulan November tahun 2019 adalah',b2*np.square(x2)+b1*x2+b0)
print('Hari hujan pada bulan Desember tahun 2019 adalah',b2*np.square(x3)+b1*x3+b0)