import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import unicodedata

#NOMOR 1
#1a: METODA KUADRATUR GAUSS KOMPOSIT
print("NOMOR 1")
print("\n")
print("Akan dihitung nilai p(s) dengan Kuadratur Gauss Komposit")
#Input
S = float(input("Masukkan batas integral s yang diinginkan:"))
M = int(input("Masukkan jumlah panel yang diinginkan:"))
N = int(input("Masukkan jumlah titik pembobotan yang diinginkan:"))

a = -S
b = S

def f(z):
    return math.exp(-z**2/2)

#Gauss Kuadratur
def fgauss(a,b,z):
    return math.exp(-(((b-a)*z+b+a)/2)**2/2)*(b-a)/2

def gauss(a,b,N,M):
    qgauss = np.linspace(a,b,M+1)
    [z,c] = np.polynomial.legendre.leggauss(N)
    I=0
    for i in range(M):
        a = qgauss[i]
        b = qgauss[i+1]
        for j in range(len(c)):
            I=I+c[j]*fgauss(a, b, z[j])
    return I
I = gauss(a, b, N, M)
hasilgauss = I/((2*math.pi)**(0.5))
print("Diperoleh nilai p(s) dengan Integral Kuadratur Gauss Komposit sebesar:",hasilgauss)
# plt.plot([2,31],[hasilgauss,hasilgauss],label="Kuadratur Gauss",color="red")
print ('-'*80)


#1b: ATURAN TRAPESIUM
print("Akan dihitung ke-30 nilai integral pada soal dengan Aturan Trapesium")

def ftrap(x) :
    return np.exp(-(x**2)/2) 
s = float(input("Silakan Anda masukkan nilai batas s: "))
a = -s
b = s

tabelat = pd.DataFrame(data=None, columns=("Banyak titik","Hasil integral dengan Aturan trapesium"))

hasiltrap = (((b-a)/2)*(ftrap(a)+ftrap(b)))*(1/(np.pi*2)**(1/2))
tabelat.loc[1] = [2,hasiltrap]

for n in range(2,32):
    h = (b-a)/n
    sum = 0
    for i in range (1,n):
        x = a+i*h
        sum = sum + ftrap(x)
        
    at = h*(ftrap(a)+2*sum+ftrap(b))/2

    hasiltrap = (1/(math.sqrt(2*math.pi)))*at
      
    tabelat.loc[n] = [n+1,hasiltrap]
print(tabelat)

#Proses mengubah output menjadi array
fungsifull=lambda z: np.exp(-z**2/2)/np.sqrt(2*np.pi)
def trap(f,N,p,q):
    sumbuawal=np.zeros([N,2])
    sumbuawal[:,0]=np.linspace(p,q,N)
    sumbuawal[:,1]=f(sumbuawal[:,0])
    hasiltrap=0
    for j in range(0,N):
        if j==0 or j==(N-1):
            hasiltrap += sumbuawal[j,1]
        else:
            hasiltrap += 2*sumbuawal[j,1]
    hasiltrap=hasiltrap*(q-p)/(2*(N-1))
    return hasiltrap

vektorhasiltrap=np.zeros([30,1])
for k in range(2,32,1):
    i=int(k-2)
    vektorhasiltrap[i,0]=trap(fungsifull,k,a,b)

#GALAT ATURAN TRAPESIUM
print ('-'*80)
eror1=(np.abs((hasilgauss-vektorhasiltrap)/hasilgauss))*100
print("Berikut Galat Relatif nya (dalam persentase) dari Aturan Trapesium untuk banyak titik =2,3,4,...,31:")
print(eror1)
print ('-'*80)
    
#1c: ATURAN SIMPSON
print("Akan dihitung ke-15 nilai integral pada soal dengan Aturan Simpson")
tabelsim = pd.DataFrame(data=None, columns=("Banyak titik","Hasil integral dengan Aturan Simpson"))

def fsimp1 (x0,xn,n):
    h = (xn-x0)/(n-1)
    integral = f(x0)+f(xn)
    
    for i in range (1,n-1):
        k = x0+i*h
        if i%2==0: 
            integral = integral +2*f(k) #untuk genap
        else:
            integral = integral +4*f(k) #untuk ganjil
            
    integral = integral*h/3
    return integral

for n in range(3,33,2):
    result = fsimp1(a,b,n)
    hasilsimp1 = (1/(math.sqrt(2*math.pi)))*result
    
    tabelsim.loc[n]=[n,hasilsimp1]
print(tabelsim)

#Proses mengubah output menjadi array
fungsifull=lambda z: np.exp(-z**2/2)/np.sqrt(2*np.pi)
def simp1(f,N,p,q):
    sumbuawal=np.zeros([N,2])
    sumbuawal[:,0]=np.linspace(p,q,N)
    sumbuawal[:,1]=f(sumbuawal[:,0])
    hasilsimp=0
    for j in range(0,N):
        if j==0 or j==(N-1):
            hasilsimp += sumbuawal[j,1]
        elif j%2==1:
            hasilsimp += 4*sumbuawal[j,1]
        elif j%2==0:
            hasilsimp += 2*sumbuawal[j,1]
    return hasilsimp*(q-p)/(3*(N-1))

vektorhasilsimp=np.zeros([15,1])
for k in range(3,32,2):
    i=int((k-3)/2)
    vektorhasilsimp[i,0]=simp1(fungsifull,k,a,b)

#GALAT ATURAN SIMPSON
print ('-'*80)
eror2=(np.abs((hasilgauss-vektorhasilsimp)/hasilgauss))*100
print("Berikut Galat Relatif nya (dalam persentase) dari Aturan Simpson untuk banyak titik =3,5,7,...,31:")
print(eror2)


#GRAFIK GABUNGAN DARI 1a,1b,1c
plt.axhline(y=hasilgauss,color='black',linestyle="-")
plt.plot(range(2,32,1),vektorhasiltrap,'r.',range(3,32,2),vektorhasilsimp[:,0],'g.')
plt.legend(["Metode Kuadratur Gauss Komposit","Aturan Trapesium komposit","Aturan Simpson Komposit"])

print ('-'*80)

#NOMOR 2
print("NOMOR 2")
print("\n")
#ATURAN TITIK TENGAH
print("Akan dihitung aproksimasi integral secara numerik dari I dengan menggunakan Aturan Titik Tengah")
print("\n")
print("Berikut adalah perhitungan integral elemen kiri, yaitu integral 0 sampai b dari 2/(1+x^2) dx")
def tiktengkiri(f,a,b,n1):
    h=float(b-a)/n1
    result=0
    for i in range(n1):
        result += f((a+h/2)+i*h)
    result *= h
    return result

def v(x):
    return  2/(1+(x**2))
n1=int(input("Masukkan banyaknya selang untuk elemen kiri:"))
print("Sesuai dengan soal, kita tetapkan batas bawah dari elemen kiri=0")
a=0
b=int(input("Masukkan batas atas dari elemen kiri (yaitu b):"))
hasilkiri=tiktengkiri(v,a,b,n1)
print('Hasil integral numerik dari elemen kiri adalah',hasilkiri)

print("Berikut adalah perhitungan integral elemen kanan, yaitu integral 0 sampai 1/b dari 2/(1+t^2) dt")
def tiktengkanan(g,c,d,n2):
    h=float(d-c)/n2
    result=0
    for i in range(n2):
        result += g((c+h/2)+i*h)
    result *= h
    return result

def v(t):
    return  2/(1+(t**2))
n2=int(input("Masukkan banyaknya selang untuk elemen kanan:"))
print("Karena batas atas dari elemen kiri=",b,"maka batas atas dari elemen kanan adalah",1/b)
c=1/math.inf
d=1/b
hasilkanan=tiktengkanan(v,c,d,n2)
print('Hasil integral numerik dari elemen kanan adalah=',hasilkanan)
print("\n")

total=hasilkiri+hasilkanan
print("Jadi, arproksimasi integral secara numerik dari I dengan menggunakan Aturan Titik Tengah adalah sebesar",total)
print("\n")

#GALAT METODA TITIK TENGAH
print("Diketahui bahwa nilai sebenarnya dari integral nomor 2 adalah:"  + unicodedata.lookup("GREEK SMALL LETTER PI"))
print("Sekarang dihitung galatnya")
epsilon_t=(abs((math.pi-total)/math.pi))*100
print("Diperoleh Galat Relatif nya (dalam persentase) sebesar",epsilon_t)
print ('-'*80)

#ATURAN SIMPSON 1/3
from fractions import Fraction
print("Akan dihitung aproksimasi numerik menggunakan metode aturan simpson 1/3 komposit\n")
print("Berikut adalah perhitungan integral elemen kiri, yaitu integral 0 sampai b dari 2/(1+x^2) dx")
def fsimpkiri(x):
    return(2/(1+x**2))

n = int(input("Masukkan banyaknya selang:"))
a = 0
b = int(input("Masukkan nilai b:"))
h = (b - a) / n
x = np.linspace(a, b, n+1)

sum_genap = 0.0
sum_ganjil = 0.0
for i in range(1,n):
    if np.mod(i,2) == 0:
        sum_genap = sum_genap + 2*fsimpkiri(x[i])
    else:
        sum_ganjil = sum_ganjil + 4*fsimpkiri(x[i])
        
hasil1=(round((h/3)*(fsimpkiri(a)+sum_ganjil+sum_genap+fsimpkiri(b)), 4))

print("Hasil integral numerik dari elemen kiri =",hasil1)
print("\n")
print("\n")
print("Berikut adalah perhitungan integral elemen kanan, yaitu integral 0 sampai 1/b dari 2/(1+t^2) dt\n")
def fsimpkanan(t):
    return(2/(1+t**2))

a = 0
print("Karena batas atas dari elemen kiri =",b,",maka batas atas dari elemen kanan =",(Fraction(1/b)))
print("\n")
h = ((1/b) - a) / n
t = np.linspace(a, (1/b), n+1)

sum_genap = 0.0
sum_ganjil = 0.0
for i in range(1,n):
    if np.mod(i,2) == 0:
        sum_genap = sum_genap + 2*fsimpkanan(t[i])
    else:
        sum_ganjil = sum_ganjil + 4*fsimpkanan(t[i])
        
hasil2=(round((h/3)*(fsimpkanan(a)+sum_ganjil+sum_genap+fsimpkanan(1/b)), 4))

print("Hasil integral numerik dari elemen kanan =",hasil2)
print("\n")
hasiltotal=(round(hasil1+hasil2, 4))   
print("Hasil integral numerik dengan metoda simpson 1/3 =",hasiltotal)
print("\n")
error=abs(((np.pi - (hasil1+hasil2))/np.pi)*100)
print("Persentase Galat Relatif yang dihasilkan dari integral numerik =", error,"%")
