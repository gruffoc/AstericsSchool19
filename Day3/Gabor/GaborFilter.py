import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.signal import convolve2d
from PIL import Image
# %matplotlib inline 

def GetImage(I, **kwargs):
    plt.figure();
    plt.axis('off')
    plt.imshow(I,cmap=plt.gray(),**kwargs)
    

#"Generate sinusoid stimuli: I(x)=Acos(wx+p)"
#
#def Sinusoid(L,A,omega,rho):
#    #L e` la dimensione dell'immagine generata
#    #ora generiamo la griglia sinusoidale
#    radius=int(L/2.0)
#    [x,y]=np.meshgrid(range(-radius,radius+1),range(-radius,radius+1))
#    
#    func=A*np.cos(omega[0]*x+omega[1]*y+rho)
#    return func
#
#theta = np.pi/2 #orizzontale
#theta2 = 0  #verticale
#theta3=np.pi/4 # obliquo
#omega = [np.cos(theta3), np.sin(theta3)]
#sinusoidParam = {'A':1, 'omega':omega, 'rho':np.pi/2, 'L':(32)}

#GetImage(Sinusoid(**sinusoidParam))

"Generate Gabor Filter"

"ψ(x;ω,θ,K)=[(ω^2/4πK^2)exp{−(ω^2/8K^2)[4(x⋅(cosθ,sinθ))^2+(x⋅(−sinθ,cosθ))^2]}]×[exp{iwx⋅(cosθ,sinθ)}exp(K^2/2)]"

def Gabor(L,omega,theta,func=np.cos,K=np.pi):
    radius = (int(L[0]/2.0), int (L[1]/2.0))
    [x,y] = np.meshgrid(range(-radius[0],radius[0]+1),range(-radius[1],radius[1]+1))
    
    a = x * np.cos(theta) + y * np.sin(theta)
    b = -x * np.sin(theta) +y * np.cos(theta)
    
    gauss = omega**2/(4*np.pi*K**2)*np.exp(-omega**2/(8*K**2)*(4*a**2 + b**2))
    sinusoid = func(omega * a) * np.exp(K**2/2)
    
    gabor = gauss * sinusoid
    return gabor

orizzLine = np.pi/2
vertLine = 0
orientamento=[orizzLine,vertLine]
L = (63,63)
omega = 0.6

applFilterOrizz = Gabor(L,omega,orientamento[0],func=np.sin)
applFilterVert = Gabor(L,omega,orientamento[1],func=np.sin)


def conv(A, B):
    ft_A = np.fft.fft2(A)
    ft_B = np.fft.fft2(B)
    conv = ft_A * ft_B
    img = np.fft.ifftshift(np.fft.ifft2(conv))   
    return np.real(img)


"Apply to Letter of Alphabet"
    
im = Image.open('alph.jpeg').convert("L")
I = np.asarray(im)
Alphabet = np.ndarray([0])

"prima riga"
a = 0
b = 63
c = 0
d = 63
lung = 63

for i in range(1,9):
    lett = I[ a:b, c:d ]
    Alphabet = np.append(Alphabet,lett)    
    #print (Alphabet)
    c = d 
    d = c+ lung
 
"seconda riga"
a = a + lung
b = a +lung
c = 0
d = 63

lung = 63    
for i in range(1,9):
    lett = I[a:b, c:d]
    Alphabet = np.append(Alphabet,lett)    
    c = d
    d = c + lung 

"terza riga"    
a = a + lung 
b = a +lung
c = 0
d = 63

lung = 63    
for i in range(1,9):
    lett = I[a:b, c:d]
    Alphabet = np.append(Alphabet,lett)    
    c = d
    d = c + lung 

"quarta riga"   
a = a + lung
b = a +lung
c = 0
d = 63

lung = 63    
for i in range(1,3):
    lett = I[a:b, c:d]
    Alphabet = np.append(Alphabet,lett)    
    c = d
    d = c + lung 

Alphabet = Alphabet.reshape(26,63,63)


for i in range(0,26):
    
    res = conv(Alphabet[i], applFilterOrizz)
    GetImage(res)






