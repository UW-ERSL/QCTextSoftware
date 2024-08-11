# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:16:44 2024

@author: Krishnan Suresh; ksuresh@wisc.edu
"""
#%% Qubits
from IPython.display import display
from qiskit.quantum_info import Statevector
import numpy as np

#%% Complex variables
x = 1 + 3j # note the 3j
print("The real part is: ", x.real)
print("The imaginary part is: ", x.imag)
print("The absolute value is: ", abs(x))



ket0 = Statevector([1, 0]) # define ket 0
ket1 = Statevector([0, 1]) # define ket 1

phi = 1/np.sqrt(2)*ket0 +   1j/np.sqrt(2)*ket1
display(phi.draw('latex'))

psi =  Statevector([1/np.sqrt(2), 1j/np.sqrt(2),0,0]) 
psi.draw('latex')

print(psi.is_valid())
#%% Using labels

ket0 = Statevector.from_label('0')# define ket 0
ket1 = Statevector.from_label('1') # define ket 1

phi = (1/2)*Statevector.from_label('00') + (np.sqrt(3)/2)*Statevector.from_label('11')
display(phi.draw('latex'))

#%%  Inner Product Qubits
psi =  Statevector([-1j/np.sqrt(2), 0,0, 1/np.sqrt(2)]) 
phi =  Statevector([-1j/np.sqrt(2), (np.sqrt(3)-1j)/4,0, 1/2]) 
print(psi.inner(phi))


#%%  Tensor Product Qubits

ket0 = Statevector([1, 0]) # define ket 0
ket1 = Statevector([0, 1]) # define ket 1

ket00 = ket0.tensor(ket0)
ket01 = ket0.tensor(ket1)
ket10 = ket1.tensor(ket0)
ket11 = ket1.tensor(ket1)

phi = 1/2*ket00 +   1j/np.sqrt(2)*ket10 +  (np.sqrt(3)+1j)/4*ket11

psi = phi.tensor(phi)
display(psi.draw('latex'))

