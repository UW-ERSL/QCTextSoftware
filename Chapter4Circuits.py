# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:16:44 2024

@author: Krishnan Suresh; ksuresh@wisc.edu
"""
#%% Modules needed for all examples below
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from IPython.display import display
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit_aer import Aer
import numpy as np


#%% Hadamard Operator, statevector
circuit = QuantumCircuit(1) # 1 qubit
circuit.h(0) # apply H to qubit 0
psi = Statevector(circuit) #extract the state
display(psi.draw('latex')) # print

#%%  Sampling a circuit on a simulator
def simulateCircuit(circuit,nShots=1000):
	backend = Aer.get_backend('qasm_simulator')
	new_circuit = transpile(circuit, backend)
	job = backend.run(new_circuit,shots = nShots)
	counts = job.result().get_counts(circuit)
	return counts

#%% Hadamard Operator, measure
# 1 qubit, 1 classical bit
circuit = QuantumCircuit(1, 1) 
circuit.h(0) # apply H to qubit 0
# measure and place result in classical bit
circuit.measure(0, 0) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)

#%% RX Operator
#1 qubit, 1 classical bit
circuit = QuantumCircuit(1, 1)  
circuit.rx(np.pi/3,0) # apply Rx to qubit 0
#  measure and place result in classical bit
circuit.measure(0, 0) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)



#%% Operators in Sequence 
circuit = QuantumCircuit(1, 1)  
circuit.x(0) 
circuit.h(0) 
circuit.measure(0, 0) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)

#%% State preparation 
circuit = QuantumCircuit(1, 1)  
q = Statevector([np.sqrt(8)/3, (1j)/3]) 
circuit.prepare_state(q,0,'Prepare q')
circuit.x(0) 
circuit.measure(0, 0) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)


#%% Multi-qubit circuit
circuit = QuantumCircuit(3, 3)  
circuit.x(0)
circuit.id(1)
circuit.h(2)
circuit.measure([0,1,2], [0,1,2]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)



#%% Multi-qubit circuit, theoretical state
circuit = QuantumCircuit(3, 3)  
circuit.x(0)
circuit.id(1)
circuit.h(2)
psi = Statevector(circuit)
display(psi.draw('latex'))


#%% Unitary operator
circuit = QuantumCircuit(1, 1) 
UMatrix = 1/np.sqrt(2)*np.array([[1,1],[1j,-1j]]) 
circuit.unitary(UMatrix,0,'myU')
circuit.measure(0,0) 
# To see the theoretical state, comment the previous line and uncomment next 2 lines
# psi = Statevector(circuit)
# display(psi.draw('latex'))
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)

#%% Universal operator
def UniversalOperator(theta,phi,lambdaAngle):
	U = np.array([[np.cos(theta/2),-np.exp(1j*lambdaAngle)*np.sin(theta/2)],
			      [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lambdaAngle))*np.cos(theta/2)]])
	return U

U = UniversalOperator(np.pi/2,np.pi,np.pi)
print(U)
U = UniversalOperator(np.pi,0,0)
print(U)


#%% Simple CNOT
circuit = QuantumCircuit(2, 2)  
circuit.x(1) # try id(1), h(1)
circuit.cx(1,0)
circuit.measure([0,1], [0,1]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)


#%% Unitary Operator of a circuit
circuit = QuantumCircuit(2, 2)  
circuit.cx(1,0)
U = Operator(circuit)
print("U: \n", U.data)


#%% Simple CH
circuit = QuantumCircuit(2, 2)  
circuit.x(1) # try id(1), h(1)
circuit.ch(1,0)
circuit.measure([0,1], [0,1]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)

#%% CNOT and controlled Hadamard
circuit = QuantumCircuit(3, 3) 
circuit.y(0) 
circuit.rx(np.pi/3,1) 
circuit.h(2) 
circuit.barrier()
circuit.cx(2,0)
circuit.ch(2,1)
circuit.measure([0,1,2], [0,1,2]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,10000)
print('Counts:',counts)
plot_histogram(counts)


#%% Controlled Phase
circuit = QuantumCircuit(2, 2)  
circuit.x(0) 
circuit.x(1) 
circuit.cp(-np.pi/2,0,1)
psi = Statevector(circuit)
display(psi.draw('latex'))

circuit.measure([0,1], [0,1]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)


#%% Controlled Phase with Hadamard
circuit = QuantumCircuit(2, 2)  
circuit.h(1) 
circuit.cp(np.pi/2,0,1)
circuit.h(0) 
psi = Statevector(circuit)
display(psi.draw('latex'))

circuit.measure([0,1], [0,1]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)


#%% Simple swap  
circuit = QuantumCircuit(2, 2)  
circuit.x(1) 
circuit.swap(1,0)
circuit.measure([0,1], [0,1]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)


#%% controlled Unitary
circuit = QuantumCircuit(2, 2) 
UMatrix = 1/np.sqrt(2)*np.array([[1,1],[1j,-1j]]) 
U = UnitaryGate(UMatrix,'myU')
UControl = U.control(1)
circuit.append(UControl,[1,0])
psi = Statevector(circuit) #extract the state
display(psi.draw('latex')) # print
circuit.measure([0,1],[0,1]) 
circuit.draw('mpl') 


#%% state controlled U Gate
circuit = QuantumCircuit(4, 1) 
cu_gate = UGate(np.pi, 0, 0).control(3, ctrl_state = '101')
# Not sure why the following does not work
# circuit.cry(theta = np.pi/3,target_qubit= 0,
#      control_qubit = [1,2,3], ctrl_state = '101')
circuit.append(cu_gate,[1,2,3,0])
circuit.measure([0],[0]) 
circuit.draw('mpl') 

#%% Hadamard test
zeroQubit = QuantumRegister(1, '0')
phiQubit = QuantumRegister(1, '\u03D5')
cl = ClassicalRegister(1,'m')
circuit = QuantumCircuit(zeroQubit,phiQubit, cl) 
circuit.h(0)
#circuit.sdg(0)
UMatrix = 1/np.sqrt(2)*np.array([[1,1],[1j,-1j]]) 
U = UnitaryGate(UMatrix,'U')
UControl = U.control(1)
circuit.append(UControl,[0,1])
circuit.h(0)
circuit.measure(0,0) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)

#%% Inner product 
circuit = QuantumCircuit(2, 1) 
circuit.h(0)
a = np.sqrt(3)
b = 1/np.sqrt(2)
UMatrix = b*np.array([[(a+1)/2,(a-1)/2],[(a-1)/2,(-a-1)/2]])
U = UnitaryGate(UMatrix,'U')
UControl = U.control(1)
circuit.append(UControl,[0,1])
circuit.h(0)
circuit.measure(0,0) 
circuit.draw('mpl')
counts = simulateCircuit(circuit,100)
print('Counts:',counts)
