# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 05:23:29 2024

@author: Krishnan Suresh; ksuresh@wisc.edu
"""
#%%
# Imports for Qiskit
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile, assemble
from qiskit_aer import Aer
import random
import numpy as np
from scipy.optimize import minimize


#%% Pauli Expansion 2x 2
def PauliExpansion2x2(A):
	I = np.array([[1,0],[0,1]])
	X = np.array([[0,1],[1,0]])
	Y = np.array([[0,-1j],[1j,0]])
	Z = np.array([[1,0],[0,-1]])
	a = 4*[0]
	a[0] = np.trace(np.matmul(A,I))/2
	a[1] = np.trace(np.matmul(A,X))/2
	a[2] = np.trace(np.matmul(A,Y))/2
	a[3] = np.trace(np.matmul(A,Z))/2
	return a

A = np.array([[2,-1],[-1,2]])
a= PauliExpansion2x2(A)
print(a)

#%% Pauli Expansion 4 x 4
def PauliExpansion4x4(A):
	I = np.array([[1,0],[0,1]])
	X = np.array([[0,1],[1,0]])
	Y = np.array([[0,-1j],[1j,0]])
	Z = np.array([[1,0],[0,-1]])
	basis2x2 = [I,X,Y,Z]
	basis4x4 = 16*[None]
	count = 0
	for i in range(4):
		for j in range(4):
			basis4x4[count] = np.kron(basis2x2[i],basis2x2[j])
			count = count+1
	a = 16*[0]
	for i in range(16):
		a[i] = np.trace(np.matmul(A,basis4x4[i]))/4

	return a

A = np.array([[1,0,0,-0.5],[0,1,0,0],[0,0,1,0],[-0.5,0,0,1]])
a= PauliExpansion4x4(A)
print(a)


#%% Ansatz
def apply_fixed_ansatz(qubits, parameters):
    for iz in range (0, len(qubits)):
        circ.ry(parameters[0][iz], qubits[iz])
    circ.cz(qubits[0], qubits[1])
    circ.cz(qubits[2], qubits[0])
    for iz in range (0, len(qubits)):
        circ.ry(parameters[1][iz], qubits[iz])

    circ.cz(qubits[1], qubits[2])
    circ.cz(qubits[2], qubits[0])
    for iz in range (0, len(qubits)):
        circ.ry(parameters[2][iz], qubits[iz])

#%%
circ = QuantumCircuit(3)
parameters = np.random.rand(3,3)
apply_fixed_ansatz([0, 1, 2], parameters)
circ.draw('mpl')

#%% Creates the Hadamard test

def had_test(gate_type, qubits, auxiliary_index, parameters):
    circ.h(auxiliary_index)
    apply_fixed_ansatz(qubits, parameters)
    for ie in range (0, len(gate_type[0])):
        if (gate_type[0][ie] == 1):
            circ.cz(auxiliary_index, qubits[ie])
    for ie in range (0, len(gate_type[1])):
        if (gate_type[1][ie] == 1):
            circ.cz(auxiliary_index, qubits[ie])
    
    circ.h(auxiliary_index)
    
circ = QuantumCircuit(4)
had_test([ [0, 0, 0], [0, 0, 1] ], [1, 2, 3], 0, [ [1, 1, 1], [1, 1, 1], [1, 1, 1] ])
circ.draw('mpl')

#%%
# Creates controlled anstaz for calculating |<b|psi>|^2 with a Hadamard test

def control_fixed_ansatz(qubits, parameters, auxiliary, reg):

    for i in range (0, len(qubits)):
        circ.cry(parameters[0][i], qiskit.circuit.Qubit(reg, auxiliary), qiskit.circuit.Qubit(reg, qubits[i]))

    circ.ccx(auxiliary, qubits[1], 4)
    circ.cz(qubits[0], 4)
    circ.ccx(auxiliary, qubits[1], 4)

    circ.ccx(auxiliary, qubits[0], 4)
    circ.cz(qubits[2], 4)
    circ.ccx(auxiliary, qubits[0], 4)

    for i in range (0, len(qubits)):
        circ.cry(parameters[1][i], qiskit.circuit.Qubit(reg, auxiliary), qiskit.circuit.Qubit(reg, qubits[i]))

    circ.ccx(auxiliary, qubits[2], 4)
    circ.cz(qubits[1], 4)
    circ.ccx(auxiliary, qubits[2], 4)

    circ.ccx(auxiliary, qubits[0], 4)
    circ.cz(qubits[2], 4)
    circ.ccx(auxiliary, qubits[0], 4)

    for i in range (0, len(qubits)):
        circ.cry(parameters[2][i], qiskit.circuit.Qubit(reg, auxiliary), qiskit.circuit.Qubit(reg, qubits[i]))

q_reg = QuantumRegister(5)
circ = QuantumCircuit(q_reg)
control_fixed_ansatz([1, 2, 3], [ [1, 1, 1], [1, 1, 1], [1, 1, 1] ], 0, q_reg)
circ.draw('mpl')

#%%
def control_b(auxiliary, qubits):
    for ia in qubits:
        circ.ch(auxiliary, ia)

circ = QuantumCircuit(4)
control_b(0, [1, 2, 3])
circ.draw('mpl')

# Create the controlled Hadamard test, for calculating <psi|psi>

def special_had_test(gate_type, qubits, auxiliary_index, parameters, reg):
    circ.h(auxiliary_index)
    control_fixed_ansatz(qubits, parameters, auxiliary_index, reg)
    for ty in range (0, len(gate_type)):
        if (gate_type[ty] == 1):
            circ.cz(auxiliary_index, qubits[ty])

    control_b(auxiliary_index, qubits)
    circ.h(auxiliary_index)

q_reg = QuantumRegister(5)
circ = QuantumCircuit(q_reg)
special_had_test([ [0, 0, 0], [0, 0, 1] ], [1, 2, 3], 0, [ [1, 1, 1], [1, 1, 1], [1, 1, 1] ], q_reg)
circ.draw('mpl')

backend = Aer.get_backend('qasm_simulator')
# Implements the entire cost function on the quantum circuit

def calculate_cost_function(parameters):
    global opt
    global backend
    overall_sum_1 = 0
    parameters = [parameters[0:3], parameters[3:6], parameters[6:9] ]
    for i in range(0, len(gate_set)):
        for j in range(0, len(gate_set)):
            global circ
            qctl = QuantumRegister(5)
            qc = ClassicalRegister(5)
            circ = QuantumCircuit(qctl, qc)

            multiply = coefficient_set[i]*coefficient_set[j]
            had_test([gate_set[i], gate_set[j] ], [1, 2, 3], 0, parameters)

            circ.save_statevector()
            t_circ = transpile(circ, backend)
            qobj = assemble(t_circ)
            job = backend.run(qobj)

            result = job.result()
            outputstate = np.real(result.get_statevector(circ, decimals=100))
            o = outputstate

            m_sum = 0
            for l in range (0, len(o)):
                if (l%2 == 1):
                    n = o[l]**2
                    m_sum+=n

            overall_sum_1+=multiply*(1-(2*m_sum))

    overall_sum_2 = 0

    for i in range(0, len(gate_set)):
        for j in range(0, len(gate_set)):
            multiply = coefficient_set[i]*coefficient_set[j]
            mult = 1
            for extra in range(0, 2):
                qctl = QuantumRegister(5)
                qc = ClassicalRegister(5)
                circ = QuantumCircuit(qctl, qc)

                backend = Aer.get_backend('aer_simulator')

                if (extra == 0):
                    special_had_test(gate_set[i], [1, 2, 3], 0, parameters, qctl)
                if (extra == 1):
                    special_had_test(gate_set[j], [1, 2, 3], 0, parameters, qctl)

                circ.save_statevector()    
                t_circ = transpile(circ, backend)
                qobj = assemble(t_circ)
                job = backend.run(qobj)

                result = job.result()
                outputstate = np.real(result.get_statevector(circ, decimals=100))
                o = outputstate

                m_sum = 0
                for l in range (0, len(o)):
                    if (l%2 == 1):
                        n = o[l]**2
                        m_sum+=n
                mult = mult*(1-(2*m_sum))
            overall_sum_2+=multiply*mult
            
    print(1-float(overall_sum_2/overall_sum_1))
    return 1-float(overall_sum_2/overall_sum_1)




coefficient_set = [0.55, 0.45]
gate_set = [ [0, 0, 0], [0, 0, 1] ]

out = minimize(calculate_cost_function, x0=[float(random.randint(0,3000))/1000 for i in range(0, 9)], method="COBYLA", options={'maxiter':200})
print(out)

out_f = [out['x'][0:3], out['x'][3:6], out['x'][6:9] ]

circ = QuantumCircuit(3, 3)
apply_fixed_ansatz([0, 1, 2], out_f)
circ.save_statevector()

print('Running on ibm simulator')
t_circ = transpile(circ, backend)
qobj = assemble(t_circ)
job = backend.run(qobj)

result = job.result()
o = result.get_statevector(circ, decimals=10)

a1 = coefficient_set[1]*np.array([ [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,-1,0,0,0], [0,0,0,0,0,-1,0,0], [0,0,0,0,0,0,-1,0], [0,0,0,0,0,0,0,-1] ])
a2 = coefficient_set[0]*np.array([ [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1] ])
a3 = np.add(a1, a2)

b = np.array([float(1/np.sqrt(8)),float(1/np.sqrt(8)),float(1/np.sqrt(8)),float(1/np.sqrt(8)),float(1/np.sqrt(8)),float(1/np.sqrt(8)),float(1/np.sqrt(8)),float(1/np.sqrt(8))])

print((b.dot(a3.dot(o)/(np.linalg.norm(a3.dot(o)))))**2)