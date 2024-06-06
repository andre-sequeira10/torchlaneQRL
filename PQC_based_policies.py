import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import functools

def ansatz_jerbi(state, weights,input_scaling=None, n_qubits=2, n_layers=1, change_of_basis=False, entanglement="all2all"):
        if change_of_basis==True:
            for l in range(len(weights)):
                for i in range(n_qubits):
                    qml.Rot(*weights[l][i],wires=i)
                    #qml.RY(weights[l][i][0],wires=i)
                    #qml.RZ(weights[l][i][1],wires=i)
        else:          
            for l in range(len(weights)):
                for i in range(n_qubits):
                    qml.RZ(weights[l][i][0],wires=i)
                    qml.RY(weights[l][i][1],wires=i)
                    #qml.RZ(weights[l][i][2],wires=i)

                #if l < n_layers:
                if entanglement == "all2all":
                    for q1 in range(n_qubits-1):    
                        for q2 in range(q1+1, n_qubits): 
                            qml.CNOT(wires=[q1,q2])
                            #qml.CZ(wires=[q1,q2])

                if l < n_layers-1:
                    if l == n_layers-1:
                        if input_scaling is not None:
                            qml.AngleEmbedding(state*input_scaling[l][0], wires=range(n_qubits),rotation="Z")
                            qml.AngleEmbedding(state*input_scaling[l][1], wires=range(n_qubits),rotation="Y")
                        else:
                            qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Z")
                            qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Y")
                    else:
                        if input_scaling is not None:
                            qml.AngleEmbedding(state*input_scaling[l][0], wires=range(n_qubits),rotation="Y")   
                            qml.AngleEmbedding(state*input_scaling[l][1], wires=range(n_qubits),rotation="Z")
                        else:
                            qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Y")
                            qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Z")

def UQC(state, w, alpha, varphi, n_qubits=2, n_layers=1, entanglement="chain"):
    for l in range(n_layers):
        qml.broadcast(qml.RY, wires=range(n_qubits), pattern="single", parameters=2*varphi[l])
        for q in range(n_qubits):
            inner_p = torch.dot(state[0], w[l][q])
            r = 2*inner_p + 2*alpha[l][q]
            qml.RZ(r, wires=q)  
        qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern=entanglement)

# Define the policy network
class BornPolicy(nn.Module):
    """
    A policy network for a reinforcement learning agent using a variational quantum circuit.

    Args:
        - circuit (function): A quantum circuit function that takes as input a tensor of parameters and returns a quantum node that represents the circuit. "UQC" and "jerbi" are the available circuits.
        - n_actions (int): The number of possible actions that the policy can take.
        - n_qubits (int): The number of qubits in the quantum circuit.
        - n_layers (int): The number of layers in the quantum circuit.
        - reuploading (bool): Determines whether the input data is re-uploaded at each layer of the quantum circuit.
        - init (str): The method used to initialize the parameters of the quantum circuit. Can be "normal_0_1", "random_0_2pi", "glorot", "random_-1_1", "random_0_1", "random_-pi_pi", "zeros".
        measurement (str): The type of measurement to perform at the end of the quantum circuit. Can be "n-local", where n is the number of qubits to measure.
        - measurement_qubits (list): A list of qubits to measure. If `None`, the qubits are measured in an ascended way following the "n-local" measurement.
        - policy (str): The policy used to convert the output of the quantum circuit into a probability distribution over actions. Can be 
            - "global" - categorical distribution over global measurements of the projectors [a_1,a_2, ... a_|A|]
            - "mean-approx" - categorical distribution over the mean of the individual qubit projectors [a_1,a_2, ... a_|A|]
            - "product-approx" - categorical distribution over the product of the individual qubit projectors [a_1,a_2, ... a_|A|]
            - "parity" - parity function obtained from a global measurement on all qubits. Only works for |A|=2.
        - device (str): The quantum device to run the circuit on. Default is "default.qubit".
        - shots (int): The number of times to sample the quantum circuit. If `None`, expectation values are computed analytically.
        - diff_method (str): The differentiation method to use for computing gradients of the quantum circuit. Can be "parameter-shift", "finite-diff", "backprop". 
        TO DO -- "ajoint" differentiation only works with expval method from pennylane and not with probs. 
    """


    def __init__(self, circuit=None, n_actions=2, n_qubits=2, n_layers=1, reuploading=False, init="normal_0_1", measurement="n-local", measurement_qubits=None, policy="global", device="default.qubit", shots=None, diff_method="backprop" , feature_size=4, softmax_activation=None):
        super(BornPolicy, self).__init__()
        self.n_qubits=n_qubits
        self.n_actions=n_actions
        self.n_layers=n_layers
        self.measurement=measurement
        self.measurement_qubits=measurement_qubits
        self.init = init
        self.shots = shots
        self.device=qml.device(device, wires=self.n_qubits, shots=self.shots)
        self.reuploading=reuploading
        self.init = init
        self.diff_method = diff_method
        self.policy = policy
        self.feature_size = feature_size
        self.softmax_activation = softmax_activation
        self.circuit_label = circuit

        locality = self.measurement.split("-")[0]
        if locality == "n":
            locality = self.n_qubits
        else:
            locality = int(locality)

        if self.measurement_qubits is None:
            self.measurement_qubits = range(locality) 

        if self.policy == "mean-approx" or self.policy == "product-approx":
            self.measure = [qml.probs(wires=i) for i in self.measurement_qubits]
        else:
            self.measure = qml.probs(wires=self.measurement_qubits)

        if circuit == "jerbi":
            
            if self.reuploading:
                self.weight_shapes = {"weights":(self.n_layers, self.n_qubits, 2),"inpt_scaling":(self.n_layers,2,self.n_qubits)}

                def qcircuit(inputs, weights, inpt_scaling):
    
                    for q in range(n_qubits):
                        qml.Hadamard(wires=q)

                    ansatz_jerbi(inputs, weights,input_scaling=inpt_scaling, n_qubits=self.n_qubits, n_layers=self.n_layers)

                    if len(self.measurement_qubits) == 1 or self.policy == "global" or self.policy == "parity" or self.policy == "contiguous" or self.policy == "modulo":
                        return qml.apply(self.measure)
                    else:
                        return [qml.apply(i) for i in self.measure]
            else:
                self.weight_shapes = {"weights":(self.n_layers, self.n_qubits, 2)}
                def qcircuit(inputs, weights):
    
                    for q in range(n_qubits):
                        qml.Hadamard(wires=q)

                    ansatz_jerbi(inputs, weights, n_qubits=self.n_qubits, n_layers=self.n_layers)

                    if len(self.measurement_qubits) == 1 or self.policy == "global":
                        return qml.apply(self.measure)
                    else:
                        return [qml.apply(i) for i in self.measure]
        
            
            self.circuit = qml.QNode(qcircuit, self.device, diff_method=self.diff_method)

        elif circuit == "UQC":
            self.weight_shapes = {"w":(self.n_layers, self.n_qubits, self.feature_size), "alpha":(self.n_layers, self.n_qubits), "varphi":(self.n_layers, self.n_qubits)}
            def qcircuit(inputs, w, alpha, varphi):
                UQC(inputs, w, alpha, varphi, n_qubits=self.n_qubits, n_layers=self.n_layers)
                return [qml.probs(wires=self.measurement_qubits)]

            self.circuit = qml.QNode(qcircuit, self.device, diff_method=self.diff_method)
        else:
            self.circuit = circuit
        
        if self.init == "random_0_2pi":
             self.init_method = functools.partial(torch.nn.init.uniform_, a=0, b=2*np.pi)
        elif self.init == "glorot":
             self.init_method = functools.partial(torch.nn.init.normal_, mean=0.0, std=np.sqrt(3/4))
        elif self.init == "random_-1_1":
            self.init_method = functools.partial(torch.nn.init.uniform_, a=-1, b=1)
        elif self.init == "random_0_1":
            self.init_method = functools.partial(torch.nn.init.uniform_, a=0, b=1)
        elif self.init == "random_-pi_pi":
            self.init_method = functools.partial(torch.nn.init.uniform_, a=-np.pi, b=np.pi)
        elif self.init == "zeros":
            self.init_method = functools.partial(torch.nn.init.zeros_)
        elif self.init == "normal_0_1":
            self.init_method = functools.partial(torch.nn.init.normal_, mean=0.0, std=1)
            
        self.qlayer = qml.qnn.TorchLayer(self.circuit, self.weight_shapes, init_method = self.init_method)
            

    def get_measurements(self):
        return self.measure
    
    def set_weights(self, weights):
        self.qlayer.load_state_dict(weights)
    
    def get_weights(self):
        return self.qlayer.qnode_weights
    
    def get_meyer_wallach(self, x):
        
        inpt=x.detach().numpy()

        dev = qml.device("default.qubit", wires=self.n_qubits)
        
        #@qml.qnode(dev)
        if self.circuit_label == "jerbi":
            def meyer_wallach_circuit(inputs, weights, inpt_scaling=None, qubit=0):
        
                for q in range(self.n_qubits):
                    qml.Hadamard(wires=q)
                    
                ansatz_jerbi(inputs, weights,input_scaling=inpt_scaling, n_qubits=self.n_qubits, n_layers=self.n_layers)

                return qml.density_matrix([qubit])

            m_w_circuit = qml.QNode(meyer_wallach_circuit, dev, diff_method=self.diff_method)
            
            #weights copy
            weights = self.qlayer.qnode_weights["weights"].clone().detach().numpy()
            if self.reuploading:
                input_scaling = self.qlayer.qnode_weights["inpt_scaling"].clone().detach().numpy()
            else:
                input_scaling = None

            entanglement = 0
            for q in range(self.n_qubits):
                rho_i = m_w_circuit(inpt, weights, inpt_scaling=input_scaling, qubit=q)[0]
                entanglement += np.trace(np.matmul(rho_i,rho_i))
        
        elif self.circuit_label == "UQC":
            def meyer_wallach_circuit(inputs, w, alpha, varphi, qubit=0):
                UQC(inputs, w, alpha, varphi, n_qubits=self.n_qubits, n_layers=self.n_layers)
                return qml.density_matrix([qubit])

            m_w_circuit = qml.QNode(meyer_wallach_circuit, dev, diff_method=self.diff_method, interface="torch")
            
            #weights copy
            w = self.qlayer.qnode_weights["w"].clone().detach()#.numpy()
            alpha = self.qlayer.qnode_weights["alpha"].clone().detach()#.numpy()
            varphi = self.qlayer.qnode_weights["varphi"].clone().detach()#.numpy()

            entanglement = 0
            for q in range(self.n_qubits):
                rho_i = m_w_circuit(x, w, alpha, varphi, qubit=q).detach().numpy()
                entanglement += np.trace(np.matmul(rho_i,rho_i))

        entanglement /= self.n_qubits
        meyer_wallach = (2-2*entanglement).real
        
        return meyer_wallach

    def forward(self, x, temperature=None):


        if self.policy == "global":
            probs = self.qlayer(x)[0]
            action_probs = torch.zeros(self.n_actions)
            #if self.n_actions == 3:
                #action_probs[0] = probs[0]
                #action_probs[1] = probs[int((2**self.n_actions)/2)]
                #action_probs[2] = probs[-1]
            #else:    
            for a in range(self.n_actions):
                #if a==1:
                    #action_probs[a] += probs[-1]
                #else:
                    #action_probs[a] += probs[a]
                action_probs[a] += probs[a]
            action_probs /= torch.sum(action_probs)

        elif self.policy == "parity":

            probs = self.qlayer(x)[0]

            action_probs = torch.zeros(self.n_actions)
            for i in range(len(probs)):
                a=[]
                for m in range(int(np.log2(self.n_actions))):
                    if m==0:    
                        bitstring = np.binary_repr(i,width=self.n_qubits)
                    else:
                        bitstring = np.binary_repr(i,width=self.n_qubits)[:-m]
                    
                    a.append(bitstring.count("1") % 2)
                action_probs[int("".join(str(x) for x in a),2)] += probs[i]

        elif self.policy == "modulo":

            probs = self.qlayer(x)[0]

            action_probs = torch.zeros(self.n_actions)
            for i in range(2**self.n_qubits):

                #a = np.binary_repr(i, width=self.n_qubits).count("1") % self.n_actions
                a = i % self.n_actions
                action_probs[a] += probs[i]
                
        elif self.policy == "contiguous":

            probs = self.qlayer(x)[0]

            partitions = list(map(len, np.array_split(list(range(2**self.n_qubits)),self.n_actions)))
            indexes = torch.split(probs, partitions)
            action_probs = torch.stack([torch.sum(i) for i in indexes])
            action_probs /= torch.sum(action_probs)

        elif self.policy == "mean-approx":

            probs = self.qlayer(x)
            probs = torch.reshape(probs, (int((self.n_qubits*2)/2), 2))

            action_probs = torch.zeros(self.n_actions)

            for a in range(self.n_actions):
                #if a==1:
                    #a_bin = '1'*self.n_qubits
                #else:
                    #a_bin = np.binary_repr(a, width=self.n_qubits)
                a_bin = np.binary_repr(a, width=self.n_qubits)
                for i in range(self.n_qubits):
                    a_bin_i = int(a_bin[i])
                    action_probs[a] += probs[i][a_bin_i]

                action_probs[a] /= self.n_qubits

            action_probs /= torch.sum(action_probs)

        elif self.policy == "product-approx":

            probs = self.qlayer(x)
            probs = torch.reshape(probs, (int((self.n_qubits*2)/2), 2))

            action_probs = torch.ones(self.n_actions)

            for a in range(self.n_actions):
                #if a==1:
                    #a_bin = '1'*self.n_qubits
                #else:
                    #a_bin = np.binary_repr(a, width=self.n_qubits)
                a_bin = np.binary_repr(a, width=self.n_qubits)

                for i in range(self.n_qubits):
                    a_bin_i = int(a_bin[i])
                    action_probs[a] *= probs[i][a_bin_i]

            action_probs /= torch.sum(action_probs)

        if self.softmax_activation:
            if temperature:
                    action_probs = action_probs/temperature
                
                #nn functional softmax
            action_probs = torch.nn.functional.softmax(action_probs, dim=-1)

        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), action_probs



class PQCSoftmax(nn.Module):
    """
    A policy network for a reinforcement learning agent using a variational quantum circuit.

    Args:
        - circuit (function): A quantum circuit function that takes as input a tensor of parameters and returns a quantum node that represents the circuit. "UQC" and "jerbi" are the available circuits.
        - n_actions (int): The number of possible actions that the policy can take.
        - n_qubits (int): The number of qubits in the quantum circuit.
        - n_layers (int): The number of layers in the quantum circuit.
        - reuploading (bool): Determines whether the input data is re-uploaded at each layer of the quantum circuit.
        - init (str): The method used to initialize the parameters of the quantum circuit. Can be "normal_0_1", "random_0_2pi", "glorot", "random_-1_1", "random_0_1", "random_-pi_pi", "zeros".
        measurement (str): The type of measurement to perform at the end of the quantum circuit. Can be "n-local", where n is the number of qubits to measure.
        - measurement_qubits (list): A list of qubits to measure. If `None`, the qubits are measured in an ascended way following the "n-local" measurement.
        - policy (str): The policy used to convert the output of the quantum circuit into a probability distribution over actions. Can be 
            - "global" - categorical distribution over global measurements of the projectors [a_1,a_2, ... a_|A|]
            - "mean-approx" - categorical distribution over the mean of the individual qubit projectors [a_1,a_2, ... a_|A|]
            - "product-approx" - categorical distribution over the product of the individual qubit projectors [a_1,a_2, ... a_|A|]
            - "parity" - parity function obtained from a global measurement on all qubits. Only works for |A|=2.
        - device (str): The quantum device to run the circuit on. Default is "default.qubit".
        - shots (int): The number of times to sample the quantum circuit. If `None`, expectation values are computed analytically.
        - diff_method (str): The differentiation method to use for computing gradients of the quantum circuit. Can be "parameter-shift", "finite-diff", "backprop". 
        TO DO -- "ajoint" differentiation only works with expval method from pennylane and not with probs. 
    """


    def __init__(self, circuit=None, n_actions=2, n_qubits=2, n_layers=1, reuploading=False, init="normal_0_1", measurement="n-local", measurement_qubits=None, observables=None, device="default.qubit", shots=None, diff_method="backprop" , feature_size=4, temperature=None, output_scaling=None):
        super(PQCSoftmax, self).__init__()
        self.n_qubits=n_qubits
        self.n_actions=n_actions
        self.n_layers=n_layers
        self.measurement=measurement
        self.measurement_qubits=measurement_qubits
        self.init = init
        self.shots = shots
        self.device=qml.device(device, wires=self.n_qubits, shots=self.shots)
        self.reuploading=reuploading
        self.init = init
        self.diff_method = diff_method
        self.feature_size = feature_size
        self.observables = observables
        self.temperature = temperature
        self.output_scaling = output_scaling
        self.circuit_label = circuit
        
        if self.output_scaling is not None:
            self.output_scaling = nn.Parameter(torch.ones(self.n_actions))

        locality = self.measurement.split("-")[0]
        if locality == "n":
            locality = self.n_qubits
        else:
            locality = int(locality)

        if self.measurement_qubits is None:
            self.measurement_qubits = range(locality) 

        if self.observables is None:
            self.observables = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(self.n_qubits)])
        
        if circuit == "jerbi":
            
            if self.reuploading:
                self.weight_shapes = {"weights":(self.n_layers, self.n_qubits, 2),"inpt_scaling":(self.n_layers,2,self.n_qubits)}

                def qcircuit(inputs, weights, inpt_scaling):
    
                    for q in range(n_qubits):
                        qml.Hadamard(wires=q)

                    ansatz_jerbi(inputs, weights,input_scaling=inpt_scaling, n_qubits=self.n_qubits, n_layers=self.n_layers)

                    return [qml.expval(o) for o in self.observables]

            else:
                self.weight_shapes = {"weights":(self.n_layers, self.n_qubits, 2)}
                def qcircuit(inputs, weights):
    
                    for q in range(n_qubits):
                        qml.Hadamard(wires=q)

                    ansatz_jerbi(inputs, weights, n_qubits=self.n_qubits, n_layers=self.n_layers)

                    return [qml.expval(o) for o in self.observables]
                    
            
            self.circuit = qml.QNode(qcircuit, self.device, diff_method=self.diff_method)

        elif circuit == "UQC":
            self.weight_shapes = {"w":(self.n_layers, self.n_qubits, self.feature_size), "alpha":(self.n_layers, self.n_qubits), "varphi":(self.n_layers, self.n_qubits)}
            def qcircuit(inputs, w, alpha, varphi):
                UQC(inputs, w, alpha, varphi, n_qubits=self.n_qubits, n_layers=self.n_layers)
                return [qml.expval(o) for o in self.observables]

            self.circuit = qml.QNode(qcircuit, self.device, diff_method=self.diff_method)
        else:
            self.circuit = circuit
        
        if self.init == "random_0_2pi":
             self.init_method = functools.partial(torch.nn.init.uniform_, a=0, b=2*np.pi)
        elif self.init == "glorot":
             self.init_method = functools.partial(torch.nn.init.normal_, mean=0.0, std=np.sqrt(3/4))
        elif self.init == "random_-1_1":
            self.init_method = functools.partial(torch.nn.init.uniform_, a=-1, b=1)
        elif self.init == "random_0_1":
            self.init_method = functools.partial(torch.nn.init.uniform_, a=0, b=1)
        elif self.init == "random_-pi_pi":
            self.init_method = functools.partial(torch.nn.init.uniform_, a=-np.pi, b=np.pi)
        elif self.init == "zeros":
            self.init_method = functools.partial(torch.nn.init.zeros_)
        elif self.init == "normal_0_1":
            self.init_method = functools.partial(torch.nn.init.normal_, mean=0.0, std=1)
            
        self.qlayer = qml.qnn.TorchLayer(self.circuit, self.weight_shapes, init_method = self.init_method)
            

    def get_measurements(self):
        return self.measure
    
    def set_weights(self, weights):
        self.qlayer.load_state_dict(weights)
    
    def get_weights(self):
        return self.qlayer.qnode_weights
    
    def get_meyer_wallach(self, x):
        
        inpt=x.detach().numpy()

        dev = qml.device("default.qubit", wires=self.n_qubits)
        
        #@qml.qnode(dev)
        if self.circuit_label == "jerbi":
            def meyer_wallach_circuit(inputs, weights, inpt_scaling=None, qubit=0):
        
                for q in range(self.n_qubits):
                    qml.Hadamard(wires=q)
                    
                ansatz_jerbi(inputs, weights,input_scaling=inpt_scaling, n_qubits=self.n_qubits, n_layers=self.n_layers)

                return qml.density_matrix([qubit])

            m_w_circuit = qml.QNode(meyer_wallach_circuit, dev, diff_method=self.diff_method)
            
            #weights copy
            weights = self.qlayer.qnode_weights["weights"].clone().detach().numpy()
            if self.reuploading:
                input_scaling = self.qlayer.qnode_weights["inpt_scaling"].clone().detach().numpy()
            else:
                input_scaling = None

            entanglement = 0
            for q in range(self.n_qubits):
                rho_i = m_w_circuit(inpt, weights, inpt_scaling=input_scaling, qubit=q)[0]
                entanglement += np.trace(np.matmul(rho_i,rho_i))
        
        elif self.circuit_label == "UQC":
            def meyer_wallach_circuit(inputs, w, alpha, varphi, qubit=0):
                UQC(inputs, w, alpha, varphi, n_qubits=self.n_qubits, n_layers=self.n_layers)
                return qml.density_matrix([qubit])

            m_w_circuit = qml.QNode(meyer_wallach_circuit, dev, diff_method=self.diff_method, interface="torch")
            
            #weights copy
            w = self.qlayer.qnode_weights["w"].clone().detach()#.numpy()
            alpha = self.qlayer.qnode_weights["alpha"].clone().detach()#.numpy()
            varphi = self.qlayer.qnode_weights["varphi"].clone().detach()#.numpy()

            entanglement = 0
            for q in range(self.n_qubits):
                rho_i = m_w_circuit(x, w, alpha, varphi, qubit=q).detach().numpy()
                entanglement += np.trace(np.matmul(rho_i,rho_i))

        entanglement /= self.n_qubits
        meyer_wallach = (2-2*entanglement).real
        
        return meyer_wallach

    def forward(self, x, temperature=None):
            
            probs = self.qlayer(x)[0]
            #check if output scaling is needed
            if self.output_scaling is not None:
                probs = probs*self.output_scaling
            if temperature is not None:
                probs = probs/temperature
            
            #nn functional softmax
            probs = torch.nn.functional.softmax(probs, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action), probs



