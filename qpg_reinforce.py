from statistics import mean
import gym

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import os 

import pennylane as qml 
from pennylane.templates import StronglyEntanglingLayers as SEL 
from pennylane.templates import BasicEntanglerLayers as BEL 
from pennylane.templates import IQPEmbedding
from pennylane.templates import AngleEmbedding
from pennylane import expval as expectation
from pennylane import PauliZ as Z 
from pennylane import PauliX as X 
from numpy import linalg 

from pennylane import numpy as np 
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torch.autograd.functional import hessian
import itertools
#import wandb
import argparse
from operator import itemgetter 
import copy

from PQC_based_policies import BornPolicy, PQCSoftmax

import json

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0) #cuda device
parser.add_argument('--policy', type=str, default="Q") #policy
parser.add_argument('--ng',type=int,default=0)
parser.add_argument('--linear', type=str, default=None) #neurons for linear layer
parser.add_argument('--hidden', type=str, default=None) #neurons for single hidden layer
parser.add_argument('--lr', type=float, default=0.1)  #learning rate
parser.add_argument('--episodes', type=int, default=1000) #number of episodes    
parser.add_argument('--gamma', type=float, default=0.99) #discount factor                                  
parser.add_argument('--init', type=str, default="normal_0_1") #discount factor                                  
parser.add_argument('--entanglement', type=str, default="all2all") #discount factor                                  
parser.add_argument('--n_layers', type=int, default=4) #discount factor  
parser.add_argument('--n_qubits', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=10) #discount factor                                  
parser.add_argument('--eigenvalue_filename', type=str, default="eigenvalue_cartpole") #discount factor                                  
parser.add_argument('--eigenvalue', type=int, default=0) #discount 
parser.add_argument('--save', type=int, default=0) #saver   
parser.add_argument('--kl_divergence', type=int, default=0)
parser.add_argument('--meyer_wallach_entanglement', type=int, default=0)
parser.add_argument('--data_reuploading', type=bool, default=False)
parser.add_argument('--environment', type=str, default='CartPole-v0')
parser.add_argument('--measurement', type=str, default='n-local')
parser.add_argument('--born_policy', type=str, default='global')
parser.add_argument('--torch_diff_method', type=str, default='backprop')
parser.add_argument('--comparator_policy', type=str, default='global')
parser.add_argument('--circuit', type=str, default="jerbi")
parser.add_argument('--temperature', type=int, default=0)
parser.add_argument('--softmax_activation', type=int, default=0)
parser.add_argument('--observables', type=str, required=False, help='<Required> Set flag')
parser.add_argument('--output_scaling', type=int, default=0)

args = parser.parse_args()

episodes=args.episodes
n_layers = args.n_layers
n_qubits = args.n_qubits
lr_q = args.lr
batch_size = args.batch_size
policy = args.policy
ng=args.ng
eigenvalue_filename = args.eigenvalue_filename
eigenvalue = args.eigenvalue
save = args.save
circuit = args.circuit
temperature = args.temperature
softmax_activation = args.softmax_activation
output_scaling = args.output_scaling

def parse_observables(observables_str):
    observables_list = json.loads(observables_str)
    return [qml.Hamiltonian(obs['coeffs'], [eval(op) for op in obs['ops']]) for obs in observables_list]

if args.observables == None:
    observables_softmax = None
else:
    observables_softmax = parse_observables(args.observables)

print("Initializing ... QFIM - {}".format(ng))
print("initializing ... ",args.init)
if args.linear == None:
    nn_linear=None
else:
    nn_linear=int(args.linear)

if args.hidden == None:
    nn_hidden=None
else:
    nn_hidden=int(args.hidden)

basis_change=False 
ent=args.entanglement
initialization=args.init
kl_divergence_ = args.kl_divergence
meyer_wallach_entanglement = args.meyer_wallach_entanglement
environment = args.environment
data_reuploading = args.data_reuploading
measurement = args.measurement
born_policy = args.born_policy
torch_diff_method = args.torch_diff_method
compp_policy = args.comparator_policy

def normalize(vector,env):
    if env == 'Acrobot-v1':
        c1=np.arccos(vector[0])
        c2=np.arccos(vector[2])

        vector = [c1,c2,vector[4],vector[5]]
    
    norm = np.max(np.abs(np.asarray(vector)))

    return vector/norm

def get_temperature(episode, num_episodes, initial_temp=1.0, final_temp=0.1):
    """
    Linear annealing of temperature from initial_temp to final_temp over num_episodes episodes.
    """
    return initial_temp + (final_temp - initial_temp) * (episode / num_episodes)


def discount_rewards(rewards, gamma=0.999):
    
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards

    return discounted_rewards
    
def reinforce(env, policy_estimator, num_episodes=600,
              batch_size=10, gamma=0.99, lr=0.01 ,ng=0, label=None, ent=False, kl_divergence=False, comparator_policy='global'):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    avg_rewards = []
    batch_actions = []
    batch_actions_tensor=[]
    batch_states = []
    batch_counter = 0
    eigen_total=[]
    meyer_wallach_avg = []
    kl_divergence_avg=[]
    tv_distance_avg=[]
    LEARNING_RATE = lr
    lr_qnpg = 0.02
    best_episode = []
    max_reward=0
   #for name,param in policy_estimator.named_parameters():
        #print(name,"\n")
        #print(param,"\n")

    if policy == "Q":
        #optimizer = optim.Adam(policy_estimator.parameters(),
           optimizer = optim.Adam(policy_estimator.parameters(),
                            lr=LEARNING_RATE,amsgrad=True)
                           #{"params": policy_estimator.beta, "lr": 0.3}],
                            #lr=LEARNING_RATE
    else:
        optimizer = optim.Adam(policy_estimator.parameters(),
                            lr=LEARNING_RATE)
   
    grads = []
    vars=[]

    import time 

    for ep in range(num_episodes):
        
        temperature = get_temperature(ep+1, num_episodes)

        s_0 = env.reset()   
        states = []
        max_reward=0
        rewards = []
        actions = []
        eigenvalues_ep = []
        log_actions = []
        complete = False
        meyer_wallach_ep = []
        kl_divergence_ep = []
        tv_distance_ep = []
        while complete == False:
            #action_probs = policy_estimator.forward(s_0).detach().numpy()
            s_0 = torch.from_numpy(normalize(s_0,environment)).float().unsqueeze(0)
            action, action_log_prob, pi = policy_estimator.forward(s_0, temperature=temperature)
            log_actions.append(action_log_prob)
            #action_probs_sampler = torch.clone(action_probs).detach().numpy()


            if ent:
                Q = policy_estimator.get_meyer_wallach(s_0)
                meyer_wallach_ep.append(Q)
            
            if kl_divergence:
                original_weights_dict = policy_estimator.get_weights()

                #global_policy = BornPolicy(circuit=None, n_actions=env.action_space.n, n_qubits=n_qubits, n_layers=n_layers, reuploading=data_reuploading, init=initialization, measurement=measurement, measurement_qubits=None, policy=comparator_policy, device="default.qubit", shots=None, diff_method=torch_diff_method)
                
                if comparator_policy == 'global' or comparator_policy == 'product-approx':
                    global_policy = BornPolicy(circuit=circuit, n_actions=env.action_space.n, n_qubits=n_qubits, n_layers=n_layers, reuploading=data_reuploading, init=initialization, measurement=measurement, measurement_qubits=None, policy=comparator_policy, device="default.qubit", shots=None, diff_method=torch_diff_method, feature_size=len(s_0),softmax_activation=softmax_activation, temperature=temperature)

                    #global_policy.set_weights(policy_estimator.get_weights())
                    global_policy.set_weights(original_weights_dict)
                elif comparator_policy == 'optimal':
                    global_policy = BornPolicy(circuit=circuit, n_actions=env.action_space.n, n_qubits=n_qubits, n_layers=n_layers, reuploading=data_reuploading, init=initialization, measurement=measurement, measurement_qubits=None, policy="product-approx", device="default.qubit", shots=None, diff_method=torch_diff_method, feature_size=len(s_0), softmax_activation=softmax_activation)

                    global_policy.load_state_dict(torch.load('optimal_policy_weights.pt'))

                action_global, action_log_prob_global, pi_global = global_policy.forward(s_0, temperature=temperature)

                pi_global = pi_global.detach().numpy()
                pi = pi.detach().numpy()

                kl_divergence_ep.append(np.sum(pi_global*np.log(pi_global/pi)))

                tv_distance_ep.append(0.5*np.sum(np.abs(pi_global-pi)))

            #action = np.random.choice([-1,0,1], p=action_probs_sampler)
            

            #Cartpole and Mountaincar
            s_1, r, complete, _ = env.step(action)
            #s_1, r, terminated, truncated, _ = env.step(action)
            #complete = terminated or truncated

            #rw = -(s_1[1] + np.sin(np.arcsin(s_1[1])+np.arcsin(s_1[3])))
            #Acrobot
            #s_1, r, complete, _ = env.step(action-1)
            
            states.append(s_0)
            
            rewards.append(r)
            actions.append(action)
            tmp = s_0
            s_0 = s_1

            if complete:
                meyer_wallach_avg.append(np.mean(meyer_wallach_ep))
                kl_divergence_avg.append(np.mean(kl_divergence_ep))
                tv_distance_avg.append(np.mean(tv_distance_ep))

                discounted_r = discount_rewards(rewards, gamma)
                batch_rewards.extend(discounted_r)
                avg_rewards.append(discounted_r)
                avg_rewards_2 = [sum(x) for x in itertools.zip_longest(*avg_rewards, fillvalue=0)]
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_actions_tensor.extend(log_actions) 
                batch_counter += 1
                total_rewards.append(sum(rewards))
                sum_rewards = sum(rewards)
                if sum_rewards >= max_reward:
                    max_reward = sum_rewards
                    best_episode = states
                #batch_avg_reward += sum(rewards)
                mean_r = np.mean(total_rewards[-10:])
                
                # If batch is complete, update network
                if (batch_counter == batch_size-1) and (np.mean(total_rewards[-50:]) < 195.0) :
                    t_init = time.time()
                    def closure():
                        optimizer.zero_grad(set_to_none=True)

                        #state_tensor = torch.FloatTensor(np.array(batch_states))
                        
                        lens = list(map(len, avg_rewards))
                        baseline = np.array(avg_rewards_2)
                        for ep in range(len(avg_rewards)):
                            for i in range(len(avg_rewards[ep])):
                                tam = 0 
                                for p in lens:
                                    if p >= i:
                                        tam+=1
                                avg_rewards[ep][i] -= baseline[i]/tam

                        batch_rewards = [] 
                        for ep in avg_rewards:
                            batch_rewards.extend(ep)

                        reward_tensor = torch.FloatTensor(np.array(batch_rewards))
                        action_tensor = torch.LongTensor(np.array(batch_actions))

                        #outs = policy_estimator.forward(state_tensor)
                        #logprob = torch.log(outs)
                        logprob = torch.stack(batch_actions_tensor)
                        #print("logprob ",logprob)
                        #entropy2 = outs.entropy()
                        selected_logprobs = torch.multiply(reward_tensor,logprob)#[np.arange(len(action_tensor)), action_tensor]
                        #print("selected logprob", selected_logprobs)
                        loss = -torch.mean(selected_logprobs)
                        #loss = loss / batch_size 
                        #print("mean - " , loss)

                        loss.backward()

                    
                    #if clip:
                        #torch.nn.utils.clip_grad_norm_(policy_estimator.parameters(), 2*np.pi)
                    optimizer.step(closure)

                    t_end = time.time()

                    print("TIME - ", t_end-t_init)
                    
                    #for name,param in policy_estimator.named_parameters():
                        #if name == "ws":
                            #print(name,"\n")
                            #print(param,"\n")
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_actions_tensor=[]
                    avg_rewards = []
                    batch_counter = 0
                    
                    for name, param in policy_estimator.named_parameters():
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad)
                            grad_var = torch.var(param.grad)
                    #grads_step = torch.cat(grads_step).pow(2).numpy().mean()
                    
                            grads.append(grad_norm)
                            vars.append(grad_var)
                    #wandb.log({"grads": grads[-1]})

                mean_r = np.mean(total_rewards[-10:])
                
                if (mean_r >= 190) and ng:
                    ng=0
                    optimizer = optim.Adam([
                           {"params": policy_estimator.qlayer.weights0, "lr": 0.01}])#, "momentum":0.9}])#,
                           #{"params": policy_estimator.beta, "lr": 0.3}],
                            #lr=LEARNING_RATE, 
                        #amsgrad=True)
                
                #wandb.log({"total_rewards": total_rewards[-1]})
                #wandb.log({"mean_rewards_10": mean_r})

                if eigenvalue:
                    eigenvalues_ep = np.array(eigenvalues_ep)
                    if np.iscomplexobj(eigenvalues_ep):
                        eigenvalues_ep = abs(eigenvalues_ep)
                    eigen_total.extend(eigenvalues_ep)
                '''
                #create list of occurences from eigenvalue list.
                eigen_counter = len(eigen_total)
                eigenvalue_occurences = np.unique(eigen_total, return_counts=True)
                data = []
                column_data = []
                numParams = sum(p.numel() for p in policy_estimator.parameters() if p.requires_grad)
                for i in range(len(eigenvalue_occurences[0])):
                    #column_data.append(str(eigenvalue_occurences[0][i]))
                    data.append([str(eigenvalue_occurences[0][i]),eigenvalue_occurences[1][i]])#/eigen_counter])
                
                #table = wandb.Table(data=[[i] for i in eigenvalues_ep])#, columns=["scores"])
                #wandb.log({"eigenvalue_dist": eigenvalues_ep})

                #table = wandb.Table(data=data, columns=["eigenvalue","counts"])
                table = wandb.Table(data=data, columns=["counts"])
                wandb.log({'my_hist': wandb.plot.histogram(table, "counts")})
                #wandb.log({'my_barchart': wandb.plot.bar(table, "eigenvalue", "counts")})
                #wandb.log({'my_barchart': wandb.plot.histogram(table, "counts")})
                '''
                # Optional
                #wandb.watch(policy_estimator)
            
                mean_meyer_wallach = meyer_wallach_avg[-1]
                mean_kl = kl_divergence_avg[-1]
                mean_tv = tv_distance_avg[-1]
                print("Ep: {} || Reward - {} || Entanglement - {} || KL - {} || TV - {}".format(
                    ep + 1, mean_r, mean_meyer_wallach, mean_kl, mean_tv))
                
                #if np.mean(total_rewards[-100:]) >= 195:
                    #print("done!")
                    #np.save("optimal_policy_weights.npy", policy_estimator.get_weights().detach().numpy())
                    #torch.save(policy_estimator.state_dict(), "optimal_policy_weights.pt")
                    #break
                #print("Meyer-Wallach entanglement mean - {}".format(mean_meyer_wallach))
    return total_rewards, grads, vars, eigen_total, meyer_wallach_avg, kl_divergence_avg, tv_distance_avg

env = gym.make(environment)
if environment == 'CartPole-v0':
    feature_size = env.observation_space.shape[0]
elif environment == 'Acrobot-v1':
    feature_size = 4

if born_policy == 'softmax':
    pe_q = PQCSoftmax(circuit=circuit, n_actions=env.action_space.n, n_qubits=n_qubits, n_layers=n_layers, reuploading=data_reuploading, init=initialization, measurement=measurement, measurement_qubits=None, device="default.qubit", shots=None, diff_method=torch_diff_method, feature_size=feature_size, observables=observables_softmax, temperature=temperature, output_scaling=output_scaling)
else:
    pe_q = BornPolicy(circuit=circuit, n_actions=env.action_space.n, n_qubits=n_qubits, n_layers=n_layers, reuploading=data_reuploading, init=initialization, measurement=measurement, measurement_qubits=None, policy=born_policy, device="default.qubit", shots=None, diff_method=torch_diff_method, feature_size=feature_size,softmax_activation=softmax_activation)


#model_q = torch.nn.DataParallel(pe_q)
print(kl_divergence_)

rewards_q , grads_q,vars,  eigenvalues, meyer_wallach_ent, kl_divergence, tv_distance = reinforce(env, pe_q , num_episodes=episodes, batch_size=batch_size, lr=lr_q, ng=ng, gamma=0.99,ent=meyer_wallach_entanglement, kl_divergence=kl_divergence_, comparator_policy=compp_policy)

if eigenvalue:  
    with open(eigenvalue_filename+'.npy', 'wb') as f:
        np.save(f, eigenvalues)

processid = os.getpid()

np.save("{}_{}_{}_n_qubits_{}_{}_is_{}_n_0_1_{}_rewards || {}.npy".format(environment,circuit,softmax_activation,n_qubits,born_policy,data_reuploading,observables_softmax,str(processid)), rewards_q)
np.save("{}_{}_{}_n_qubits_{}_{}_is_{}_n_0_1_{}_grads_norm || {}.npy".format(environment,circuit,softmax_activation,n_qubits,born_policy,data_reuploading,observables_softmax,str(processid)), grads_q)
np.save("{}_{}_{}_n_qubits_{}_{}_is_{}_n_0_1_{}_vars || {}.npy".format(environment,circuit,softmax_activation,n_qubits,born_policy,data_reuploading,observables_softmax,str(processid)), vars)
np.save("{}_{}_{}_n_qubits_{}_{}_is_{}_n_0_1_{}_meyer_wallach || {}.npy".format(environment,circuit,softmax_activation,n_qubits,born_policy,data_reuploading,observables_softmax,str(processid)), meyer_wallach_ent)
np.save("{}_{}_{}_n_qubits_{}_{}_is_{}_n_0_1_{}_kl_divergence_global || {}.npy".format(environment,circuit,softmax_activation,n_qubits,born_policy,data_reuploading,observables_softmax,str(processid)), kl_divergence)
np.save("{}_{}_{}_n_qubits_{}_{}_is_{}_n_0_1_{}_TV_distance_global || {}.npy".format(environment,circuit,softmax_activation,n_qubits,born_policy,data_reuploading,observables_softmax,str(processid)), tv_distance)

#np.save("cartpole_meyer_wallach"+policy+"_"+str(init)+"_"+str(processid)+".npy", meyer_wallach_ent)
'''
for i in range(10):
    s0 = env.reset()
    complete = False
    while not complete:
        #action_probs = pe_q.forward(s0).detach().numpy()
        action, action_log_prob = pe_q.forward(s0)

                #action = np.random.choice(action_space, p=action_probs)
        #action = np.random.choice([-1,0,1], p=action_probs)
        s_1, r, complete, _ = env.step(action)
        env.render()
        s0 = s_1
'''