"""
Lead: Dr. Martin Kaloev as Independent scientist

paper:
Kaloev, M., Krastev, G. (2024). 

Exploring Advantage Estimation with Higher-Order Summation of State Values in Deep Reinforcement Learning: 
Experimental Insights for Online Temporal Difference Approach. 

In: Silhavy, R., Silhavy, P. (eds) Artificial Intelligence Algorithm Design for Systems. CSOC 2024. 
Lecture Notes in Networks and Systems, 
vol 1120. Springer, Cham. https://doi.org/10.1007/978-3-031-70518-2_16 
https://doi.org/10.1007/978-3-031-70518-2_16
======= 
Abstract:
This study investigates the potential advantages of redefining advantage calculation 
in actor-critic deep reinforcement learning. Departing from the conventional temporal 
difference approach, where the advantage is determined by the difference between the 
values of two consecutive states, we investigate an alternative method. Our approach 
involves computing the advantage as the difference between the value of the current 
state and the summation of multiple discounted values of subsequent states. Through a 
series of experiments, we explore the implications of this alternative advantage 
calculation on the training dynamics of actor-critic networks. Furthermore, we 
extend the actor’s loss function by incorporating the summation of even multiple 
discounted rewards with appropriate gamma discounts. The experiments aim to discern 
whether this extended loss function enhances the learning capabilities of the actor in 
complex reinforcement learning tasks. We find that the advantages are contingent upon the 
task’s reward horizon—indicating the frequency of rewards and the presence of negative 
rewards. Notably, the effectiveness of extended discounting is highly dependent on
selecting an appropriate order of calculation, signifying the number of state values 
considered after initial. This order, akin to context, proves crucial in determining the 
efficacy of the extended discounting approach.

code summuray:
This code illustrates high-order learning in temporal difference (TD) deep reinforcement learning (DRL), 
building on the concepts discussed in the referenced paper. It aims to test whether DRL performance can be improved 
by increasing the context—considering more detailed information from prior states—and using higher-order 
calculations to refine decision-making. The implementation emphasizes how expanded state representations and 
advanced temporal dependencies influence learning efficiency and policy optimization.

-> #Key line of interest <- , marked in the code, highlight the implementation of formulas derived from the paper, 
demonstrating their practical application in a computational framework.

Line of interest:
[162] Bottleneck, forcing computation to go into the CPU instead of the GPU to more clearly measure the time needed for extra recursions.
[177] Storage for memory, since this type of DRL is a combination of online and offline learning.
[226] Formula for high-order calculations, where the advantage is calculated from the value of the current state and the line of discounted summation of previous state values.
[240] LA, paper, page 182.
[256] Formula for high-order calculations, plus a chain of summation of discounted rewards.
[263, 270] LAR, paper, page 182.
[318] Control for LA.
[320] Control for LAR.

Read the paper because:
1. It explains more in-depth concepts and has a discussion on results.
2. It shows 3D graphs with multiple simulations and a higher content of data. 
3. In-depth explanation of the experimental process.

More papers for optimizing DRL:
Comparative Analysis of Activation Functions Used in the Hidden Layers of Deep Neural Networks
DOI: 10.1109/HORA52670.2021.9461312

Experiments Focused on Exploration in Deep Reinforcement Learning
DOI: 10.1109/ISMSIT52890.2021.9604690

Comprehensive Review of Benefits from the Use of Neuron Connection Pruning Techniques During the Training Process of Artificial Neural Networks in Reinforcement Learning: Experimental Simulations in Atari Games
DOI: 10.1109/ISMSIT58785.2023.10304968

Comprehensive Review of Benefits from the Use of Sparse Updates Techniques in Reinforcement Learning: Experimental Simulations in Complex Action Space Environments
DOI: 10.1109/CIEES58940.2023.10378830

paper: Tailored Learning Rates for Reinforcement Learning: A Visual Exploration and Guideline Formulation
doi: 10.1109/ISAS60782.2023.10391644

My New DRL algorithm 

paper: Martin Kaloev. Introducing a Novel DRL Algorithm: Condensing Thousands of Episodes into Few with Sequential Multi-ANN Agent for Accelerated Learning. TechRxiv. November 06, 2024.
DOI: 10.36227/techrxiv.173091755.56216751/v1

Cite:
If this paper and accompanying code contribute to the development of valuable real-world technologies, kindly consider citing this work in your references.
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import time
import imageio
import sys
from collections import deque

print(f'python version {sys.version}')
print(f'gym version: {gym.__version__}')
print(f'torch version: {torch.__version__}')
print(f'matplotlib version: {matplotlib.__version__}')
print(f'imageio version: {imageio.__version__}')

if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("No GPU found. Using CPU.")

# Define the neural net for the actor-critic
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size, sizer=3):
        super(ActorCritic, self).__init__()
        self.sizer=sizer
        self.fc1 = nn.Linear(input_size, (98*self.sizer)) #196
        self.drop1 = nn.Dropout(p=0.16)
        self.fc2 = nn.Linear(98*self.sizer, 64)
        self.drop2 = nn.Dropout(p=0.16)
        self.fc3 = nn.Linear(64, 128) #128
        self.actor = nn.Linear(128, output_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.relu(self.fc3(x))
        
        actor_output = self.actor(x)
        critic_output = self.critic(x)
        return actor_output, critic_output

    def actor_output_size(self):
        # Return the output size of the actor
        return self.actor.out_features
    
    def num_connections(self):
        # Calculate and print the number of connections in the neural network
        total_connections = 0
        for i, layer in enumerate([self.fc1, self.fc2, self.fc3, self.actor, self.critic]):
            if isinstance(layer, nn.Linear):
                connections = layer.in_features * layer.out_features
                total_connections += connections
                print(f"Layer {i + 1}: {layer.in_features} (input) x {layer.out_features} (output) = {connections} connections")
        print(f"Total connections: {total_connections}")
        return total_connections

# Define the A2C agent
class A2C():
    def __init__(self, env, gamma, fine_rl=0.000025, rough_rl=0.25, sizer=1, label="ctr", chain_order=0, chain_order_rewards=0):
        self.env = env ; control='empty'
        self.lr = 0.000025 # 0.000025
        self.gamma = gamma
        self.cn_g=control
        self.label=label
        self.actor_critic = ActorCritic(env.observation_space.shape[0], env.action_space.n, sizer=sizer)
        self.chain_order=chain_order
        self.chain_order_rewards=chain_order_rewards
        #use GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu") #line of interest #need to specificaly be longer compution time to highligy extra recursion on the high order rewards ; do change to use gpu in next version if needed. 
        self.actor_critic.to(self.device)
        for param in self.actor_critic.parameters():
            print(param.device)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        self.optimizer_fine = optim.Adam(self.actor_critic.parameters(), lr=fine_rl)
        self.optimizer_rough = optim.Adam(self.actor_critic.parameters(), lr=rough_rl)

        connections = self.actor_critic.num_connections()
        actor_output_size = self.actor_critic.actor_output_size()
        env_name = env.unwrapped.spec.id

        print(f"Currently running Gym environment: {env_name}, Number of connections: {connections} , Actor output size: {actor_output_size}, label: {self.label}, control drop: {self.cn_g}")
    
    #high order collections #line of interest 177 # online to ofline memory holding
    def set_order_memory_size(self, order_size):
        self.order_size_dq=deque(maxlen=order_size)
        self.reward_size_dq=deque(maxlen=order_size)
        self.high_order_dq=deque(maxlen=order_size)
        self.action_dq=deque(maxlen=order_size)

    #fix chose action stuf #it is legacy code to be cleaned some times in future 
    def choose_action(self, state):
        
        logits, _ = self.actor_critic(state)
        dist = torch.distributions.Categorical(logits=logits)
        
        dist = Categorical(F.softmax(logits, dim=-1))
        
        if logits.max() > 0.5:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        action = logits.argmax(dim=-1) # fixing exploration on this one
        return action.item()
    
    #fix custom loss updater #standart loss function # is not used here # legacy code 
    def compute_loss(self, state, action, reward, next_state, done, acts):
        
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        
        _, critic_output = self.actor_critic(state)
        _, next_critic_output = self.actor_critic(next_state)
        
        if done:
            advantage = reward - critic_output
            target = reward
        else:
            advantage = reward + self.gamma * next_critic_output - critic_output
            target = reward + self.gamma * next_critic_output
        actor_output, critic_output = self.actor_critic(state)
        dist = Categorical(F.softmax(actor_output, dim=-1))
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        actor_loss = -(log_prob * advantage.detach() + 0.001 * entropy).mean() 
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss ; 
        return advantage, actor_loss ,critic_loss ,loss
    
    def compute_loss2_high_order(self, state, action, reward, next_state, done, acts):

        high_order_calc=0 ; disc_R=0
        #line of interest 226 loop as described in paper, page 182 LA (1)
        for upd_ in range(0, len(self.high_order_dq)):
            reward = torch.tensor(self.reward_size_dq[upd_]).float()

            next_state = torch.from_numpy(self.high_order_dq[upd_]).float()
            _, next_critic_output = self.actor_critic(next_state)		
            high_order_calc=high_order_calc+((self.gamma**(upd_+1))*next_critic_output)
            #disc_R= disc_R + self.reward_size_dq[upd_]*(self.gamma**upd_)			

        action = torch.tensor(self.action_dq[0]).long()
        
        state = torch.from_numpy(self.order_size_dq[0]).float()
        actor_output, critic_output = self.actor_critic(state)
             	
	    #set advantage 1) simle 2) complex #line of interest 240
        advantage = self.reward_size_dq[0] + high_order_calc - critic_output
        #advantage = disc_R + high_order_calc - critic_output  
        
        dist = Categorical(F.softmax(actor_output, dim=-1))
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        actor_loss = -(log_prob * advantage.detach() + 0.001 * entropy).mean() 
        critic_loss=advantage

        loss = actor_loss + critic_loss ; #print(self.my_deque[0])
        return advantage, actor_loss ,critic_loss ,loss

    def compute_loss2_hight_order_plus_rewards_chain(self, state, action, reward, next_state, done, acts):

        high_order_calc=0 ; disc_R=0
        #line of interest 256, loop as described in paper, page 182 LAR (2)
        for upd_ in range(0, len(self.high_order_dq)):
            reward = torch.tensor(self.reward_size_dq[upd_]).float()

            next_state = torch.from_numpy(self.high_order_dq[upd_]).float()
            _, next_critic_output = self.actor_critic(next_state)		
            high_order_calc=high_order_calc+((self.gamma**(upd_+1))*next_critic_output)
            disc_R= disc_R + self.reward_size_dq[upd_]*(self.gamma**upd_) #line of interest 263 calculation chain of discounted rewards			

        action = torch.tensor(self.action_dq[0]).long()
        
        state = torch.from_numpy(self.order_size_dq[0]).float()
        actor_output, critic_output = self.actor_critic(state)
             	
	    #set advantage 1) simle 2) complex #line of interest 270
        advantage = disc_R + high_order_calc - critic_output  
        
        dist = Categorical(F.softmax(actor_output, dim=-1))
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        actor_loss = -(log_prob * advantage.detach() + 0.001 * entropy).mean() 
        critic_loss=advantage

        loss = actor_loss + critic_loss ; #print(self.my_deque[0])
        return advantage, actor_loss ,critic_loss ,loss

    def train(self, num_episodes, count_tr="ctrt"): 
        episode_actions = []
        episode_rewards = []
        episode_times = []
        highest_reward = float('-inf')
        best_animation = None
        for i in range(num_episodes):
            state = self.env.reset()
            ep_reward = 0 ; self.order_size_dq.clear() ; self.reward_size_dq.clear() ; self.high_order_dq.clear() ; self.action_dq.clear()
            done = False
            acts=0
            images = [] ; grads_keeper=[]
            ep_beg_time=time.time()
            while not done:
                state_ad=state
                state = torch.tensor(state, dtype=torch.float, device=self.device)
              
                action = self.choose_action(state)

                #add animation
                if acts % 1 ==0:
                    img = self.env.render(mode='rgb_array')
                    images.append(img)    
                
                next_state, reward, done, _ = self.env.step(action)
                
                acts=acts+1 ; 
                self.order_size_dq.append(state_ad) ; self.reward_size_dq.append(reward) ; self.high_order_dq.append(next_state) ; self.action_dq.append(action)

                if acts > 800:
                    done=True
                #env.render()
                
                ep_reward += reward

                if acts >10:  # custom ofline to online collector expirence 
                    if self.chain_order==1: #high order #line of interest 318 choise of chain order 
                        adv,act_los, crit_los ,loss = self.compute_loss2_high_order(state, action, reward, next_state, done, acts)
                    if self.chain_order_rewards==1: #highorder plus reward history #line of interest 320 choise of chain order plus reward context
                        adv,act_los, crit_los ,loss = self.compute_loss2_hight_order_plus_rewards_chain(state, action, reward, next_state, done, acts)

                    grads_keeper.append(float(act_los))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state
            end_ep=time.time()
            run_time= end_ep - ep_beg_time
            
            #add reward basics ; acts; times;
            episode_actions.append(acts)
            episode_rewards.append(ep_reward)
            episode_times.append(run_time)

            # Save the animation with the highest reward
            if ep_reward > highest_reward:
                highest_reward = ep_reward
                best_animation = images.copy() 

        #save animations
        with imageio.get_writer(f'/results/{self.label}results.gif', mode='I', fps=60) as writer:
            for image in best_animation:
                writer.append_data(image)
        imageio.imwrite(f'/results/{self.label}_last_frame.jpg', best_animation[-1])
        
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        
        # Bar chart for rewards
        axs[0].bar(range(1, num_episodes + 1), episode_rewards, color='green')
        axs[0].set_title('Rewards per Episode')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Rewards')
                
        # Bar chart for actions
        axs[1].bar(range(1, num_episodes + 1), episode_actions, color='blue')
        axs[1].set_title('Actions per Episode')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Actions')

        # Bar chart for times
        axs[2].bar(range(1, num_episodes + 1), episode_times, color='red')
        axs[2].set_title('Time per Episode')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Time (seconds)')

        # Automatically adjust layout to prevent overlapping
        plt.tight_layout()

        # Adjust layout to prevent overlapping with title
        plt.subplots_adjust(top=0.9)

        # Add a title over the entire set of subplots
        plt.suptitle(f'{self.label} , connections {self.actor_critic.num_connections()}  Summary', fontsize=16)

        # Save the figure as a PDF
        plt.savefig(f'/results/{self.label}metrics_summary.pdf')
        
def main(env_name_test='Pooyan-ram-v0', lbl="test1", control=1, HR=0.25 , FR=0.000025, s=1 , chain=0 , chain_r=0):
    
    env = gym.make(env_name_test)
    a2c_agent = A2C(env, gamma=0.9, label=lbl , rough_rl= HR, fine_rl=FR, sizer=s ,chain_order=chain , chain_order_rewards= chain_r )
    a2c_agent.set_order_memory_size(control+1)
    a2c_agent.train(num_episodes=11) 

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Example usage
directory_path = 'results'
create_directory(directory_path)

#scenarious run
main(lbl="v1 control test hight order 1, Pooyan-ram-v0", control=1 ,chain=1)
main(lbl="v2 high order N=2, Pooyan-ram-v0", control=2, chain=1)
main(lbl="v3 high order N=5, Pooyan-ram-v0", control=5, chain=1)
main(lbl="v4 high order N=2, chain order=2, Pooyan-ram-v0", control=2 , chain_r=1)
main(lbl="v5 high order N=5, chain order=5, Pooyan-ram-v0" ,  control=5, chain_r=1)
main(lbl="v6 control test hight order 1, KungFuMaster-ram-v0",env_name_test='KungFuMaster-ram-v0', control=1 ,chain=1)
main(lbl="v7 high order N=2, KungFuMaster-ram-v0",env_name_test='KungFuMaster-ram-v0', control=2, chain=1)
main(lbl="v8 high order N=5, KungFuMaster-ram-v0",env_name_test='KungFuMaster-ram-v0', control=5, chain=1)
main(lbl="v9 high order N=2, chain order=2, KungFuMaster-ram-v0",env_name_test='KungFuMaster-ram-v0', control=2 , chain_r=1)
main(lbl="v10 high order N=5, chain order=5, KungFuMaster-ram-v0" ,env_name_test='KungFuMaster-ram-v0',  control=5, chain_r=1)
#main(lbl="v6 fine LR=0.000000025\nANN:", HR=0.000000025, s=2)


