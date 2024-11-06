    def compute_loss2_high_Order_only_adv(self, state, action, reward, next_state, done, acts):
        print("lock in") ; 
        print(f" line 551 s0: {self.order_size_dq[0]} r: {self.reward_size_dq[0]}, s1: {self.high_order_dq[0]} , acts: {acts}")
        #print(self.order_size_dq)
        high_order_calc=0 ; disc_R=0

        for upd_ in range(0, len(self.high_order_dq)):
            reward = torch.tensor(self.reward_size_dq[upd_]).float()
            print(f'line 548 (gama**{upd_+1})V(_s{upd_}) +R*(gama**{upd_}')
            next_state = torch.from_numpy(self.high_order_dq[upd_]).float()
            _, next_critic_output = self.actor_critic(next_state)		
            high_order_calc=high_order_calc+((self.gamma**(upd_+1))*next_critic_output)
            disc_R= disc_R + self.reward_size_dq[upd_]*(self.gamma**upd_)			

        action = torch.tensor(self.action_dq[0]).long()
        
        state = torch.from_numpy(self.order_size_dq[0]).float()
        actor_output, critic_output = self.actor_critic(state)
        
     	
	    #set advantage 1) simle 2) complex
        advantage = self.reward_size_dq[0] + high_order_calc - critic_output
        #advantage = disc_R + high_order_calc - critic_output  
        
        dist = Categorical(F.softmax(actor_output, dim=-1))
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        actor_loss = -(log_prob * advantage.detach() + 0.001 * entropy).mean() 
        critic_loss=advantage
        #critic_loss = advantage.pow(2).mean()
        print(f"this is line 577 {advantage} this update loss: {critic_loss}")
        #critic_loss = advantage
        loss = actor_loss + critic_loss ; #print(self.my_deque[0])
        return advantage, actor_loss ,critic_loss ,loss


    def compute_loss2_high_order_advantage_plus_reward_ordering(self, state, action, reward, next_state, done, acts):
        print("lock in") ; 
        print(f" line 551 s0: {self.order_size_dq[0]} r: {self.reward_size_dq[0]}, s1: {self.high_order_dq[0]} , acts: {acts}")
        #print(self.order_size_dq)
        high_order_calc=0 ; disc_R=0

        for upd_ in range(0, len(self.high_order_dq)):
            reward = torch.tensor(self.reward_size_dq[upd_]).float()
            print(f'line 548 (gama**{upd_+1})V(_s{upd_}) +R*(gama**{upd_}')
            next_state = torch.from_numpy(self.high_order_dq[upd_]).float()
            _, next_critic_output = self.actor_critic(next_state)		
            high_order_calc=high_order_calc+((self.gamma**(upd_+1))*next_critic_output)
            disc_R= disc_R + self.reward_size_dq[upd_]*(self.gamma**upd_)			

        action = torch.tensor(self.action_dq[0]).long()
        
        state = torch.from_numpy(self.order_size_dq[0]).float()
        actor_output, critic_output = self.actor_critic(state)
        
     	
	    #set advantage 1) simle 2) complex
        #advantage = torch.tensor(self.reward_size_dq[0]).float() + high_order_calc - critic_output
        advantage = disc_R + high_order_calc - critic_output  
        
        dist = Categorical(F.softmax(actor_output, dim=-1))
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        actor_loss = -(log_prob * advantage.detach() + 0.001 * entropy).mean() 
        critic_loss=advantage
        #critic_loss = advantage.pow(2).mean()
        print(f"this is line 577 {advantage} this update loss: {critic_loss}")
        #critic_loss = advantage
        loss = actor_loss + critic_loss ; #print(self.my_deque[0])
        return advantage, actor_loss ,critic_loss ,loss
