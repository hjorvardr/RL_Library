

# actor learn
# INPUT: state, action, td_error

prediction = self.actor_model.predict(state)
log_prob = log(prediction[0][action])

#new_prediction = log_prob * td_error
target = log_prob * self.critic_model.predict(state)[0]

new_prediction = self.actor_model.predict(state)
new_prediction[0][action] = -target
self.actor_model.fit(state, new_prediction, verbose=0)




# critic learn
# INPUT: state, new_state, reward

td_error = reward + 
		 self.gamma * self.critic_model.predict(new_state)[0]
		 - self.critic_model.predict(state)[0] # only diff

self.critic.fit(state, td_error, verbose=0)




# actor learn
# INPUT: state, action, td_error

prediction = self.actor_model.predict(state)
log_prob = log(prediction[0][action])

new_prediction = log_prob * td_error

new_prediction = self.actor_model.predict(state)
new_prediction[0][action] = -target
self.actor_model.fit(state, new_prediction, verbose=0)

