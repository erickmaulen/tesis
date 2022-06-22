
import numpy as np
import os
import PIL
import PIL.Image as im
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import cv2
import scipy.signal
import time
import cv2




def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

def load_img():
  global global_img
  global_img =  [('agent',cv2.imread(f"agent_{i}.png", 0)) for i in range (1,6)] 
  global_img += [ ('enemy',cv2.imread(f"enemy_{i}.png", 0)) for i in range(1,4)]
  global_img += [('enemy',cv2.imread(f"enemy_{i}{2}.png", 0)) for i in range(1,4)]
  global_img += [('point',cv2.imread(f"point_{i}.png", 0)) for i in range(1,4)]
  global_img += [('point',cv2.imread(f"point_{i}{2}.png", 0)) for i in range(3,4)]
  global_img += [('bonus',cv2.imread(f"bonus.png", 0))]

global_img = []
load_img()



# board
# objets = global_obj
# threashold = margen de error (umbral)
def find_objects(board, objects, threshold=0.95):
  obj_centroids={}
  i = 0
  for obj in objects:
    if obj[0] not in obj_centroids: 
      obj_centroids[obj[0]] = [] # Primer atributo que identifica el nombre Ej: agent, enemy, point, bonus
      #print(board.shape)
      #print("nombre de objeto ",obj[0])
    res = cv2.matchTemplate(board, obj[1], cv2.TM_CCOEFF_NORMED)
    loc = np.argwhere(res >= threshold)

    w, h = obj[1].shape[::-1]
    for o in loc:
        y, x = o
        centroid = (x + (w / 2), y + (h / 2))
        obj_centroids[obj[0]].append(list(centroid))
    i+=1
  #print("dentro de find_objects ",obj_centroids)
  return obj_centroids

def preprocess_frame(state, int_state=None, show=False):
  
  global global_img
  #gray = state
  gray=cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
  objects = find_objects(gray,global_img)

#SI NO SE PUEDE COMER A LOS FANTASMAS 
  if int_state is not None:
    #gray=cv2.cvtColor(int_state, cv2.COLOR_BGR2GRAY)
    objects2 = find_objects(gray,[global_img[1]])
  else: objects2=None

  if show: cv2_imshow(gray)

  if len(objects['agent'])==0: return np.zeros((480,))

#distancia entre los enemigos y el agente
  for obj in objects:
    if obj=='agent': continue
    
    for item in objects[obj]:

      item[0]-=objects['agent'][0][0]
      item[1] =objects['agent'][0][1] - item[1]

  if objects2 is not None:
    for obj in objects2:
      if obj=='enemy': continue
      for item in objects2[obj]:
        item[0]-=objects['agent'][0][0]
        item[1] =objects['agent'][0][1] - item[1]

  # take second element for sort
  def abs_x(elem):
      return abs(elem[0])+0.01*elem[1]

  def second(elem):
      return abs(elem[1])
#ordenar para ver cercanía
    # sort list with key

  if objects2 is None: objects['enemy'].sort(key=abs_x)  
  else: objects2['point'].sort(key=second)


  state = np.zeros((160,3))
  state[0][0]=objects['agent'][0][0]/160
  state[0][1]=objects['agent'][0][1]/210

  #enemies
  if objects2 is None:
    i=1
    for enemy in objects['enemy']:
      state[i][0]=enemy[0]/160
      state[i][1]=enemy[1]/210
      state[i][2]=np.sqrt(state[i][0]**2+state[i][1]**2)
      i+=1
      if i>3: break

  #points
  i=4
  for point in objects['point']:
    state[i][0]=point[0]/160
    state[i][1]=point[1]/210
    state[i][2]=np.sqrt(state[i][0]**2+state[i][1]**2)
    i+=1
    if i>153: break

  #phantoms' points
  i=154
  if objects2 is not None:
    for point in objects2['point']:
      state[i][0]=point[0]/160
      state[i][1]=point[1]/210
      state[i][2]=np.sqrt(state[i][0]**2+state[i][1]**2)
      i+=1
      if i>156: break

  #bonus
  i=157
  for bonus in objects['bonus']:
    state[i][0]=bonus[0]/160
    state[i][1]=bonus[1]/210
    state[i][2]=np.sqrt(state[i][0]**2+state[i][1]**2)
    i+=160
    if i>=160: break
  #AGREGAR GUINDA
  state.shape = ((160*3),)
  return state

from collections import deque

stacked_frames = None
n_frames=4

def env_reset(env):
  env.reset()
  state, reward, done, lives = env_step(env, 0, iters=5, new_episode=True) #starting
  return state

def stack_frames(state, is_new_episode):  
    global stacked_frames
    if is_new_episode or stacked_frames==None:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((observation_dimensions), dtype=np.int) for i in range(n_frames)], maxlen=n_frames)
        
        # Because we're in a new episode, copy the same frame 4x
        for i in range(n_frames):
          stacked_frames.append(state)
        
        # Stack the state
        stacked_state = np.stack(stacked_frames, axis=1)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        if state is not None: stacked_frames.append(state)
        else: stacked_frames.append(np.zeros((observation_dimensions), dtype=np.int))

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=1) 
    
    return stacked_state

#Solo sirve si se quiere imprimir la imagen
images = None
def env_step(env, action, iters=5, new_episode=False, show=False):

  reward=0.
  #action = possible_actions[action]
  #print("ACTION : ",action)
  int_state = None
  for i in range(iters):
    
    next_state, r, done, lives = env.step(action)
    reward+=r
    if done: break

  if(show):
    if int_state is not None: images.append(int_state)
    images.append(next_state)

  if next_state is not None: 
#    print(next_state, "NEXT STATE ANTES")
#    print(next_state.shape)

    next_state = preprocess_frame(next_state, int_state) ## arreglo de posiciones
    #print("ESTO ME TIRA DE INFORMACION : ",next_state.shape)#print("DONE : ",done)
    
 #   print(next_state, "NEXT STATE DESPUES")
 #   print("observation_dimensions ", observation_dimensions)
  else: next_state = np.zeros((observation_dimensions), dtype=np.float)
  
  #CAMBIAR A CUANDO PIERDE UNA VIDA
  if next_state[0]==0: done=True

  state = stack_frames(next_state,new_episode)

  return state, reward, done, lives

# Hyperparameters of the PPO algorithm
steps_per_epoch = 6000
epochs = 100
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)

# True if you want to render the environment
render = False

# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

#import retro
# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
# Create our environment
# env.close()
env=gym.make('MsPacman-v0')
observation_dimensions = 3*160*n_frames # 10 objects
num_actions = env.action_space.n

possible_actions = np.array(np.identity(num_actions,dtype=int).tolist())

#print("The size of our frame is: ", env.observation_space)

# Here we create an hot encoded version of our actions
# possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())

# Initialize the buffer
buffer = Buffer(observation_dimensions, steps_per_epoch)

# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
actor = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
)
critic = keras.Model(inputs=observation_input, outputs=value)

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

# Initialize the observation, episode return and episode length

observation, episode_return, episode_length = env_reset(env), 0, 0
print("observation : ", observation.shape)

# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0
    episode_max_return = 0

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        if render:
            env.render()

        # Get the logits, action, and take one step in the environment
        observation = observation.reshape(1, -1)
        
        logits, action = sample_action(observation)
        #observation_new, reward, done, _ = env.step(action[0].numpy()) #AQUI
        observation_new, reward, done, _ = env_step(env, action[0].numpy(), iters=5) #AQUI

        #Que termine si muere una sola vez
        #if _ == {'ale.lives': 2}:
        #  done = True

        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = critic(observation)
        logprobability_t = logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(observation, action, reward, value_t, logprobability_t)
        #buffer2.store(observation, observation_new, reward)
        # Update the observation
        observation = observation_new

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(observation.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            if episode_return > episode_max_return: episode_max_return=episode_return
            observation, episode_return, episode_length = env_reset(env), 0, 0  #ACA
            

    # Get values from the buffer
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = train_policy(
            observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, return_buffer)
    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. MeanReturn: {sum_return / num_episodes}. MeanLength: {sum_length / num_episodes}. MaxReturn: {episode_max_return}"
    )

    actor.save('actor-1000')
    critic.save('critic-1000')

while(episode_return < 1000):
  observation, episode_return, episode_length = env_reset(env), 0, 0


  images = []
  for t in range(steps_per_epoch):

      # Get the logits, action, and take one step in the environment
      observation = observation.reshape(1, -1)
      
      logits, action = sample_action(observation)
      #observation_new, reward, done, _ = env.step(action[0].numpy()) #AQUI
      observation_new, reward, done, _ = env_step(env, action[0].numpy(), iters=5, show=True) #AQUI
      

      episode_return += reward
      episode_length += 1

      # Get the value and log-probability of the action
      value_t = critic(observation)
      logprobability_t = logprobabilities(logits, action)

      # Update the observation
      observation = observation_new

      # Finish trajectory if reached to a terminal state
      terminal = done
      if terminal:
        break


  print("e : ",episode_return)

res=(160,210) #resulotion
out = cv2.VideoWriter('video2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10.0, res)
for image in images:
    out.write(image)

out.release()