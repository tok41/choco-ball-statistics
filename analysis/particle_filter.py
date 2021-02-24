"""Particle Filter
"""

import numpy as np
from tqdm import tqdm
from empirical_dist import EmpiricalDist

class ParticleFilter():
    def __init__(self, init_particles, system_model, obs_model, params):
        n_particles, _ = init_particles.shape
        self.particles = init_particles
        self.n_particles = n_particles
        self.system_model = system_model
        self.observ_model = obs_model
        self.params = params
        self.hist_particles = []
        #self.hist_particles.append(init_particles)
        self.hist_weigts = []
        self.likelihoods = []
        print(f'n_particles : {self.n_particles}')
    
    def update(self):
        self.particles = np.array([self.system_model(xp, self.params) 
                                   for xp in self.particles])
    
    def weight(self, obs):
        self.ws = [self.observ_model(obs, x_t, self.params) 
                   for i, x_t in enumerate(self.particles)]
        self.hist_weigts.append(self.ws)
        self.likelihoods.append(np.mean(self.ws))
    
    def resampling(self):
        if sum(self.ws) < 0.000000001:
            ws_ = np.ones(self.n_particles) / self.n_particles
        else:
            ws_ = self.ws / sum(self.ws)
        indx = np.random.choice(np.arange(self.n_particles), 
                                size=self.n_particles, replace=True, 
                                p=ws_)
        X_fltr = self.particles[indx]
        self.particles = X_fltr
        self.hist_particles.append(self.particles)
    
    def update_params(self):
        empdist = self.params['emp_dist']
        empdist.set_sample_data(self.particles)
        self.params['emp_dist'] = empdist

    def calc(self, obs):
        print(f'particle filtering')
        for i, y in tqdm(enumerate(obs)):
            self.update_params()
            self.update()
            self.weight(y)
            self.resampling()
        self.hist_particles = np.array(self.hist_particles)
        self.marginal_log_lik = np.log(self.likelihoods).sum()
