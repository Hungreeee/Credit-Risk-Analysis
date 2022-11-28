data {
  int<lower=0> N_train; // Observation
  int<lower=0> D; // Columns
  int<lower=0> L_train;
  matrix[N_train, D] X_train; 
  int<lower=0, upper=L_train> ll_train[N_train];
  int<lower=0, upper=1> y_train[N_train];
  
  int<lower=0> N_test;
  int<lower=0> L_test;
  matrix[N_test, D] X_test; 
  int<lower=0, upper=L_test> ll_test[N_test];
}

parameters {
  real mu_0;
  real<lower=0> sigma_0;
  real beta_0[L_train];
  
  real mu[D];
  real<lower=0> sigma[D];
  vector[D] beta[L_train];
}

model {
  mu_0 ~ normal(0, 1);
  sigma_0 ~ inv_gamma(0.5, 1);
  
  beta_0 ~ normal(mu_0, sigma_0);
  
  mu ~ normal(0, 1);
  sigma ~ inv_gamma(0.5, 1);
  
  mu[3] ~ normal(0, 0.01);
  sigma[3] ~ inv_gamma(1, 0.01);
  
  // mu[4] ~ normal(0, 0.01);
  // sigma[4] ~ inv_gamma(1, 0.01);
  
  // mu[6] ~ normal(0, 0.01);
  // sigma[6] ~ inv_gamma(1, 0.01);
  
  for(i in 1:L_train) {
    beta[i] ~ normal(mu, sigma);
  }
  
  for(i in 1:N_train) 
    y_train[i] ~ bernoulli_logit(beta_0[ll_train[i]] +  X_train[i] * beta[ll_train[i]]);
}

generated quantities {
  int<lower=0, upper=1> y_pred[N_test];
  
  for(i in 1:N_test) 
    y_pred[i] = bernoulli_logit_rng(beta_0[ll_test[i]] + X_test[i] * beta[ll_test[i]]);
    
  vector[N_train] log_lik;
  for(i in 1:N_train) 
    log_lik[i] = bernoulli_logit_lpmf(y_train[i] | beta_0[ll_train[i]] + X_train[i] * beta[ll_train[i]]);
  
}





