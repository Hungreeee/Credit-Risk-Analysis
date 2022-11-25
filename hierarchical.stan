data {
  int<lower=0> N_train; // Observation
  int<lower=0> D; // Columns
  int<lower=0> L_train;
  row_vector[D] X_train[N_train]; 
  int<lower=0, upper=L_train> ll_train[N_train];
  int<lower=0, upper=1> y_train[N_train];
  
  int<lower=0> N_test;
  int<lower=0> L_test;
  row_vector[D] X_test[N_test]; 
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
  mu_0 ~ normal(0, 2);
  sigma_0 ~ normal(0, 2);
  beta_0 ~ normal(mu_0, sigma_0);
  
  for(i in 1:D) {
    mu[i] ~ normal(0, 2);
    sigma[i] ~ normal(0, 2);
    for(j in 1:L_train) {
      beta[i, j] ~ normal(mu[i], sigma[i]);
    }
  }
  
  for(i in 1:N_train) 
    y_train[i] ~ bernoulli_logit(beta_0[ll_train[i]] +  X_train[i] * beta[ll_train[i]]);
}

generated quantities {
  int<lower=0, upper=1> y_pred[N_test];
  
  for(i in 1:N_test) 
    y_pred[i] = bernoulli_logit_rng(beta_0[ll_test[i]] + X_test[i] * beta[ll_test[i]]);
}





