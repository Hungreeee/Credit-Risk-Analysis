data {
  int<lower=0> N_train; // Observation
  int<lower=0> D; // Columns
  matrix[N_train, D] X_train;
  int<lower=0, upper=1> y_train[N_train];
  
  int<lower=0> N_test;
  matrix[N_test, D] X_test;
}

parameters {
  real beta_0;
  vector[D] beta;
}

model {
  beta_0 ~ normal(0, 1);
  beta ~ normal(0, 1);
  beta[3] ~ normal(0, 0.01);
  // beta[4] ~ normal(0, 0.01);
  // beta[6] ~ normal(0, 0.01);
  y_train ~ bernoulli_logit(beta_0 + X_train * beta);
}

generated quantities {
  int<lower=0, upper=1> y_pred[N_test] = bernoulli_logit_rng(beta_0 + X_test * beta);
  
  vector[N_train] log_lik;
  for(i in 1:N_train)
    log_lik[i] = bernoulli_logit_lpmf(y_train[i] | beta_0 + X_train[i] * beta);
}





