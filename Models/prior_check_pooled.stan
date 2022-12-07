data {
  int<lower=0> N_train; 
  int<lower=0> K; 
  matrix[N_train, K] X_train;
  int<lower=0, upper=1> y_train[N_train];
  
  int<lower=0> N_test;
  matrix[N_test, K] X_test;
}

parameters {
  real beta_0;
  vector[K] beta;
}

model {
  beta_0 ~ normal(2, 0.3); // Change prior
  beta ~ normal(0, 1);
  beta[3] ~ normal(0, 0.01);
  y_train ~ bernoulli_logit(beta_0 + X_train * beta);
}

generated quantities {
  int<lower=0, upper=1> y_pred[N_test] = bernoulli_logit_rng(beta_0 + X_test * beta);
}





