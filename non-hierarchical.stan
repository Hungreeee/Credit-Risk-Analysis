data {
  int<lower=0> N_train; // Observation
  int<lower=0> D; // Columns
  matrix[N_train, D] X_train;
  int<lower=0, upper=1> y_train[N_train];
  
  int<lower=0> N_test;
  matrix[N_test, D] X_test;
  // int<lower=0, upper=1> y_test[N_test];
}

parameters {
  real beta_0;
  vector[D] beta;
}

model {
  beta_0 ~ normal(0, 2);
  beta[1] ~ normal(0, 2);
  beta[2] ~ normal(0, 2);
  beta[3] ~ normal(0, 2);
  beta[4] ~ normal(0, 2);
  beta[5] ~ normal(0, 2);
  beta[6] ~ normal(0, 2);
  beta[7] ~ normal(0, 2);
  beta[8] ~ normal(0, 2);
  beta[9] ~ normal(0, 2);
  y_train ~ bernoulli_logit(beta_0 + X_train * beta);
}

generated quantities {
  int<lower=0, upper=1> y_pred[N_test] = bernoulli_logit_rng(beta_0 + X_test * beta);
}




