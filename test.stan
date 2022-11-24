data {
  int<lower=0> N_train; // Observation
  int<lower=0> J; // Columns
  matrix[N_train, J] X_train; 
  int<lower=0, upper=1> y_train[N_train];
  
  int<lower=0> N_test;
  matrix[N_test, J] X_test;
  int<lower=0, upper=1> y_test[N_test];
}

parameters {
  real beta_0;
  vector[J] beta;
}

model {
  y_train ~ bernoulli_logit(beta_0 + X_train * beta);
}

generated quantities {
  int<lower=0, upper=1> y_pred[N_test] = bernoulli_logit_rng(beta_0 + X_test * beta);
}





