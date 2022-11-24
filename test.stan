data {
  int<lower=0> N;
  int<lower=0> J;
  matrix[N, J] X;
  int<lower=0, upper=1> y[N];
  
  int<lower=0> N_test;
  matrix[N_test, J] X_test;
  int<lower=0, upper=1> y_test[N_test];
}

parameters {
  real alpha;
  vector[J] beta;
}

model {
  y ~ bernoulli_logit(alpha + X * beta);
}

generated quantities {
  int y_pred[N] = bernoulli_logit_rng(alpha + X_test * beta);
  // real log_loss = -bernoulli_logit_lpmf(y_test | alpha + X_test * beta);
}





