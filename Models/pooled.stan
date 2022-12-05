data {
  int<lower=0> N_train;                   // Number of observation in train set
  int<lower=0> K;                         // Number of columns 
  matrix[N_train, K] X_train;             // Observations in train set
  int<lower=0, upper=1> y_train[N_train]; // Target value in train set

  int<lower=0> N_test;                    // Number of observation in test set
  matrix[N_test, K] X_test;               // Observations in test set
}
parameters {
  real beta_0;
  vector[K] beta;
}
model {
  // Set priors
  beta_0 ~ normal(0, 1);
  beta ~ normal(0, 1);
  
  // Set regularization priors
  beta[3] ~ normal(0, 0.01);
  
  // Compute likelihood
  y_train ~ bernoulli_logit(beta_0 + X_train * beta);
}
generated quantities {
  // Compute predictions
  int<lower=0, upper=1> y_pred[N_test] = bernoulli_logit_rng(beta_0 + X_test * beta);

  // Compute log likelihood
  vector[N_train] log_lik;
  for(i in 1:N_train)
    log_lik[i] = bernoulli_logit_lpmf(y_train[i] | beta_0 + X_train[i] * beta);
}

