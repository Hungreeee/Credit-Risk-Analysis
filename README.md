# Credit-Risk-Analysis
Course project for CS-E5710 - Bayesian Data Analysis D

Analyze credit risks using pooled and hierarchical Logistic Regression.

Author: Hiep Nguyen and Hung Nguyen

## Introduction

This project is part of the Aalto University Bayesian Data Analysis 2022 Course. 

One of the most important services of banks that attract their customers is providing credit. However, when lenders offer home loans, auto loans, or business loans, there is an inherent risk that borrowers will default on their payments, and this is termed credit risk. Credit risk is universally known as the possibility of a loss for a lender due to a borrower’s failure to repay a loan. When the credit risk is mishandled by lenders, the consequences can be catastrophic. The collapse of the housing market in 2008 and the ensuing recession were one of the best illustrations of how severe the outcome can be: in just a few months, the banking sector nearly collapsed due to a significant overexposure to credit defaults. Therefore, it is essential to determine the borrowers’ ability to meet debt obligations as well as the risks involved, and this process is known as credit risk analysis.

The reason this topic is chosen is because of its meaningfulness and significance to the credit market, and the fact that there has been limited Bayesian data analysis project performed on the topic. Our goal is to use Logistic Regression to identify the riskiness of a loan by classifying them into good and bad loans. A **pooled** and **hierarchical** model will be applied to this problem, and through results analysis, we can compare and choose a stronger-performed model for this problem.

The report consists of the following parts: Introduction, Data Description, Models, Results, Discussion, Conclusion, and Appendices. First, we formulate the problem and show how we handle the data through pre-processing and feature selection processes. In the Models section, we describe the two Stan models used and justify their likelihood and justification of their choice. In the Results section, we analyze the performance of the models through convergence analysis, posterior predictive checks, predictive performance assessment, model selection, and prior sensitivity analysis. Finally, in the Discussion and Conclusion section, we will address the issues and potential improvements for our models, as well as provide some interesting insights that we have learned while doing the project. The Appendices part will include all the Stan models being used in this report. 

## Data description
The data set is obtained through Kaggle. It contains 1000 data points with 20 categorical attributes created by Professor Hofmann. In this data set, each entry represents a person taking credit from a bank, and each person is classified as having good or bad credit risk according to the set of attributes. The full data set can be found [here](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk).

The data consists of 10 columns, 9 explanatory variables, and 1 target variable, defined as follows:

Explanatory variables:

- Age (numerical/continuous): The age of the subject.
- Duration (numerical/continuous): The contracted loan due time (in months).
- Credit.amount (numerical/continuous): The amount of credit of the subject's loan.
- Sex (textual/categorical): The gender of the subject. This includes "`male`" and "`female`".
- Job (numerical/categorical): The level of employment of the subject. This includes "`0`" - unskilled and non-resident, "`1`" - unskilled and resident, "`2`" - skilled, "`3`" - highly skilled.
- Housing (textual/categorical): The type of housing of the subject. This includes "`own`", "`rent`" and "`free`".
- Saving.accounts (textual/categorical): The level of wealth of subject's saving account. This includes "`little`", "`moderate`", "`quite rich`", "`rich`".
- Checking.account (textual/categorical): The level of wealth of subject's checking account. This includes "`little`", "`moderate`", "`quite rich`", "`rich`".
- Purpose (textual/categorical): The purpose of the subject's loan. This includes "`business`", "`car`", "`domestic appliances`", "`education`", "`furniture/equipemt`", "`radio/TV`", "`repairs`", "`vacation/others`".

Target variable:

- Risk (textual/categorical): The fact that the subject has defaulted (not repaying the loan). This includes "`good`" - good loan (the subject has repaid on time), "`bad`" - bad loan (the subject has defaulted).