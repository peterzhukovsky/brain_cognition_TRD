# Overview
This repo includes scripts used to analyze brain-cognition relationships in the OPTIMUM-NEURO dataset (OPT-NEURO ClinicalTrials.gov number, NCT02960763 and OPTIMUM Award TRD-1511-33321, while OPTIMUM-Neuro was funded by the NIMH via a collaborative R01 mechanism (Pittsburgh: MH114969; Washington University: MH114966, CAMH/Toronto: MH114970; UCLA: MH114981; Columbia: MH114980) of treatment-resistant late-life depression.

The results are published under
Brain-cognition relationships and treatment outcome in treatment-resistant late-life depression
Peter Zhukovsky, Meryl A. Butters, Helen Lavretsky, Patrick Brown, Joshua S. Shimony, Eric J Lenze, Daniel M Blumberger, Alastair J Flint, Jordan Karp, Steven Roose, Erin W Dickie, Daniel Felsky, Ginger E Nichol, Yar Al-Dabagh, Nicole Schoer, Feyifolu Obiri, Ashlyn Runk, Kayla Conaty, Benoit H Mulsant, Aristotle N Voineskos

Individual scripts include:

# 1. rsfc_OPT_PLS_2024
This script imports resting-state fMRI derivatives (210 pairwise connectivities between large ICs derived from over 4k UKB participants); it cleans the cognitive data and runs partial least squares analyses testing for brain-cogntion relationships; finally it runs separate sets of elastic net regularized logitstic regression models with cross-validation to predict remission (MADRS<=10) in step 1 and step 2 of the OPTIMUM RCT starting with clinical data only, then adding cognitive data and then adding RSFC data as predictors in the same set of subjects. 

# 2. dti_pls_2024
This script imports fractional anisotropy (FA) derivatives from DTI imaging (mean FA in ~60 tracts that were successfully reconstructed using UKF tractography); it cleans the cognitive data and runs partial least squares analyses testing for brain-cogntion relationships; finally it runs separate sets of elastic net regularized logitstic regression models with cross-validation to predict remission (MADRS<=10) in step 1 and step 2 of the OPTIMUM RCT starting with clinical data only, then adding cognitive data and then adding FA data as predictors in the same set of subjects. 

# 2. grant_predict_CT_2024
This script imports fractional anisotropy (FA) derivatives from DTI imaging (mean FA in ~60 tracts that were successfully reconstructed using UKF tractography); it cleans the cognitive data and runs partial least squares analyses testing for brain-cogntion relationships; finally it runs separate sets of elastic net regularized logitstic regression models with cross-validation to predict remission (MADRS<=10) in step 1 and step 2 of the OPTIMUM RCT starting with clinical data only, then adding cognitive data and then adding FA data as predictors in the same set of subjects. 
