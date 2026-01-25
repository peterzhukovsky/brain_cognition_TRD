# Overview
This repo includes scripts used to analyze brain-cognition relationships in the OPTIMUM-NEURO dataset (OPT-NEURO ClinicalTrials.gov number, NCT02960763 and OPTIMUM Award TRD-1511-33321, while OPTIMUM-Neuro was funded by the NIMH via a collaborative R01 mechanism (Pittsburgh: MH114969; Washington University: MH114966, CAMH/Toronto: MH114970; UCLA: MH114981; Columbia: MH114980) of treatment-resistant late-life depression.

The results are published under

*Brain-cognition relationships and treatment outcome in treatment-resistant late-life depression* Peter Zhukovsky, Meryl A. Butters, Helen Lavretsky, Patrick Brown, Joshua S. Shimony, Eric J Lenze, Daniel M Blumberger, Alastair J Flint, Jordan Karp, Steven Roose, Erin W Dickie, Daniel Felsky, Ginger E Nichol, Yar Al-Dabagh, Nicole Schoer, Feyifolu Obiri, Ashlyn Runk, Kayla Conaty, Benoit H Mulsant, Aristotle N Voineskos

Individual scripts include:

# 1. rsfc_pls_elnet
This script imports resting-state fMRI derivatives (210 pairwise connectivities between large ICs derived from over 4k UKB participants); it cleans the cognitive data and runs partial least squares regression analyses testing for brain-cogntion relationships; finally it runs separate sets of elastic net regularized logitstic regression models with cross-validation to predict remission (MADRS<=10) in step 1 and step 2 of the OPTIMUM RCT starting with clinical data only, then adding cognitive data and then adding RSFC data as predictors in the same set of subjects. 

The script is split into several parts:

&nbsp;&nbsp;&nbsp;&nbsp;**Cross-sectional analyses**
1. Data import and merging for cognitive and rs-fMRI data
2. Data cleaning\
   a. recoding cognitive test missing values to NaNs from 95s\
   b. option to subset data to MCI or non-MCI subgroups _(ln 80:84)_\
4. ComBat data harmonization and controlling for age, sex, site, mean motion, race/ethnicity
5. PLS regression predicting 6 cognitive tests from rs-FC data _(ln 85:230)_\
   a. Permutation testing for singificance of overall model\
   b. Bootstrapping for obtaining robust X-PLS weights (mapping X onto PLS latent X-scores XS)\
   c. Plotting of associations, including Bonferroni corrected correlations between X-scores (XS) and Y variables\
6. PLS cross-validation generatlizability testing leaving one site out _(ln 540:597)_
7. PLS latent score associations with\
   a. education _(ln 465:480)_\
   b. PHQ9 _(ln 484:491)_\
   c. centile brain scores _(ln 513:525)_\
   d. antidepressant treatment history form (ATHF), i.e. treatment resistance (ln 526:539)\
\
**Baseline MRI, cognitive data vs longitudinal MADRS data from OPTIMUM clinical trial**\
8. Importing longitudinal MADRS data _(ln 235)_\
   a. merging MADRS longitudinal data dates with the baseline data (dates of assessment) and defining remitters/nonremitters after 6wks of treatment vs patients at baseline (_ln 236:270)_\
   b. plotting and merging MADRS with baseline data _(ln 280:305)_\
9. Elastic net models predicting remission (MADRS) from baseline data _(ln 330:394)_\
   a. cross-validation _(330:394)_\
   b. plotting _(ln 395:410)_\
10. Visualizing the timeline of assessments _(ln 670:720)_\

# 2. dti_pls
This script imports fractional anisotropy (FA) derivatives from DTI imaging (mean FA in ~60 tracts that were successfully reconstructed using UKF tractography); it cleans the cognitive data and runs partial least squares regression analyses testing for brain-cogntion relationships; finally it runs separate sets of elastic net regularized logitstic regression models with cross-validation to predict remission (MADRS<=10) in step 1 and step 2 of the OPTIMUM RCT starting with clinical data only, then adding cognitive data and then adding FA data as predictors in the same set of subjects. 

# 3. CT_pls_elnet
This script imports Freesurfer derivatives (cortical thickness in the *aparc* atlas and subcortical volumes of the hippocampus, amygdala and striatal volumes in the *aseg* atlas); it cleans the cognitive data and runs partial least squares regression analyses testing for brain-cogntion relationships; finally it runs separate sets of elastic net regularized logitstic regression models with cross-validation to predict remission (MADRS<=10) in step 1 and step 2 of the OPTIMUM RCT starting with clinical data only, then adding cognitive data and then adding gray matter Freesurfer variables as predictors in the same set of subjects. 

#
MRI data was preprocessed using *fmriprep* 21 and rsfmri data was denoised using 24 fixed confound regression. Partial least squares models followed previous work (Zhukovsky et al 2022 PNAS, Morgan et al 2019 PNAS) using bootstrapping to identify robust weights and using permutation testing for model significance. We also cross-validate the PLS regression models by splitting the sample by the 4 sites with usable data, training the model on 3 sites and testing on the held-out site. 
