%% OPT rsfmri analysis
clear
%OPT data import
%OPT NEURO
load('C:\Users\peter\Documents\OPT\OPT\reports\rsfc_2023\OPT_rsfmri_281_12.mat');
%load('C:\Users\peter\Documents\OPT\OPT\reports\rsfc_2021\feat.mat');
cd C:\Users\peter\Documents\OPT\OPT\reports\rsfc_2023
CTl=readtable('C:\Users\peter\Documents\OPT\OPT\data\brain_age\lh.aparc_stats.txt');CTr=readtable('C:\Users\peter\Documents\OPT\OPT\data\brain_age\rh.aparc_stats.txt');CTl.BrainSegVolNotVent=[]; CTl.eTIV=[];CTr.rh_aparc_thickness=[];
subjects=horzcat(CTl, CTr);subjects.lh_MeanThickness_thickness=[];
aseg=readtable('C:\Users\peter\Documents\OPT\OPT\data\brain_age\aseg_stats.txt');
subjects.lh_hippo=aseg.Left_Hippocampus./aseg.EstimatedTotalIntraCranialVol*1000;subjects.rh_hippo=aseg.Right_Hippocampus./aseg.EstimatedTotalIntraCranialVol*1000;
subjects.lh_amygdala=aseg.Left_Amygdala./aseg.EstimatedTotalIntraCranialVol*1000;subjects.rh_amygdala=aseg.Right_Amygdala./aseg.EstimatedTotalIntraCranialVol*1000;
subjects.lh_striatum=aseg.Left_Accumbens_area./aseg.EstimatedTotalIntraCranialVol+aseg.Left_Caudate./aseg.EstimatedTotalIntraCranialVol+aseg.Left_Putamen./aseg.EstimatedTotalIntraCranialVol;subjects.lh_striatum=subjects.lh_striatum*1000;
subjects.rh_striatum=aseg.Right_Accumbens_area./aseg.EstimatedTotalIntraCranialVol+aseg.Right_Caudate./aseg.EstimatedTotalIntraCranialVol+aseg.Right_Putamen./aseg.EstimatedTotalIntraCranialVol;subjects.rh_striatum=subjects.rh_striatum*1000;

subjects.qc=aseg.SurfaceHoles>380;%subjects.qc=aseg.SurfaceHoles>220;
subjects.scanner= ~cellfun(@isempty,(strfind(subjects.lh_aparc_thickness, 'CU'))) |  ~cellfun(@isempty,(strfind(subjects.lh_aparc_thickness, 'UT1UT1')))  |  ~cellfun(@isempty,(strfind(subjects.lh_aparc_thickness, 'UT1UT3'))) ;
try; subjects.ID={'CU10001';'CU10003';'CU10007';'CU10009';'CU10010';'CU10012';'CU10014';'CU10021';'CU10022';'CU10023';'CU10024';'CU10026';'CU10031';'CU10033';'CU10034';'CU10038';'CU10045';'CU10056';'CU10057';'CU10060';'CU10070';'CU10074';'CU10076';'CU10078';'CU10079';'CU10081';'CU10083';'CU10084';'CU10088';'CU10089';'CU10091';'CU10098';'CU10099';'CU10101';'CU10104';'CU10109';'CU10110';'CU10114';'CU10119';'CU10141';'CU10146';'CU20002';'CU20003';'CU20005';'CU20006';'CU20011';'CU20015';'CU20016';'LA10006';'LA10008';'LA10026';'LA10032';'LA10035';'LA10038';'LA10040';'LA10044';'LA10048';'LA10054';'LA10056';'LA10074';'LA10099';'LA10104';'LA10105';'LA20002';'LA20012';'LA20019';'LA20021';'LA20032';'LA20039';'LA20043';'UP10004';'UP10007';'UP10044';'UP10047';'UP10057';'UP10071';'UP10074';'UP10076';'UP10077';'UP10092';'UP10094';'UP10110';'UP10124';'UP10155';'UP10266';'UP10280';'UP10005';'UP10021';'UP10046';'UP10053';'UP10061';'UP10062';'UP10087';'UP10109';'UP10111';'UP10133';'UP10137';'UP10148';'UP10165';'UP10203';'UP10206';'UP10231';'UP10258';'UP10261';'UP10001';'UP10003';'UP10006';'UP10009';'UP10020';'UP10026';'UP10032';'UP10049';'UP10054';'UP10058';'UP10066';'UP10080';'UP10081';'UP10090';'UP10096';'UP10098';'UP10101';'UP10112';'UP10113';'UP10114';'UP10125';'UP10126';'UP10128';'UP10130';'UP10135';'UP10136';'UP10151';'UP10156';'UP10158';'UP10161';'UP10163';'UP10171';'UP10173';'UP10175';'UP10184';'UP10187';'UP10188';'UP10196';'UP10201';'UP10204';'UP10209';'UP10210';'UP10229';'UP10244';'UP10250';'UT10006';'UT10018';'UT10023';'UT10066';'UT10067';'UT10072';'UT10073';'UT10075';'UT10078';'UT10079';'UT10080';'UT10083';'UT10091';'UT10109';'UT10111';'UT10115';'UT10116';'UT10125';'UT10126';'UT10128';'UT10130';'UT10137';'UT10148';'UT30006';'UT30008';'UT30011';'UT30018';'UT30019';'UT30021';'UT30025';'UT30026';'UT30027';'UT30028';'UT30030';'UT30032';'UT30033';'UT30034';'UT30036';'UT30040';'UT30042';'UT10001';'UT10003';'UT10004';'UT10013';'UT10015';'UT10016';'UT10019';'UT10021';'UT10025';'UT10033';'UT10035';'UT10039';'UT10046';'UT10054';'UT10081';'UT10087';'UT10090';'UT10114';'UT10120';'UT10123';'UT10144';'UT30003';'UT30029';'UT30031';'UT30039';'UT30041';'UT30043';'WU10089';'WU10031';'WU10055';'WU10057';'WU10085';'WU10099';'WU10105';'WU10114';'WU10115';'WU10117';'WU10127';'WU10131';'WU10137';'WU10146';'WU10150';'WU10152';'WU10157';'WU10161';'WU10164';'WU10168';'WU10174';'WU10189';'WU20001';'WU20002';'WU20004';'WU20005';'WU20006';'WU10008';'WU10014';'WU10028';'WU10034';'WU10036';'WU10054';'WU10058';'WU10059';'WU10062';'WU10064';'WU10066';'WU10071';'WU10073';'WU10077';'WU10083';'WU10087';'WU10092';'WU10093';'WU10096';'WU10104';'WU10107';'WU10108';'WU10112';'WU10121';'WU10122';'WU10124';'WU10128';'WU10135';'WU10141';'WU10145';'WU10147';'WU10149';'WU10151';'WU10154';'WU10158';'WU10165';'WU10166';'WU10177'};end
%try; subjects.ID={'CU10001';'CU10007';'CU10009';'CU10010';'CU10012';'CU10014';'CU10021';'CU10022';'CU10023';'CU10026';'CU10031';'CU10033';'CU10034';'CU10038';'CU10045';'CU10056';'CU10057';'CU10060';'CU10070';'CU10074';'CU10076';'CU10078';'CU10079';'CU10081';'CU10083';'CU10084';'CU10088';'CU10089';'CU10091';'CU10098';'CU10099';'CU10101';'CU10104';'CU10109';'CU10110';'CU10114';'CU10119';'CU10141';'CU10146';'CU20002';'CU20003';'CU20005';'CU20006';'CU20011';'CU20015';'CU20016';'LA10006';'LA10008';'LA10026';'LA10032';'LA10035';'LA10038';'LA10040';'LA10044';'LA10048';'LA10054';'LA10056';'LA10074';'LA10099';'LA10104';'LA10105';'LA20012';'LA20019';'LA20021';'LA20032';'LA20039';'LA20043';'UP10004';'UP10007';'UP10044';'UP10047';'UP10057';'UP10071';'UP10074';'UP10076';'UP10077';'UP10092';'UP10094';'UP10110';'UP10124';'UP10155';'UP10266';'UP10280';'UP10005';'UP10021';'UP10046';'UP10053';'UP10061';'UP10062';'UP10087';'UP10109';'UP10111';'UP10133';'UP10137';'UP10148';'UP10165';'UP10203';'UP10206';'UP10231';'UP10258';'UP10261';'UP10001';'UP10003';'UP10006';'UP10009';'UP10020';'UP10026';'UP10032';'UP10049';'UP10054';'UP10058';'UP10066';'UP10080';'UP10081';'UP10090';'UP10096';'UP10098';'UP10101';'UP10112';'UP10113';'UP10114';'UP10125';'UP10126';'UP10128';'UP10130';'UP10135';'UP10136';'UP10151';'UP10156';'UP10158';'UP10161';'UP10163';'UP10171';'UP10173';'UP10175';'UP10184';'UP10187';'UP10188';'UP10196';'UP10201';'UP10204';'UP10209';'UP10210';'UP10229';'UP10244';'UP10250';'UT10006';'UT10018';'UT10023';'UT10066';'UT10067';'UT10072';'UT10073';'UT10075';'UT10078';'UT10079';'UT10080';'UT10083';'UT10091';'UT10109';'UT10111';'UT10115';'UT10116';'UT10125';'UT10126';'UT10128';'UT10130';'UT10137';'UT10148';'UT30006';'UT30008';'UT30011';'UT30018';'UT30019';'UT30021';'UT30025';'UT30026';'UT30027';'UT30028';'UT30030';'UT30032';'UT30033';'UT30034';'UT30036';'UT30040';'UT30042';'UT10003';'UT10004';'UT10013';'UT10015';'UT10016';'UT10019';'UT10021';'UT10025';'UT10033';'UT10035';'UT10039';'UT10046';'UT10054';'UT10081';'UT10087';'UT10090';'UT10114';'UT10120';'UT10123';'UT10144';'UT30003';'UT30029';'UT30031';'UT30039';'UT30041';'UT30043';'WU10089';'WU10031';'WU10055';'WU10057';'WU10085';'WU10099';'WU10105';'WU10114';'WU10115';'WU10117';'WU10127';'WU10131';'WU10137';'WU10146';'WU10150';'WU10152';'WU10157';'WU10161';'WU10164';'WU10168';'WU10174';'WU10189';'WU20001';'WU20002';'WU20004';'WU20005';'WU20006';'WU10008';'WU10014';'WU10028';'WU10034';'WU10036';'WU10054';'WU10058';'WU10059';'WU10062';'WU10064';'WU10066';'WU10071';'WU10073';'WU10077';'WU10083';'WU10087';'WU10092';'WU10093';'WU10096';'WU10104';'WU10107';'WU10108';'WU10112';'WU10121';'WU10122';'WU10124';'WU10128';'WU10135';'WU10141';'WU10145';'WU10147';'WU10149';'WU10151';'WU10154';'WU10158';'WU10165';'WU10166';'WU10177'};end
%qcfail={'CU10010';'CU10114';'LA10040';'UP10094';'UP10124';'UP10049';'UP10081';'UP10096';'UP10114';'WU10014'};
%d=readtable('C:\Users\peter\Documents\OPT\OPT\data\ds_ON_BASELINE_DATA_April2022n_for_cleaning_ID_6_5_22.xlsx');
d=readtable('C:\Users\peter\Documents\OPT\OPT\data\ON_DATASET_08302024.xlsx'); d.ID_original=d.ID; d.ID=d.ID_complete;
qc=readtable('C:\Users\peter\Documents\OPT\OPT\reports\freesurfer_longitudinal\QC_ABCD_outputs.txt');
subjects(subjects.qc==1, :)=[];
%%
d.CWI3CSSFinal_01(d.CWI3CSSFinal_01>80)=NaN;
d.DTMTS4_01(d.DTMTS4_01>80)=NaN;

sum(~isnan(d.DTMTS4_01))
sum(~isnan(d.CWI3CSSFinal_01))
d.DTMTS4_01(isnan(d.DTMTS4_01))=nanmean(d.DTMTS4_01);
d.CWI3CSSFinal_01(isnan(d.CWI3CSSFinal_01))=nanmean(d.CWI3CSSFinal_01);

d.EXEC_01=(d.DTMTS4_01+d.CWI3CSSFinal_01)/2;

%corr(d.EXEC_01, d.CWI3CSSFinal_01)
d(d.SITE_MRI_1_WU_2_UT_3_UP_4_LA_5_CU==5,:)=[];



%corr(Y, 'rows', 'pairwise')
%% 
demo_OPT=innerjoin(d, subjects);
figure;histogram(demo_OPT.rh_entorhinal_thickness)
%demo_OPT([110,214],:)=[]; % two outliers with very low values
demo_OPT( strcmp(demo_OPT.ID, 'WU10014'), :)=[];
CT=demo_OPT{:,[573:640, 644:649]};
ct_names=demo_OPT.Properties.VariableNames([573:640, 644:649]);

Y=[demo_OPT.AIS_01, demo_OPT.IMIS_01, demo_OPT.MDMIS_01, demo_OPT.LIS_01, demo_OPT.MVCIS_01, demo_OPT.EXEC_01];
figure; for i=1:6;     subplot(2,3,i);histogram(Y(:,i)); end

%% AFTER ComBatH
batch = demo_OPT.SITE_MRI_1_WU_2_UT_3_UP_4_LA_5_CU';
dat = CT';
age = demo_OPT.AGE;
sex = demo_OPT.GENDER;
sex = dummyvar(sex);
mod = [demo_OPT.RACE==1 demo_OPT.HL age sex(:,2)];
%part_harmonized = combat(dat, batch, mod, 1);part_harmonized=part_harmonized';
%dat = full_icad25';
full_harmonized = combat(dat, batch, mod, 1);
full_harmonized=full_harmonized';
[rval,pval]=corr(full_harmonized, demo_OPT.RSR_Z_01, 'rows', 'pairwise'); %figure; scatter(full_harmonized(:,191), demo_OPT.RSM_Z)
%demo_OPT{:,[573:640]}=full_harmonized;
[rval,pval]=corr(full_harmonized, demo_OPT.MDMIS_01, 'rows', 'pairwise'); %figure; scatter(full_harmonized(:,191), demo_OPT.RSM_Z)
max(rval)
min(rval)
ct_names(pval<0.01)

%% PLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y=[demo_OPT.AIS_01, demo_OPT.IMIS_01, demo_OPT.MDMIS_01, demo_OPT.LIS_01, demo_OPT.MVCIS_01, demo_OPT.EXEC_01];

X=full_harmonized; %X=CT;
%%% remove nans
naninx=sum(isnan(X)')'>0| sum(isnan(Y)')'>0 |demo_OPT.qc==1; Y=Y(naninx==0,:);  X=X(naninx==0,:);  d_OPT=demo_OPT(naninx==0,:);
%clear x; for i=1:length(X(1,:)); mdl = fitlm(table(d_OPT.AGE, d_OPT.GENDER, d_OPT.HL,d_OPT.RACE, X(:,i)) ); x(:,i)=mdl.Residuals.Raw;end; X=x;
Y=zscore(Y); X=zscore(X);x=X;
ncomp=3
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,Y,ncomp);PCTVAR

permutations=5000;  
allobservations=Y;
for ncomp=1:3
   
    parfor n = 1:permutations
    % selecting either next combination, or random permutation
    permutation_index = randperm(length(allobservations));
    % creating random sample based on permutation index
    randomSample = allobservations(permutation_index,:);
    % running the PLS for this permutation
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,randomSample,ncomp);
    Rsq(n) = sum(PCTVAR(2,:));
    Rsq1(n) = sum(PCTVAR(1,:));  %if ncomp==4; c_perm_pls2(n,:)=corr(XS(:,2), Y);c_perm_pls1(n,:)=corr(XS(:,1), Y);c_perm_pls3(n,:)=corr(XS(:,3), Y);end
    end
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,Y,ncomp);
    p(ncomp)=sum(sum(PCTVAR(2,:))<Rsq')/permutations
    p_1(ncomp)=sum(sum(PCTVAR(1,:))<Rsq1')/permutations
end

[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,Y,ncomp);PCTVAR
[c,pval]=corr(XS, Y); %c(1,:)=c(1,:)*-1; %c(3,:)=c(3,:)*-1; % c(2,:)=c(2,:)*-1;   reshape(mafdr(reshape(pval, [1 36]), 'BHFDR', 'True'), [4 9])
figure;imagesc(c(1:3,:)); colormap bone; colorbar; xticklabels([{'Attn'}, {'MemI'}, {'MemD'}, {'Lang'}, {'Vis'}, {'Exec'}]);
l=length(Y(1,:)); combs=allcomb([1:3], [1:l]);figure(13);
for i=1:length(combs)
hold off; ix1=combs(i,1);ix2=combs(i,2); subplot(3,l,i);
scatter(XS(d_OPT.CDRSCORE==0,ix1), Y(d_OPT.CDRSCORE==0,ix2), 5,'b', 'filled');
hold on;scatter(XS(d_OPT.CDRSCORE>0,ix1), Y(d_OPT.CDRSCORE>0,ix2), 5, 'r', 'filled')
mdl= fitlm(XS(:, ix1), Y(:,ix2)); [ypred,yci] = predict(mdl,XS(:, ix1), 'Alpha',0.001); hold on
yci=[yci,XS(:, ix1)]; yci=sortrows(yci, 3);
plot(XS(:, ix1), ypred, 'k', 'LineWidth', 2);
plot(yci(:,3), yci(:,1), 'k', 'LineWidth', 0.5);
plot(yci(:,3), yci(:,2), 'k', 'LineWidth', 0.5); xlim([-.22 .22]);
text(-0.2, -2, strcat('R=',num2str(round(c(ix1, ix2),3))) ); %annotation('textbox', [0.5, 0.2, 0.1, 0.1], 'string', num2str(c(ix1, ix2)))
end; clear mdl ypred yci mdl ix1 ix2 combs c

figure; scatter(XS(:,1), YS(:,1), 'filled', 'k');lsline;
figure; scatter(XS(:,2), YS(:,2), 'filled', 'k');lsline;
figure; scatter(XS(:,3), YS(:,3), 'filled', 'k');lsline;
corr(XS(:,1), YS(:,1))
corr(XS(:,2), YS(:,2))
corr(XS(:,3), YS(:,3))
tmp=sum(PCTVAR');
figure; histogram(Rsq); xline(tmp(2)); clear tmp

%% bootstrapping to get the func connectivity weights for PLS1, 2 and 3
dim=3
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,Y,dim);PCTVAR
%PLS4w=stats.W(:,4);
PLS3w=stats.W(:,3);
PLS2w=stats.W(:,2);
PLS1w=stats.W(:,1);
bootnum=5000;
PLS1weights=[];
PLS2weights=[];
PLS3weights=[];
PLS4weights=[];
parfor i=1:bootnum
    i;
    myresample = randsample(size(x,1),size(x,1),1);
    res(i,:)=myresample; %store resampling out of interest
    Xr=x(myresample,:); % define X for resampled subjects
    Yr=Y(myresample,:); % define X for resampled subjects
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(Xr,Yr,dim); %perform PLS for resampled data
     
    newW=stats.W(:,3);%extract PLS2 weights
    if corr(PLS3w,newW)<0 % the sign of PLS components is arbitrary - make sure this aligns between runs
        newW=-1*newW;
    end
    PLS3weights=[PLS3weights,newW]; %store (ordered) weights from this bootstrap run    
   
    newW=stats.W(:,2);%extract PLS2 weights
    if corr(PLS2w,newW)<0 % the sign of PLS components is arbitrary - make sure this aligns between runs
        newW=-1*newW;
    end
    PLS2weights=[PLS2weights,newW];
   
    newW=stats.W(:,1);%extract PLS2 weights
    if corr(PLS1w,newW)<0 % the sign of PLS components is arbitrary - make sure this aligns between runs
        newW=-1*newW;
    end
    PLS1weights=[PLS1weights,newW];
   
    %newW=stats.W(:,4);%extract PLS2 weights
    %if corr(PLS4w,newW)<0 % the sign of PLS components is arbitrary - make sure this aligns between runs
    %    newW=-1*newW;
    %end
    %PLS4weights=[PLS4weights,newW];
end
%PLS4sw=std(PLS4weights');
%plsweights4=PLS4w./PLS4sw';
PLS3sw=std(PLS3weights');
plsweights3=PLS3w./PLS3sw';
PLS2sw=std(PLS2weights');
plsweights2=PLS2w./PLS2sw';
PLS1sw=std(PLS1weights');
plsweights1=PLS1w./PLS1sw';

ct_names(plsweights2>3)
ct_names(plsweights2<-3)

ct_names(plsweights3>3)
ct_names(plsweights3<-3)

ct_names(plsweights1>3)'
ct_names(plsweights1<-3)'

corr(demo_OPT.rh_lateralorbitofrontal_thickness, demo_OPT.MDMIS_01, 'rows','pairwise')
corr(demo_OPT.lh_superiorfrontal_thickness, demo_OPT.MDMIS_01, 'rows','pairwise')
corr(demo_OPT.rh_transversetemporal_thickness, demo_OPT.MDMIS_01, 'rows','pairwise')
corr(demo_OPT.lh_superiortemporal_thickness, demo_OPT.MDMIS_01, 'rows','pairwise')

corr(demo_OPT.lh_superiorfrontal_thickness, demo_OPT.LIS_01, 'rows','pairwise')

%% hold out results
[~,scores]=pca(Y(:, pval(2,:)<0.05/36)); 
Ycomb(:,1)=scores(:,1);%Ycomb=Y(:, pval(2,:)<0.05/24);
%[~,scores]=pca(Y(:,9:12));Ycomb(:,2)=scores(:,1);

clear accuracy*
%Ycomb(:,1)=Y(:,3);Ycomb(:,1)=Y(:,4);

l=length(X(:,1)); 
%randomsample=randperm(l); X=X(randomsample,:);Y=Y(randomsample,:);
l=round(0.2*(l),0); 
%for h=1:5 %ixtest=d_OPT.SITE_MRI_1_WU_2_UT_3_UP_4_LA_5_CU==h;
figure(11); hold off

y_pred_all=[];
for h=1:4; 
xstart=(h-1)*l+1;
xend=h*l;if (h==5);xend=length(X(:,1));end
ixtest=zeros(1,length(X(:,1)));
ixtest(xstart:xend)=1; 

ixtest=d_OPT.SITE_MRI_1_WU_2_UT_3_UP_4_LA_5_CU==h;
X_train=x(ixtest==0,:); X_test=x(ixtest==1 ,:);
Y_train=Ycomb(ixtest==0,:); Y_test=Ycomb(ixtest==1 ,:);
%l=length(X(:,1)); l=round(0.25*l,0); X_test=x(1:l,:); X_train=x(l+1:length(X(:,1)) ,:);Y_test=Ycomb(1:l,:); Y_train=Ycomb(l+1:length(X(:,1)) ,:);
dim=3;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X_train,Y_train,dim);PCTVAR;
r=corr(XS, Y_train); accuracytrain(h,:)=r(2,:);
y_pred = [ones(size(X_test ,1),1) X_test]*BETA;
r=corr(Y_test, y_pred);accuracyholdout(h,:)=r(eye(length(Ycomb(1,:)))==1)';y_pred_all=[y_pred_all;y_pred];
hold off; subplot(4,1,h); scatter(y_pred, Y_test, 'x', 'k'); p = polyfit(y_pred,Y_test,1); pred = polyval(p,y_pred); hold on; plot(y_pred,pred,'r','LineWidth',3); set(gca,'xtick',[]); set(gca,'ytick',[]); 
end
%tmp=b(~isnan(b));figure; scatter(y_pred_all,tmp'*30, 15, 'filled', 'k');lsline; corr(tmp', y_pred_all)
accuracyholdout

%%%%% holdout by GE/prisma scanner
[~,scores]=pca(Y);%(:, pval(2,:)<0.05/24)); 
Ycomb(:,1)=scores(:,1);%Ycomb=Y(:, pval(2,:)<0.05/24);
%[~,scores]=pca(Y(:,9:12));Ycomb(:,2)=scores(:,1);
clear accuracy*
ix=d_OPT.scanner==0; %permutation_index = randperm(length(Ycomb));ix=zeros([length(Ycomb), 1]); ix(permutation_index(1:150))=1;
X_train=x(ix==0,:); X_test=x(ix==1 ,:);
Y_train=Ycomb(ix==0,:); Y_test=Ycomb(ix==1 ,:);
dim=2;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X_train,Y_train,dim);PCTVAR;
r=corr(XS, Y_train)
y_pred = [ones(size(X_test ,1),1) X_test]*BETA;
r=corr(Y_test, y_pred)
figure; scatter(y_pred, Y_test, 'x', 'k'); p = polyfit(y_pred,Y_test,1); pred = polyval(p,y_pred); hold on; plot(y_pred,pred,'r','LineWidth',3); set(gca,'xtick',[]); set(gca,'ytick',[]); 

%%
edu=readtable('C:\Users\peter\Documents\OPT\OPT\data\OPTIMUMMainDatabaseF_DATA_2023-02-01_PZedit.csv');
subs=table(d_OPT.ID, 'VariableNames',{'ID'});
edu=innerjoin(subs, edu);edu=outerjoin(subs, edu);

[rr,pp]=corr(d_OPT.ED, YS, 'rows', 'pairwise')
[rr,pp]=corr(d_OPT.ED, XS,'rows', 'pairwise')
figure; scatter(d_OPT.ED, YS(:,2), 15, 'k', 'filled'); lsline;xlim([7 21])
figure; scatter(d_OPT.ED, XS(:,2), 15, 'k', 'filled'); lsline;xlim([7 21])

figure; scatter(d_OPT.ED, YS(:,1), 15, 'k', 'filled'); lsline; xlim([7 21])

%%
madrs=readtable('C:\Users\peter\Documents\OPT\OPT\data\MADRS_Longitudinally_20230711_updated');
subs=table(d_OPT.ID, 'VariableNames',{'ID'});subs.NPDateDays=d_OPT.NPDateDaysKaylaUPDATE;
%madrs=outerjoin(subs, madrs);

for i=1:length(subs.ID);try
    sub=subs.ID{i};
    data=madrs(strcmp(madrs.ID, sub),:);
    MADRS_BL(i)=data.madrs_tot_scr(1);
    data.DaysDiff=data.MADRSDateDays-d_OPT.NPDateDaysKaylaUPDATE(strcmp(d_OPT.ID, sub));
    ix=abs(data.DaysDiff)==min(abs(data.DaysDiff));
    MADRS_NP(i)=data.madrs_tot_scr(ix);MADRS_DateDiffNP(i)=data.DaysDiff(ix);%MADRS_TRUE_BL(i)=(ix(1)==1);
    startstudydate=min(min([data.start_step1days, data.start_step2days]));
    MADRS_length_studyNP(i)=data.MADRSDateDays(ix)-startstudydate;MADRS_TRUE_BL(i)=(MADRS_length_studyNP(i))<42; 
    catch
    MADRS_NP(i)=NaN;MADRS_BL(i)=NaN;MADRS_DateDiff(i)=NaN;MADRS_length_studyNP(i)=NaN;
end; end

MADRS_NP(abs(MADRS_DateDiffNP)>60)=NaN;MADRS_TRUE_BL=double(MADRS_TRUE_BL);MADRS_TRUE_BL(isnan(MADRS_NP))=NaN;

figure; histogram((MADRS_BL-MADRS_NP)./MADRS_BL);
figure; histogram(MADRS_NP);sum(MADRS_NP<11)

responders=((MADRS_BL-MADRS_NP)./MADRS_BL>0.5);
sum(isnan(MADRS_NP))
sum(MADRS_NP<11)
MADRS_groups=ones([length(subs.ID),1]);MADRS_groups(MADRS_groups==1)=NaN;
MADRS_groups(MADRS_TRUE_BL==1 &  ~isnan(MADRS_NP) )=1;
MADRS_groups(MADRS_TRUE_BL==0 &  MADRS_NP<11)=0; % responders==1
MADRS_groups(MADRS_TRUE_BL==0 &  MADRS_NP>=11)=2;
[p,t,stats]=anova1(YS(:,1), MADRS_groups); [c,m,h,gnames] = multcompare(stats, 'CriticalValueType','hsd');
[p,t,stats]=anova1(YS(:,2), MADRS_groups); [c,m,h,gnames] = multcompare(stats, 'CriticalValueType','hsd');
figure; violinplot(YS(:,1), MADRS_groups);xlim([0.5 3.5])
figure; violinplot(YS(:,2), MADRS_groups);xlim([0.5 3.5])
gscatter(d_OPT.AGE, YS(:,2), MADRS_groups);lsline




%%
%%
%%
%% import madrs data
madrs=readtable('C:\Users\peter\Documents\OPT\OPT\data\OPT N MADRS Longitudinally with Study Meds 20230921');madrs(strcmp(madrs.ID, 'UP10209'),:)=[];
%madrs=readtable('C:\Users\peter\Documents\OPT\OPT\data\MADRS_Longitudinally_20230711_updated');madrs(strcmp(madrs.ID, 'UP10209'),:)=[];
subs=table(d.ID, 'VariableNames',{'ID'});subs.NPDateDays=d.NPDateDaysKaylaUPDATE;
%madrs=outerjoin(subs, madrs);

figure;dayss=[];madrs_cut=table;
uniqueMADRSIDS=unique(madrs.ID);
for i=1:length(uniqueMADRSIDS);
    try
    sub=uniqueMADRSIDS{i};
    data=madrs(strcmp(madrs.ID, sub),:);
    data.baseline_madrs_scr(1:length(data.ID))=data.madrs_tot_scr(1);
    remitted(i)=data.madrs_tot_scr(data.MADRSDateDays==max(data.MADRSDateDays))>=11;
    if data.madrs_tot_scr(data.MADRSDateDays==max(data.MADRSDateDays))>=11
    plot(data.MADRSDateDays-data.MADRSDateDays(1), data.madrs_tot_scr,'k'); hold on;
    elseif data.madrs_tot_scr(data.MADRSDateDays==max(data.MADRSDateDays))<11
    plot(data.MADRSDateDays-data.MADRSDateDays(1), data.madrs_tot_scr,'r'); hold on;
    [b(i),rint] =regress(data.madrs_tot_scr, data.MADRSDateDays-data.MADRSDateDays(1));
    end
    dayss=[dayss;data.MADRSDateDays-data.MADRSDateDays(1)];
    madrs_cut=vertcat(madrs_cut,data);
    %missing_FU=sum(~cellfun(@isempty, strfind(data.redcap_event_name, 'week') ));
catch; b(i)=NaN; end; end %madrs_cut.days_bl=days;madrs_cut.Properties.VariableNames{1} = 'ID';

madrs_cut.time_in_d=double(dayss); madrs_cut=outerjoin(madrs_cut, d_OPT);madrs_cut(isnan(madrs_cut.ID_original), :)=[];
madrs_cut.CDRSCORE_01( madrs_cut.CDRSCORE_01>1.5)=NaN;madrs_cut.CDRSUMBOX_01( madrs_cut.CDRSUMBOX_01>=2)=NaN;

madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'month'))) ;madrs_cut(~madrs_cut.remission==0,:)=[];
madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'mth'))) ;madrs_cut(~madrs_cut.remission==0,:)=[];
madrs_cut.remission=madrs_cut.madrs_tot_scr<=10; %madrs_cut(isnan(madrs_cut.madrs_tot_scr),:)=[];

%% defining treatment response via change in MADRS (point change) - LME slopes
mdl=fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d*MDMIS_01+(1+time_in_d|ID_madrs_cut)' )
mdl=fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d*EXEC_01+(1+time_in_d|ID_madrs_cut)' )

violinplot(madrs_cut.madrs_tot_scr, madrs_cut.CDRSCORE)
figure;scatter(madrs_cut.RSM_Z, madrs_cut.madrs_tot_scr); lsline

fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d*CDRSCORE_01+(1+time_in_d|ID_madrs_cut)' )
fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d*CDRSUMBOX_01+(1+time_in_d|ID_madrs_cut)' )
fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d*DTMTS4_01+(1+time_in_d|ID_madrs_cut)' )
fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d*rh_MeanThickness_thickness+(1+time_in_d|ID_madrs_cut)' )


for i=1:74
mdlterm=strcat('madrs_tot_scr~AGE+GENDER+time_in_d*', ct_names{i},'+(1+time_in_d|ID_madrs_cut)');
mdl=fitlme(madrs_cut, mdlterm );
pval(i)=mdl.Coefficients.pValue(6);   
rval(i)=mdl.Coefficients.tStat(6);   
end
ct_names(bhfdr(pval)<0.05)'
rval(bhfdr(pval)<0.1)'
pval(bhfdr(pval)<0.1)'


uniqueMADRSIDS=unique(madrs_cut.ID_madrs_cut);uniqueMADRSIDS(1)=[]; b=[];bl_madrs=[];
for i=1:length(uniqueMADRSIDS);
    try
    sub=uniqueMADRSIDS{i};
    data=madrs_cut(strcmp(madrs_cut.ID_madrs_cut, sub),:);
    bl_madrs(i)=data.madrs_tot_scr(1);
    b(i)=data.madrs_tot_scr(1)-data.madrs_tot_scr(length(data.madrs_tot_scr)); %b(i)=b(i)/data.time_in_d(length(data.madrs_tot_scr));
catch; b(i)=NaN; end; end %madrs_cut.days_bl=days;madrs_cut.Properties.VariableNames{1} = 'ID';

visualtable=table();
visualtable.ID=uniqueMADRSIDS;
visualtable=innerjoin(visualtable, d_OPT);

delta_days=madrs_cut.MADRSDateDays-madrs_cut.NPDateDaysKaylaUPDATE;%madrs_cut.baseline_madrs_scr(delta_days<-60)=NaN;
[r,p]=partialcorr(visualtable.lh_postcentral_thickness, b', [visualtable.AGE, visualtable.GENDER],   'rows', 'pairwise')
[r,p]=partialcorr(visualtable{:,[573:640, 644:649]}, b', [visualtable.AGE, visualtable.GENDER],   'rows', 'pairwise');
ct_names(p<0.05)
%[r,p]=partialcorr(log(visualtable.WMH), b', [visualtable.AGE, visualtable.GENDER],   'rows', 'pairwise')
%scatter(visualtable.DTMT4_Scaled, 30*b'); lsline
figure;subplot(1,3,2);scatter(visualtable.lh_postcentral_thickness, b', 10,'filled','k'); lsline
[r,p]=partialcorr(bl_madrs', b', [visualtable.AGE, visualtable.GENDER],   'rows', 'pairwise')
subplot(1,3,1);scatter(bl_madrs, b', 10,'filled','k'); lsline
[r,p]=partialcorr(visualtable.lh_insula_thickness, b', [visualtable.AGE, visualtable.GENDER],   'rows', 'pairwise')
subplot(1,3,3);scatter(visualtable.lh_insula_thickness, b', 10,'filled','k'); lsline
figure;
[r,p]=partialcorr(visualtable.MDMIS_01, b', [visualtable.AGE, visualtable.GENDER],   'rows', 'pairwise')
subplot(1,3,1);scatter(visualtable.MDMIS_01, b', 10,'filled','k'); lsline
[r,p]=partialcorr(visualtable.EXEC_01, b', [visualtable.AGE, visualtable.GENDER],   'rows', 'pairwise')
subplot(1,3,2);scatter(visualtable.EXEC_01, b', 10,'filled','k'); lsline
[r,p]=partialcorr(visualtable.rh_amygdala, b', [visualtable.AGE, visualtable.GENDER],   'rows', 'pairwise')
subplot(1,3,3);scatter(visualtable.rh_amygdala, b', 10,'filled','k'); lsline
[r,p]=partialcorr(visualtable.AIS_01, b', [visualtable.AGE, visualtable.GENDER],   'rows', 'pairwise')
subplot(1,3,3);scatter(visualtable.AIS_01, b', 10,'filled','k'); lsline


visualtable.CDRSCORE( visualtable.CDRSCORE>1.5)=NaN;visualtable.CDRSUMBOX( visualtable.CDRSUMBOX>=2)=NaN;visualtable.CDRSCORE(visualtable.CDRSCORE==0.0500)=0.5; visualtable.CDRSCORE(visualtable.CDRSCORE==-1)=0; 
anova1(b, visualtable.CDRSCORE_01)
figure; violinplot(b*30, visualtable.CDRSCORE)
%% plsregression with change in scores by participant
mdlterm=strcat('madrs_tot_scr~AGE+GENDER+time_in_d+(1+time_in_d|ID_madrs_cut)');
mdl=fitlme(madrs_cut, mdlterm );
B = randomEffects(mdl); corr(b', -1*B(2:2:length(B)), 'rows','pairwise')

Y=b';%Y=-1*B(2:2:512);
x=[visualtable{:,[573:640, 644:649]}, visualtable.MDMIS_01, visualtable.EXEC_01,visualtable.AIS_01, bl_madrs'];
naninx=sum(isnan(x)')'>0| isnan(Y)>0; Y=Y(naninx==0,:);  X=x(naninx==0,:);  visualtable=visualtable(naninx==0,:);  
Y=zscore(Y); X=zscore(X);x=X;
ncomp=3
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,Y,ncomp);PCTVAR

permutations=5000;  
allobservations=Y;
for ncomp=1:3
   
    parfor n = 1:permutations
    % selecting either next combination, or random permutation
    permutation_index = randperm(length(allobservations));
    % creating random sample based on permutation index
    randomSample = allobservations(permutation_index,:);
    % running the PLS for this permutation
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,randomSample,ncomp);
    Rsq(n) = sum(PCTVAR(2,:));
    Rsq1(n) = sum(PCTVAR(1,:));  %if ncomp==4; c_perm_pls2(n,:)=corr(XS(:,2), Y);c_perm_pls1(n,:)=corr(XS(:,1), Y);c_perm_pls3(n,:)=corr(XS(:,3), Y);end
    end
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,Y,ncomp);
    p(ncomp)=sum(sum(PCTVAR(2,:))<Rsq')/permutations
    p_1(ncomp)=sum(sum(PCTVAR(1,:))<Rsq1')/permutations
end


dim=3
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,Y,dim);PCTVAR
%PLS4w=stats.W(:,4);
PLS3w=stats.W(:,3);
PLS2w=stats.W(:,2);
PLS1w=stats.W(:,1);
bootnum=5000;
PLS1weights=[];
PLS2weights=[];
PLS3weights=[];
PLS4weights=[];
clear res 
parfor i=1:bootnum
    i;
    myresample = randsample(size(x,1),size(x,1),1);
    res(i,:)=myresample; %store resampling out of interest
    Xr=x(myresample,:); % define X for resampled subjects
    Yr=Y(myresample,:); % define X for resampled subjects
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(Xr,Yr,dim); %perform PLS for resampled data
     
    newW=stats.W(:,3);%extract PLS2 weights
    if corr(PLS3w,newW)<0 % the sign of PLS components is arbitrary - make sure this aligns between runs
        newW=-1*newW;
    end
    PLS3weights=[PLS3weights,newW]; %store (ordered) weights from this bootstrap run    
   
    newW=stats.W(:,2);%extract PLS2 weights
    if corr(PLS2w,newW)<0 % the sign of PLS components is arbitrary - make sure this aligns between runs
        newW=-1*newW;
    end
    PLS2weights=[PLS2weights,newW];
   
    newW=stats.W(:,1);%extract PLS2 weights
    if corr(PLS1w,newW)<0 % the sign of PLS components is arbitrary - make sure this aligns between runs
        newW=-1*newW;
    end
    PLS1weights=[PLS1weights,newW];
   
    %newW=stats.W(:,4);%extract PLS2 weights
    %if corr(PLS4w,newW)<0 % the sign of PLS components is arbitrary - make sure this aligns between runs
    %    newW=-1*newW;
    %end
    %PLS4weights=[PLS4weights,newW];
end
%PLS4sw=std(PLS4weights');
%plsweights4=PLS4w./PLS4sw';
PLS3sw=std(PLS3weights');
plsweights3=PLS3w./PLS3sw';
PLS2sw=std(PLS2weights');
plsweights2=PLS2w./PLS2sw';
PLS1sw=std(PLS1weights');
plsweights1=PLS1w./PLS1sw';

varnames=[ct_names, 'delayed mem', 'exec', 'attn','bl_madrs'];%, 'tmt5','rsm','rsr'
varnames(plsweights2>3)
varnames(plsweights3>3)

varnames(plsweights2<-3)
varnames(plsweights3<-3)

y_pred = [ones(size(x ,1),1) x]*BETA;
r=corr(Y, y_pred)
tmp=b(~isnan(b));figure(6); scatter(y_pred,tmp', 15, 'filled', 'k');lsline; corr(tmp', y_pred) %figure;scatter(y_pred, Y, '.', 'k');lsline
figure;scatter(XS(:,2),Y);lsline; corr(XS(:,3),Y)

l=length(X(:,1)); l=round(0.2*(l),0); figure(11); hold off

y_pred_all=[];y_test_all=[];
for h=1:4
xstart=(h-1)*l+1;
xend=h*l;if (h==5);xend=length(X(:,1));end
ixtest=zeros(1,length(X(:,1)));
ixtest(xstart:xend)=1; 

ixtest=visualtable.SITE_MRI_1_WU_2_UT_3_UP_4_LA_5_CU==h;
X_train=x(ixtest==0,:); X_test=x(ixtest==1 ,:);
Y_train=Y(ixtest==0); Y_test=Y(ixtest==1);
%l=length(X(:,1)); l=round(0.25*l,0); X_test=x(1:l,:); X_train=x(l+1:length(X(:,1)) ,:);Y_test=Ycomb(1:l,:); Y_train=Ycomb(l+1:length(X(:,1)) ,:);
dim=3;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X_train,Y_train,dim);PCTVAR;
r=corr(XS, Y_train); accuracytrain(h,:)=r(2,:);
y_pred = [ones(size(X_test ,1),1) X_test]*BETA;
r=corr(Y_test, y_pred);accuracyholdout(h,:)=r;y_pred_all=[y_pred_all;y_pred]; y_test_all=[y_test_all;Y_test];
subplot(4,1,h); scatter(y_pred, Y_test, 'x', 'k'); p = polyfit(y_pred,Y_test,1); pred = polyval(p,y_pred); hold on; plot(y_pred,pred,'r','LineWidth',3); set(gca,'xtick',[]); set(gca,'ytick',[]); 
end
%tmp=b(~isnan(b));figure; scatter(y_pred_all,tmp'*30, 15, 'filled', 'k');lsline; corr(tmp', y_pred_all)
accuracyholdout

r=corr(y_test_all, y_pred_all)
figure; scatter(y_pred_all, y_test_all,15, 'filled', 'k'); p = polyfit(y_pred_all,y_test_all,1); pred = polyval(p,y_pred_all); hold on; plot(y_pred_all,pred,'r','LineWidth',3); %set(gca,'xtick',[]); set(gca,'ytick',[]); 

%% defining treatment response via categorical variables (MADRS<10) - step 1
%% BINARY LOGISTIC REG
madrs_cut.madrs_tot_scr( ~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'end'))) );
madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'end'))) ;
madrs_cut(madrs_cut.remission==0,:)=[];
madrs_cut.remission=madrs_cut.madrs_tot_scr<=10; %arm 7 = step 2; arm 8 = step 1
%madrs_cut(madrs_cut.OnAripiprazole~=1,:)=[];%madrs_cut(madrs_cut.OnBupropion~=1,:)=[];%madrs_cut(madrs_cut.OnNortriptyline~=1,:)=[];%
%madrs_cut(~(madrs_cut.OnAripiprazole==1| madrs_cut.OnBupropion==1),:)=[]; % madrs_cut(madrs_cut.remission==0,:)=[]; madrs_cut.remission=madrs_cut.OnAripiprazole==1;
%madrs_cut(madrs_cut.remission==1& ~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'step_1'))),:)=[]; madrs_cut(madrs_cut.remission==1& ~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'arm_8'))),:)=[];

madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'step_2'))) ;
madrs_cut(madrs_cut.remission==1,:)=[]; 
madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'arm_7'))) ;
madrs_cut(madrs_cut.remission==1,:)=[]; 
madrs_cut.remission=madrs_cut.madrs_tot_scr<=10;

madrs_cut.NC_MCI_DEM=zeros([length(madrs_cut.GENDER),1]);madrs_cut.NC_MCI_DEM(madrs_cut.MCI_01==1)=1; madrs_cut.NC_MCI_DEM(madrs_cut.DEM_01==1)=2; 
delta_days=madrs_cut.MADRSDateDays-madrs_cut.NPDateDaysKaylaUPDATE;%madrs_cut(delta_days<-60,:)=[]; %madrs_cut.baseline_madrs_scr(delta_days<-60)=NaN;%baseline_madrs_scr
model = fitglm(madrs_cut,['remission ~RC_Z+DTMT4_Scaled+lh_postcentral_thickness' ...
    '+lh_rostralanteriorcingulate_thickness+rh_transversetemporal_thickness+rh_bankssts_thickness+lh_bankssts_thickness'...
    '+ baseline_madrs_scr+AGE+GENDER+ED'],'link','logit','Distribution','binomial') %lme4/nlme 
%RSM_Z+RSR_Z+RC_Z+DCSSS+DCFTCSS+DTMT4_Scaled+DTMT5_Scaled+CWI4CSSfinal+CWI3CSSfinal+
scores = model.Fitted.Probability; [X,Y,T,AUC] = perfcurve(madrs_cut.remission,scores, 1);AUC %confusionchart(Ylogist,scores>0.4)
figure(2);hold on;plot(X,Y,'Color',[76/255 0 153/255])
model = fitglm(madrs_cut,['remission ~lh_postcentral_thickness+lh_rostralanteriorcingulate_thickness+rh_transversetemporal_thickness+rh_bankssts_thickness+lh_bankssts_thickness+ baseline_madrs_scr+AGE+GENDER+ED'],'link','logit','Distribution','binomial') ;
scores = model.Fitted.Probability; [X,Y,T,AUC] = perfcurve(madrs_cut.remission,scores, 1); figure(2);hold on;plot(X,Y,'Color',[204/255 0 204/255]); AUC
model = fitglm(madrs_cut,['remission ~baseline_madrs_scr+AGE+GENDER+ED'],'link','logit','Distribution','binomial') ;
scores = model.Fitted.Probability; [X,Y,T,AUC] = perfcurve(madrs_cut.remission,scores, 1); figure(2);hold on;plot(X,Y,'Color',[255/255 0 127/255]); AUC

Xlogist=[madrs_cut{:,[595:662, 666:671]}, madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr, madrs_cut.MDMIS_01, madrs_cut.EXEC_01, madrs_cut.AIS_01]; 
%anova1(madrs_cut.EXEC_01, madrs_cut.remission);anova1(madrs_cut.baseline_madrs_scr, madrs_cut.remission);
varnames=[ct_names, {'age'}, {'sex'},{'blmadrs'}, {'MemD'}, {'Exec'}, {'Attn'}];
Ylogist=madrs_cut.remission;

[B,FitInfo] = lassoglm(Xlogist,Ylogist,'binomial','CV',10, 'PredictorNames', varnames,'alpha', 0.1);
%lassoPlot(B,FitInfo,'PlotType','CV'); legend('show','Location','best') % show legend
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
MinModelPredictors = FitInfo.PredictorNames(B(:,idxLambdaMinDeviance)~=0)
idxLambda1SE = FitInfo.Index1SE;
sparseModelPredictors = FitInfo.PredictorNames(B(:,idxLambda1SE)~=0)
B0 = FitInfo.Intercept(idxLambda1SE);
coef = [B0; B(:,idxLambda1SE)];
yhat = glmval(coef,Xlogist,'logit');[X,Y,T,AUC] = perfcurve(Ylogist,yhat, 1);AUC
%Xlogist=Xlogist(:,B(:,idxLambda1SE)~=0);varnames=varnames(B(:,idxLambda1SE)~=0); %parsimonious model test > set alpha to very low 0.001
%Xlogist=Xlogist(:,tmp<5);varnames=varnames(tmp<5)

%Xlogist=[madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr, madrs_cut.MDMIS_01, madrs_cut.EXEC_01, madrs_cut.AIS_01]; varnames=[{'age'}, {'sex'},{'blmadrs'}, {'MemD'}, {'Exec'}, {'Attn'}];
%Xlogist=[madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr]; varnames=[{'age'}, {'sex'},{'blmadrs'}];
permutation_index = randperm(length(Ylogist));%for randfold=1%:20
figure;coef_all=[];confusionmatrices=[];yhat_all=[];ytest_all=[];yhatBinom_all=[]; n=length(madrs_cut.ID_madrs_cut)
for i=1:100 %8
%   if(i==1); ix=zeros([n,1]); ix(1:20)=1; elseif (i==8)
%   ix=zeros([n,1]); ix(20*(i-1)+1:n)=1; else
%   ix=zeros([n,1]); ix(20*(i-1)+1:20*(i-1)+20)=1; end; ix=ix(permutation_index);
permutation_index = randperm(length(Ylogist));ix=zeros([length(madrs_cut.ID_madrs_cut), 1]);
ix(permutation_index(1:20))=1;  %ix(permutation_index(31:304))=0;  
XTest=Xlogist(ix==1,:);XTrain=Xlogist(ix==0,:); 
yTest=Ylogist(ix==1);yTrain=Ylogist(ix==0);

[B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','CV',10, 'PredictorNames', varnames,'alpha', 0.1);%0.0001 - ideal thresh for best model assessment 
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; B(:,idxLambdaMinDeviance)]; coef_all=[coef_all,coef];
%predicted vs observed
yhat = glmval(coef,XTest,'logit');yhat_all=[yhat_all; yhat];ytest_all=[ytest_all; yTest];
yhatBinom = (yhat>=0.35);yhatBinom_all=[yhatBinom_all; yhatBinom];
if i<11;    subplot(2,5,i); c=confusionchart(yTest,yhatBinom); end
c=confusionchart(yTest,yhatBinom);
confusionmatrices(i,:,:)=c.NormalizedValues;
[X,Y,T,AUC] = perfcurve(yTest,yhat, 1);AUC_test(i)=AUC;
end
%AUC_test
[X,Y,T,AUC] = perfcurve(ytest_all,yhat_all, 1);AUC %AUC_test(randfold)=AUC;
%end
yhatBinom_all=yhat_all>0.29;figure;c=confusionchart(ytest_all,double(yhatBinom_all)) %33 25
[sensitivity, specificity, accuracy, F1score]=gofmeasures_2d_square(c.NormalizedValues)

sensitivity=sum(confusionmatrices(:,2,2))/sum(sum(confusionmatrices(:,2,:)))
specificity=sum(confusionmatrices(:,1,1))/sum(sum(confusionmatrices(:,1,:)))
accuracy=(sum(confusionmatrices(:,1,1))+sum(confusionmatrices(:,2,2)))/sum(sum(sum(confusionmatrices(:,:,:))))
F1score=2*sum(confusionmatrices(:,2,2))/(2*sum(confusionmatrices(:,2,2))+ sum(confusionmatrices(:,2,1))+ sum(confusionmatrices(:,1,2)) )


coef_all(coef_all==0)=NaN;tmp=sum(isnan(coef_all'));coef=coef_all; tmp(1)=[];varnames(tmp<5)%coef(tmp>5,:)=NaN;coef=nanmean(coef')';
%%%%%%%% apply ARI model to BUP and BUP model to ARI
yhat = glmval(coef,Xlogist,'logit'); [~,~,~,AUC] = perfcurve(Ylogist,yhat, 1);AUC %AUC_test(randfold)=AUC;
yhatBinom = (yhat>=0.26); %0.26 for BUP and 0.38 for ARI
c=confusionchart(Ylogist,yhatBinom)
yhat_ARI=load('yhat_val_ARI.mat');yhat_ARI=yhat_ARI.yhat;yhat_BUP=load('yhat_val_BUP.mat');yhat_BUP=yhat_BUP.yhat;
figure; bar(sortrows([yhat_ARI,yhat_BUP], 1), 'stacked' )
yhatBinom_ARI=yhat_ARI>=0.39; c=confusionchart(Ylogist,yhatBinom_ARI)
yhatBinom_BUP=yhat_BUP>=0.28; c=confusionchart(Ylogist,yhatBinom_BUP)
%%%TP1 TN2 FP3 FN4 
%confusion_ARI=[];confusion_ARI(yhatBinom_ARI==1 & Ylogist==1)=1; confusion_ARI(yhatBinom_ARI==0 & Ylogist==0)=2; confusion_ARI(yhatBinom_ARI==1 & Ylogist==0)=3; confusion_ARI(yhatBinom_ARI==0 & Ylogist==1)=4; confusion_BUP=[];confusion_BUP(yhatBinom_BUP==1 & Ylogist==1)=1; confusion_BUP(yhatBinom_BUP==0 & Ylogist==0)=2; confusion_BUP(yhatBinom_BUP==1 & Ylogist==0)=3; confusion_BUP(yhatBinom_BUP==0 & Ylogist==1)=4; tbl=crosstab(confusion_ARI, confusion_BUP)
 
%%%TP1 FP2 TN3 FN4 
confusion_ARI=[];confusion_ARI(yhatBinom_ARI==1 & Ylogist==1)=1; confusion_ARI(yhatBinom_ARI==0 & Ylogist==0)=3;
confusion_ARI(yhatBinom_ARI==1 & Ylogist==0)=2; confusion_ARI(yhatBinom_ARI==0 & Ylogist==1)=4;
confusion_BUP=[];confusion_BUP(yhatBinom_BUP==1 & Ylogist==1)=1; confusion_BUP(yhatBinom_BUP==0 & Ylogist==0)=3;
confusion_BUP(yhatBinom_BUP==1 & Ylogist==0)=2; confusion_BUP(yhatBinom_BUP==0 & Ylogist==1)=4;
tbl=crosstab(confusion_ARI, confusion_BUP)

Ylogist=madrs_cut.remission; Ylogist=zeros(length(Ylogist),1);
Ylogist(madrs_cut.OnBupropion==1 & madrs_cut.madrs_tot_scr<=10)=2;
Ylogist(madrs_cut.OnAripiprazole==1 & madrs_cut.madrs_tot_scr<=10)=3;

yhat_3way=zeros(204,1); yhat_3way(yhat_ARI>0.38)=3;yhat_3way(yhat_BUP>0.28)=2; 
ConfusionMat1 = confusionchart(Ylogist,yhat_3way);


figure; coutofsample=[];yhat_all=[];ytest_all=[];yhatBinom_all=[];
for i=1:100
%if(i==1); ix=zeros([304,1]); ix(1:34)=1; else
%ix=zeros([304,1]); ix(30*(i-1):30*(i-1)+30)=1; end
permutation_index = randperm(length(Ylogist));ix=zeros([length(madrs_cut.ID_madrs_cut), 1]);
ix(permutation_index(1:30))=1;  %ix(permutation_index(31:304))=0;  
XTest=Xlogist(ix==1,:);XTrain=Xlogist(ix==0,:); 
yTest=Ylogist(ix==1);yTrain=Ylogist(ix==0);
%predicted vs observed
yhat = glmval(coef,XTest,'logit');
yhatBinom = (yhat>=0.35);yhat_all=[yhat_all; yhat];ytest_all=[ytest_all; yTest];
if i<11;    subplot(2,5,i); c=confusionchart(yTest,yhatBinom); end
c=confusionchart(yTest,yhatBinom);yhatBinom_all=[yhatBinom_all; yhatBinom];
coutofsample(i,:,:)=c.NormalizedValues;
end
[X,Y,T,AUC] = perfcurve(ytest_all,yhat_all, 1);AUC %AUC_test(randfold)=AUC;
[sensitivity, specificity, accuracy, F1score]=gofmeasures_3d_square(coutofsample)

load('CT_holdoutAUC_rerun_100.mat');figure(2);hold on;plot(X,Y,'Color',[76/255 0 153/255]);AUC_all(1)=AUC
AUC_test_all(3,:)=AUC_test;
load('COGMADRS_only_NoCT_holdoutAUC_rerun100.mat');figure(2);hold on;plot(X,Y,'Color',[204/255 0 204/255]); AUC_all(2)=AUC
AUC_test_all(2,:)=AUC_test;
load('MADRS_onlyNoCT_holdoutAUC_rerun_100.mat');figure(2);hold on;plot(X,Y,'Color',[255/255 0 127/255]); set(gca,'box','off');AUC_all(3)=AUC
AUC_test_all(1,:)=AUC_test; %figure; b = bar(AUC_all,'k');
load('AUC_alpha.001_CTmodel_3.mat'); figure(2);hold on;plot(X,Y,'k');

figure(3); yyaxis right; set(gca, 'color', 'none'); b = bar(mean(AUC_test_all'));
SEM = std(AUC_test_all')/sqrt(100); 
hold on; [ngroups,nbars] = size(mean(AUC_test_all'));
tmp = nan(nbars, ngroups); for i = 1:nbars    tmp(i) = b.XEndPoints(i); end
errorbar(tmp',mean(AUC_test_all'), 2*SEM,'k','linestyle','none');ylim([0.5 0.8]);hold on
figure(3); b = bar([1; 2; 3],diag(mean(AUC_test_all')),'stacked');
b(1).FaceColor=[255/255 0 127/255];b(2).FaceColor=[204/255 0 204/255]; b(3).FaceColor=[76/255 0 153/255];ylim([0.5 0.81])
%%
%% MULTINOMIAL !!! LOGISTIC REG
Xlogist=[madrs_cut{:,[188:255, 259:264]}, madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr, madrs_cut.RC_Z, madrs_cut.DTMT4_Scaled];%, madrs_cut.DTMT5_Scaled, madrs_cut.RSM_Z, madrs_cut.RSR_Z
varnames=[ct_names, {'age'}, {'sex'},{'blmadrs'}, 'coding', 'tmt4'];%, 'tmt5','rsm','rsr'
%Xlogist=[madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr, madrs_cut.RC_Z, madrs_cut.DTMT4_Scaled]; varnames=[{'age'}, {'sex'},{'blmadrs'}, 'coding', 'tmt4'];%, 'tmt5','rsm','rsr'
%Xlogist=[madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr]; varnames=[{'age'}, {'sex'},{'blmadrs'}];%, 'tmt5','rsm','rsr'
Ylogist=madrs_cut.remission; Ylogist=zeros(length(Ylogist),1);
Ylogist(madrs_cut.OnBupropion==1 & madrs_cut.madrs_tot_scr<=10)=2;
Ylogist(madrs_cut.OnAripiprazole==1 & madrs_cut.madrs_tot_scr<=10)=3;
[B,dev,stats] = mnrfit(Xlogist, categorical(Ylogist))
Yprob = mnrval(B,Xlogist);
%imagesc(sortrows(Yprob,1)); %plot(sortrows(Yprob,1));
Ypred=zeros(length(Ylogist),1);
Ypred(Yprob(:,1)>0.28)=0;
Ypred(Yprob(:,2)>0.19)=2;
Ypred(Yprob(:,3)>0.19)=3;
tbl=crosstab(Ylogist, Ypred)
[specificity, recall1, recall2, accuracy]=gofsquare_x3(tbl)
tmp=abs(stats.t(:,1))>1.2 |abs(stats.t(:,2))>1.2;tmp(1)=[]; varnames((tmp)>0)
%tmp=stats.p(:,1)<0.1 |stats.p(:,2)<0.1;tmp(1)=[]; varnames((tmp)>0)
Xlogist(:,tmp==0)=[];


permutation_index = randperm(length(Ylogist));%for randfold=1%:20
coef_all=[];confusionmatrices=[];yhat_all=[];ytest_all=[];yhatBinom_all=[];sumstats=[];sumstats_T=[];
for i=1:10
    if(i==1); ix=zeros([203,1]); ix(1:20)=1; elseif (i==10)
    ix=zeros([203,1]); ix(20*(i-1)+1:203)=1; else
    ix=zeros([203,1]); ix(20*(i-1)+1:20*(i-1)+20)=1; end; ix=ix(permutation_index);
%permutation_index = randperm(length(Ylogist));ix=zeros([length(madrs_cut.ID_madrs_cut), 1]);
%ix(permutation_index(1:30))=1;  %ix(permutation_index(31:304))=0;  
XTest=Xlogist(ix==1,:);XTrain=Xlogist(ix==0,:); 
yTest=Ylogist(ix==1);yTrain=Ylogist(ix==0);

[B,dev,stats] = mnrfit(XTrain, categorical(yTrain)); sumstats(i,:,:)=stats.p;sumstats_T(i,:,:)=stats.t;
%predicted vs observed
yhat = mnrval(B,XTest); yhat_all=[yhat_all; yhat];ytest_all=[ytest_all; yTest];
%Ypred=zeros(length(yTest),1);
%Ypred(yhat(:,1)>0.28)=0; Ypred(yhat(:,2)>0.19)=2; Ypred(yhat(:,3)>0.19)=3;%
%tbl_all(i,:,:)=crosstab(yTest, Ypred);
end
%AUC_test
Ypred=zeros(length(ytest_all),1);
Ypred(yhat_all(:,1)>0.28)=0; Ypred(yhat_all(:,2)>0.14)=2; Ypred(yhat_all(:,3)>0.19)=3;
%Ypred(yhat_all(:,1)>0.28)=0;delta=yhat_all(:,2)-yhat_all(:,3); Ypred(delta>0.09)=2; Ypred(delta<-0.09)=3;
tbl=crosstab(ytest_all, Ypred)
[specificity, recall1, recall2, accuracy]=gofsquare_x3(tbl)

tmp=abs(sumstats_T(:,:,1))>1.5 | abs(sumstats_T(:,:,2))>1.5;tmp(:,1)=[]; imagesc(tmp); varnames(sum(tmp)>1)
Xlogist(:,~(sum(tmp)>1))=[];
%end


%% NAIVE BAYES
Xlogist=[madrs_cut{:,[188:255, 259:264]}, madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr, madrs_cut.RC_Z, madrs_cut.DTMT4_Scaled];%, madrs_cut.DTMT5_Scaled, madrs_cut.RSM_Z, madrs_cut.RSR_Z
varnames=[ct_names, {'age'}, {'sex'},{'blmadrs'}, 'coding', 'tmt4'];%, 'tmt5','rsm','rsr'
%Xlogist=[madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr, madrs_cut.RC_Z, madrs_cut.DTMT4_Scaled]; varnames=[{'age'}, {'sex'},{'blmadrs'}, 'coding', 'tmt4'];%, 'tmt5','rsm','rsr'
%Xlogist=[madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr]; varnames=[{'age'}, {'sex'},{'blmadrs'}];%, 'tmt5','rsm','rsr'
Ylogist=madrs_cut.remission; Ylogist=zeros(length(Ylogist),1);
Ylogist(madrs_cut.OnBupropion==1 & madrs_cut.madrs_tot_scr<=10)=2;
Ylogist(madrs_cut.OnAripiprazole==1 & madrs_cut.madrs_tot_scr<=10)=3; Ylogist=categorical(Ylogist);

Mdl = fitcnb(Xlogist,Ylogist);
[predLabels, Yprob] = resubPredict(Mdl); %resub is out of sample
ConfusionMat1 = confusionchart(Ylogist,predLabels);
[recall1, recall2, recall3, accuracy]=gofsquare_x3(ConfusionMat1.NormalizedValues)
Ypred=zeros(length(Ylogist),1);Ypred(Yprob(:,1)>0.28)=0;
Ypred(Yprob(:,2)>0.30)=2;
Ypred(Yprob(:,3)>0.15)=3;
tbl=crosstab(Ylogist, Ypred)
[recall1, recall1, recall2, accuracy]=gofsquare_x3(tbl)
ConfusionMat1 = confusionchart(Ylogist,categorical(Ypred));
figure; bar(sortrows(Yprob,1), 'stacked'); ylim([0 1]);
% explain shaps values for 1 person
explainer=shapley(Mdl); 
for i=1:n; queryPoint = Xlogist(i,:); explainer = fit(explainer,queryPoint); shapsvals(i,:)=explainer.ShapleyValues.true; end %plot(explainer)

CVMdl1 = fitcnb(Xlogist,Ylogist, 'CrossVal','on');
classErr1 = kfoldLoss(CVMdl1,'LossFun','ClassifErr')

t = templateNaiveBayes();
CVMdl2 = fitcecoc(Xlogist,Ylogist, 'CrossVal','on','Learners',t);
classErr2 = kfoldLoss(CVMdl2,'LossFun','ClassifErr')
isGenRate = resubLoss(Mdl,'LossFun','ClassifErr')

permutation_index = randperm(length(Ylogist));%for randfold=1%:20
coef_all=[];confusionmatrices=[];yhat_all=[];ytest_all=[];yhatBinom_all=[]; n=length(madrs_cut.ID_madrs_cut)
for i=1:6
    if(i==1); ix=zeros([n,1]); ix(1:20)=1; elseif (i==10)
    ix=zeros([n,1]); ix(20*(i-1)+1:n)=1; else
    ix=zeros([n,1]); ix(20*(i-1)+1:20*(i-1)+20)=1; end; ix=ix(permutation_index);
%permutation_index = randperm(length(Ylogist));ix=zeros([length(madrs_cut.ID_madrs_cut), 1]);
%ix(permutation_index(1:20))=1;  %ix(permutation_index(31:304))=0;  
XTest=Xlogist(ix==1,:);XTrain=Xlogist(ix==0,:); 
yTest=Ylogist(ix==1);yTrain=Ylogist(ix==0);
Mdl = fitcnb(XTrain,yTrain);
%oosGenRate(i) = loss(Mdl,XTest,yTest)
end

%%

cvp = cvpartition(Ylogist,"KFold",7);
%cross-validation classification error for neural network classifiers with different regularization strengths. 
% Try regularization strengths on the order of 1/n, where n is the number of observations. 
% Specify to standardize the data before training the neural network models.

lambda = (0:0.5:5)*1e-4;
cvloss = zeros(length(lambda),1);

for i = 1:length(lambda)
    cvMdl = fitcnet(Xlogist, Ylogist,"Lambda",lambda(i), ...
        "CVPartition",cvp,"Standardize",true);
    cvloss(i) = kfoldLoss(cvMdl,"LossFun","classiferror");
end
plot(lambda,cvloss)
xlabel("Regularization Strength")
ylabel("Cross-Validation Loss")
[~,idx] = min(cvloss);
bestLambda = lambda(idx)

c = cvpartition(Ylogist,"Holdout",0.20);
trainingIndices = training(c); % Indices for the training set
testIndices = test(c); % Indices for the test set
Ytrain= Ylogist(trainingIndices);Xtrain= Xlogist(trainingIndices);
Ytest= Ylogist(testIndices);Xtest= Xlogist(testIndices,:);
cvMdl = fitcnet(Xtrain, Ytrain,"Lambda",bestLambda)
Mdl = fitcnet(Xtest, Ytest);
testAccuracy = 1 - loss(cvMdl,Xtest,Ytest, "LossFun","classiferror")
figure; confusionchart(Ytest,predict(cvMdl,Xtest))

%%
%% predicting 1/0 response in step 2
%%
madrs_cut.madrs_tot_scr( ~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'end'))) )
madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'end'))) ;
madrs_cut(madrs_cut.remission==0,:)=[];
madrs_cut.remission=madrs_cut.madrs_tot_scr<=10;

madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'step_1'))) ;
madrs_cut(madrs_cut.remission==1,:)=[];
madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'arm_8'))) ;
madrs_cut(madrs_cut.remission==1,:)=[];
madrs_cut.remission=madrs_cut.madrs_tot_scr<=10;


delta_days=madrs_cut.MADRSDateDays-madrs_cut.NPDateDaysKaylaUPDATE;%madrs_cut.baseline_madrs_scr(delta_days<-60)=NaN;
model = fitglm(madrs_cut,['remission ~RC_Z+DTMT4_Scaled+lh_postcentral_thickness' ...
    '+lh_cuneus_thickness+lh_superiortemporal_thickness+rh_cuneus_thickness+rh_lingual_thickness'...
    '+ baseline_madrs_scr+AGE+GENDER+ED'],'link','logit','Distribution','binomial') %lme4/nlme 
%RSM_Z+RSR_Z+RC_Z+DCSSS+DCFTCSS+DTMT4_Scaled+DTMT5_Scaled+CWI4CSSfinal+CWI3CSSfinal+
scores = model.Fitted.Probability; [X,Y,T,AUC] = perfcurve(madrs_cut.remission,scores, 1);AUC %confusionchart(Ylogist,scores>0.4)
figure(2);hold off;plot(X,Y,'Color',[76/255 0 153/255]);hold on;
model = fitglm(madrs_cut,['remission ~lh_postcentral_thickness+lh_rostralanteriorcingulate_thickness+rh_transversetemporal_thickness+rh_bankssts_thickness+lh_bankssts_thickness+ baseline_madrs_scr+AGE+GENDER+ED'],'link','logit','Distribution','binomial') ;
scores = model.Fitted.Probability; [X,Y,T,AUC] = perfcurve(madrs_cut.remission,scores, 1); figure(2);hold on;plot(X,Y,'Color',[204/255 0 204/255]); AUC
model = fitglm(madrs_cut,['remission ~baseline_madrs_scr+AGE+GENDER+ED'],'link','logit','Distribution','binomial') ;
scores = model.Fitted.Probability; [X,Y,T,AUC] = perfcurve(madrs_cut.remission,scores, 1); figure(2);hold on;plot(X,Y,'Color',[255/255 0 127/255]); AUC


Xlogist=[madrs_cut{:,[183:250, 254:259]}, madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr, madrs_cut.RC_Z, madrs_cut.DTMT4_Scaled];%, madrs_cut.DTMT5_Scaled, madrs_cut.RSM_Z, madrs_cut.RSR_Z
Ylogist=madrs_cut.remission;
varnames=[ct_names, {'age'}, {'sex'},{'blmadrs'}, 'coding', 'tmt4'];%, 'tmt5','rsm','rsr'
[B,FitInfo] = lassoglm(Xlogist,Ylogist,'binomial','CV',10, 'PredictorNames', varnames,'alpha', 0.8);
%lassoPlot(B,FitInfo,'PlotType','CV'); legend('show','Location','best') % show legend
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
MinModelPredictors = FitInfo.PredictorNames(B(:,idxLambdaMinDeviance)~=0)
idxLambda1SE = FitInfo.Index1SE;
sparseModelPredictors = FitInfo.PredictorNames(B(:,idxLambda1SE)~=0)
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; B(:,idxLambdaMinDeviance)];
yhat = glmval(coef,Xlogist,'logit');[X,Y,T,AUC] = perfcurve(Ylogist,yhat, 1);AUC

permutation_index = randperm(length(Ylogist));%for randfold=1%:20
figure;coef_all=[];confusionmatrices=[];yhat_all=[];ytest_all=[];yhatBinom_all=[];
for i=1:5
    if(i==1); ix=zeros([101,1]); ix(1:20)=1; elseif (i==5)
    ix=zeros([101,1]); ix(20*(i-1)+1:length(ix))=1; else
    ix=zeros([101,1]); ix(20*(i-1)+1:20*(i-1)+20)=1; end; ix=ix(permutation_index);
%permutation_index = randperm(length(Ylogist));ix=zeros([length(madrs_cut.ID_madrs_cut), 1]);
%ix(permutation_index(1:20))=1;  %ix(permutation_index(31:304))=0;  
XTest=Xlogist(ix==1,:);XTrain=Xlogist(ix==0,:); 
yTest=Ylogist(ix==1);yTrain=Ylogist(ix==0);

[B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','CV',10, 'PredictorNames', varnames,'alpha', 0.5);
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; B(:,idxLambdaMinDeviance)]; coef_all=[coef_all,coef];
%predicted vs observed
yhat = glmval(coef,XTest,'logit');yhat_all=[yhat_all; yhat];ytest_all=[ytest_all; yTest];
yhatBinom = (yhat>=0.35);yhatBinom_all=[yhatBinom_all; yhatBinom];
if i<6;    subplot(1,5,i); c=confusionchart(yTest,yhatBinom); end
%c=confusionchart(yTest,yhatBinom);
%confusionmatrices(i,:,:)=c.NormalizedValues;
[X,Y,T,AUC] = perfcurve(yTest,yhat, 1);AUC_test(i)=AUC;
end
[X,Y,T,AUC] = perfcurve(ytest_all,yhat_all, 1);AUC %sum(confusionmatrices)
yhatBinom_all=yhat_all>0.31;figure;c=confusionchart(ytest_all,double(yhatBinom_all))
[sensitivity, specificity, accuracy, F1score]=gofmeasures_2d_square(c.NormalizedValues)

sensitivity=sum(confusionmatrices(:,2,2))/sum(sum(confusionmatrices(:,2,:)))
specificity=sum(confusionmatrices(:,1,1))/sum(sum(confusionmatrices(:,1,:)))
accuracy=(sum(confusionmatrices(:,1,1))+sum(confusionmatrices(:,2,2)))/sum(sum(sum(confusionmatrices(:,:,:))))
F1score=2*sum(confusionmatrices(:,2,2))/(2*sum(confusionmatrices(:,2,2))+ sum(confusionmatrices(:,2,1))+ sum(confusionmatrices(:,1,2)) )

load('step2_CT_holdout100.mat');figure(2);hold off;plot(X,Y,'Color',[76/255 0 153/255]);AUC_all(1)=AUC
AUC_test_all(3,:)=AUC_test;
load('step2_COGMADRS_noCT_holdout100.mat');figure(2);hold on;plot(X,Y,'Color',[204/255 0 204/255]); AUC_all(2)=AUC
AUC_test_all(2,:)=AUC_test;
load('step2_MADRS_noCT_holdout100.mat');figure(2);hold on;plot(X,Y,'Color',[255/255 0 127/255]); set(gca,'box','off');AUC_all(3)=AUC
AUC_test_all(1,:)=AUC_test; %figure; b = bar(AUC_all,'k');
load('step2_AUC_alpha.0001_CT_model3.mat'); figure(2);hold on;plot(X,Y,'k');
figure; yyaxis right; set(gca, 'color', 'none'); b = bar(mean(AUC_test_all'));
SEM = std(AUC_test_all')/sqrt(100);
hold on; [ngroups,nbars] = size(mean(AUC_test_all'));
tmp = nan(nbars, ngroups); for i = 1:nbars    tmp(i) = b.XEndPoints(i); end
errorbar(tmp',mean(AUC_test_all'), 2*SEM,'k','linestyle','none');ylim([0.5 0.8]);hold on
b = bar([1; 2; 3],diag(mean(AUC_test_all')),'stacked');
b(1).FaceColor=[255/255 0 127/255];b(2).FaceColor=[204/255 0 204/255]; b(3).FaceColor=[76/255 0 153/255];