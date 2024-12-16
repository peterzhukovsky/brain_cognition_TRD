clear
%OPT data import
%OPT NEURO
%load('D:\Canada_2020\OPT\reports\rsfc_2021\feat.mat');
cd C:\Users\peter\Documents\OPT\OPT\reports\dti_2023
subjects={'sub-CU1CU10003';'sub-CU1CU10009';'sub-CU1CU10010';'sub-CU1CU10012';'sub-CU1CU10014';'sub-CU1CU10021';'sub-CU1CU10022';'sub-CU1CU10023';'sub-CU1CU10024';'sub-CU1CU10026';'sub-CU1CU10033';'sub-CU1CU10034';'sub-CU1CU10045';'sub-CU1CU10056';'sub-CU1CU10057';'sub-CU1CU10060';'sub-CU2CU10070';'sub-CU2CU10074';'sub-CU2CU10076';'sub-CU2CU10078';'sub-CU2CU10079';'sub-CU2CU10081';'sub-CU2CU10083';'sub-CU2CU10084';'sub-CU2CU10089';'sub-CU2CU10091';'sub-CU2CU10098';'sub-CU2CU10099';'sub-CU2CU10101';'sub-CU2CU10104';'sub-CU2CU10109';'sub-CU2CU10110';'sub-CU2CU10114';'sub-CU2CU10119';'sub-CU2CU10141';'sub-CU2CU10146';'sub-CU2CU20002';'sub-CU2CU20003';'sub-CU2CU20005';'sub-CU2CU20006';'sub-CU2CU20011';'sub-CU2CU20015';'sub-CU2CU20016';'sub-LA1LA10006';'sub-LA1LA10008';'sub-LA1LA10026';'sub-LA1LA10032';'sub-LA1LA10035';'sub-LA1LA10038';'sub-LA1LA10040';'sub-LA1LA10044';'sub-LA1LA10048';'sub-LA1LA10054';'sub-LA1LA10056';'sub-LA1LA10099';'sub-LA1LA10104';'sub-LA1LA10105';'sub-LA1LA20012';'sub-LA1LA20019';'sub-LA1LA20021';'sub-LA1LA20032';'sub-LA1LA20039';'sub-LA1LA20043';'sub-UP10004';'sub-UP10007';'sub-UP10044';'sub-UP10047';'sub-UP10057';'sub-UP10071';'sub-UP10074';'sub-UP10076';'sub-UP10077';'sub-UP10092';'sub-UP10094';'sub-UP10110';'sub-UP10124';'sub-UP10155';'sub-UP10266';'sub-UP10280';'sub-UP1UP10005';'sub-UP1UP10021';'sub-UP1UP10053';'sub-UP1UP10061';'sub-UP1UP10062';'sub-UP1UP10087';'sub-UP1UP10109';'sub-UP1UP10111';'sub-UP1UP10133';'sub-UP1UP10137';'sub-UP1UP10148';'sub-UP1UP10165';'sub-UP1UP10203';'sub-UP1UP10206';'sub-UP1UP10231';'sub-UP1UP10258';'sub-UP1UP10261';'sub-UP2UP10003';'sub-UP2UP10006';'sub-UP2UP10009';'sub-UP2UP10020';'sub-UP2UP10026';'sub-UP2UP10032';'sub-UP2UP10049';'sub-UP2UP10054';'sub-UP2UP10058';'sub-UP2UP10066';'sub-UP2UP10080';'sub-UP2UP10081';'sub-UP2UP10090';'sub-UP2UP10096';'sub-UP2UP10098';'sub-UP2UP10101';'sub-UP2UP10112';'sub-UP2UP10113';'sub-UP2UP10114';'sub-UP2UP10125';'sub-UP2UP10126';'sub-UP2UP10128';'sub-UP2UP10130';'sub-UP2UP10135';'sub-UP2UP10136';'sub-UP2UP10151';'sub-UP2UP10156';'sub-UP2UP10158';'sub-UP2UP10161';'sub-UP2UP10163';'sub-UP2UP10171';'sub-UP2UP10173';'sub-UP2UP10175';'sub-UP2UP10184';'sub-UP2UP10187';'sub-UP2UP10188';'sub-UP2UP10196';'sub-UP2UP10201';'sub-UP2UP10204';'sub-UP2UP10209';'sub-UP2UP10210';'sub-UP2UP10229';'sub-UP2UP10244';'sub-UP2UP10250';'sub-UT1UT10018';'sub-UT1UT10023';'sub-UT1UT10066';'sub-UT1UT10067';'sub-UT1UT10072';'sub-UT1UT10073';'sub-UT1UT10075';'sub-UT1UT10078';'sub-UT1UT10079';'sub-UT1UT10080';'sub-UT1UT10083';'sub-UT1UT10091';'sub-UT1UT10109';'sub-UT1UT10111';'sub-UT1UT10115';'sub-UT1UT10116';'sub-UT1UT10125';'sub-UT1UT10126';'sub-UT1UT10128';'sub-UT1UT10137';'sub-UT1UT30006';'sub-UT1UT30008';'sub-UT1UT30011';'sub-UT1UT30018';'sub-UT1UT30019';'sub-UT1UT30021';'sub-UT1UT30025';'sub-UT1UT30026';'sub-UT1UT30027';'sub-UT1UT30028';'sub-UT1UT30030';'sub-UT1UT30032';'sub-UT1UT30033';'sub-UT1UT30034';'sub-UT1UT30036';'sub-UT1UT30042';'sub-UT2UT10003';'sub-UT2UT10004';'sub-UT2UT10013';'sub-UT2UT10015';'sub-UT2UT10016';'sub-UT2UT10019';'sub-UT2UT10021';'sub-UT2UT10025';'sub-UT2UT10033';'sub-UT2UT10039';'sub-UT2UT10046';'sub-UT2UT10054';'sub-UT2UT10081';'sub-UT2UT10087';'sub-UT2UT10090';'sub-UT2UT10114';'sub-UT2UT10120';'sub-UT2UT10123';'sub-UT2UT10144';'sub-UT2UT30003';'sub-UT2UT30029';'sub-UT2UT30031';'sub-UT2UT30039';'sub-UT2UT30041';'sub-UT2UT30043';'sub-WU10089';'sub-WU1WU10031';'sub-WU1WU10055';'sub-WU1WU10057';'sub-WU1WU10085';'sub-WU1WU10099';'sub-WU1WU10105';'sub-WU1WU10114';'sub-WU1WU10115';'sub-WU1WU10117';'sub-WU1WU10127';'sub-WU1WU10131';'sub-WU1WU10137';'sub-WU1WU10146';'sub-WU1WU10150';'sub-WU1WU10152';'sub-WU1WU10157';'sub-WU1WU10161';'sub-WU1WU10164';'sub-WU1WU10168';'sub-WU1WU10174';'sub-WU1WU10189';'sub-WU1WU20001';'sub-WU1WU20002';'sub-WU1WU20004';'sub-WU1WU20005';'sub-WU1WU20006';'sub-WU2WU10008';'sub-WU2WU10014';'sub-WU2WU10028';'sub-WU2WU10036';'sub-WU2WU10054';'sub-WU2WU10058';'sub-WU2WU10059';'sub-WU2WU10062';'sub-WU2WU10064';'sub-WU2WU10066';'sub-WU2WU10071';'sub-WU2WU10073';'sub-WU2WU10077';'sub-WU2WU10083';'sub-WU2WU10087';'sub-WU2WU10092';'sub-WU2WU10093';'sub-WU2WU10096';'sub-WU2WU10104';'sub-WU2WU10107';'sub-WU2WU10108';'sub-WU2WU10112';'sub-WU2WU10121';'sub-WU2WU10122';'sub-WU2WU10124';'sub-WU2WU10128';'sub-WU2WU10135';'sub-WU2WU10141';'sub-WU2WU10145';'sub-WU2WU10147';'sub-WU2WU10149';'sub-WU2WU10151';'sub-WU2WU10154';'sub-WU2WU10158';'sub-WU2WU10165';'sub-WU2WU10166';'sub-WU2WU10177'};
subjects_short={'CU10003';'CU10009';'CU10010';'CU10012';'CU10014';'CU10021';'CU10022';'CU10023';'CU10024';'CU10026';'CU10033';'CU10034';'CU10045';'CU10056';'CU10057';'CU10060';'CU10070';'CU10074';'CU10076';'CU10078';'CU10079';'CU10081';'CU10083';'CU10084';'CU10089';'CU10091';'CU10098';'CU10099';'CU10101';'CU10104';'CU10109';'CU10110';'CU10114';'CU10119';'CU10141';'CU10146';'CU20002';'CU20003';'CU20005';'CU20006';'CU20011';'CU20015';'CU20016';'LA10006';'LA10008';'LA10026';'LA10032';'LA10035';'LA10038';'LA10040';'LA10044';'LA10048';'LA10054';'LA10056';'LA10099';'LA10104';'LA10105';'LA20012';'LA20019';'LA20021';'LA20032';'LA20039';'LA20043';'UP10004';'UP10007';'UP10044';'UP10047';'UP10057';'UP10071';'UP10074';'UP10076';'UP10077';'UP10092';'UP10094';'UP10110';'UP10124';'UP10155';'UP10266';'UP10280';'UP10005';'UP10021';'UP10053';'UP10061';'UP10062';'UP10087';'UP10109';'UP10111';'UP10133';'UP10137';'UP10148';'UP10165';'UP10203';'UP10206';'UP10231';'UP10258';'UP10261';'UP10003';'UP10006';'UP10009';'UP10020';'UP10026';'UP10032';'UP10049';'UP10054';'UP10058';'UP10066';'UP10080';'UP10081';'UP10090';'UP10096';'UP10098';'UP10101';'UP10112';'UP10113';'UP10114';'UP10125';'UP10126';'UP10128';'UP10130';'UP10135';'UP10136';'UP10151';'UP10156';'UP10158';'UP10161';'UP10163';'UP10171';'UP10173';'UP10175';'UP10184';'UP10187';'UP10188';'UP10196';'UP10201';'UP10204';'UP10209';'UP10210';'UP10229';'UP10244';'UP10250';'UT10018';'UT10023';'UT10066';'UT10067';'UT10072';'UT10073';'UT10075';'UT10078';'UT10079';'UT10080';'UT10083';'UT10091';'UT10109';'UT10111';'UT10115';'UT10116';'UT10125';'UT10126';'UT10128';'UT10137';'UT30006';'UT30008';'UT30011';'UT30018';'UT30019';'UT30021';'UT30025';'UT30026';'UT30027';'UT30028';'UT30030';'UT30032';'UT30033';'UT30034';'UT30036';'UT30042';'UT10003';'UT10004';'UT10013';'UT10015';'UT10016';'UT10019';'UT10021';'UT10025';'UT10033';'UT10039';'UT10046';'UT10054';'UT10081';'UT10087';'UT10090';'UT10114';'UT10120';'UT10123';'UT10144';'UT30003';'UT30029';'UT30031';'UT30039';'UT30041';'UT30043';'WU10089';'WU10031';'WU10055';'WU10057';'WU10085';'WU10099';'WU10105';'WU10114';'WU10115';'WU10117';'WU10127';'WU10131';'WU10137';'WU10146';'WU10150';'WU10152';'WU10157';'WU10161';'WU10164';'WU10168';'WU10174';'WU10189';'WU20001';'WU20002';'WU20004';'WU20005';'WU20006';'WU10008';'WU10014';'WU10028';'WU10036';'WU10054';'WU10058';'WU10059';'WU10062';'WU10064';'WU10066';'WU10071';'WU10073';'WU10077';'WU10083';'WU10087';'WU10092';'WU10093';'WU10096';'WU10104';'WU10107';'WU10108';'WU10112';'WU10121';'WU10122';'WU10124';'WU10128';'WU10135';'WU10141';'WU10145';'WU10147';'WU10149';'WU10151';'WU10154';'WU10158';'WU10165';'WU10166';'WU10177'};

tracts={'T_AF_left ';'T_AF_right ';'T_CB_left ';'T_CB_right ';'T_CC1 ';'T_CC2 ';'T_CC3 ';'T_CC4 ';'T_CC5 ';'T_CC6 ';'T_CC7 ';'T_CPC ';'T_CR-F_left ';'T_CR-F_right ';'T_CR-P_left ';'T_CR-P_right ';'T_CST_left ';'T_CST_right ';'T_EC_left ';'T_EC_right ';'T_EmC_left ';'T_EmC_right ';'T_ICP_left ';'T_ICP_right ';'T_ILF_left ';'T_ILF_right ';'T_IOFF_left ';'T_IOFF_right ';'T_Intra-CBLM-I&P_left ';'T_Intra-CBLM-I&P_right ';'T_Intra-CBLM-PaT_left ';'T_Intra-CBLM-PaT_right ';'T_MCP ';'T_MdLF_left ';'T_MdLF_right ';'T_PLIC_left ';'T_PLIC_right ';'T_SF_left ';'T_SF_right ';'T_SLF-III_left ';'T_SLF-III_right ';'T_SLF-II_left ';'T_SLF-II_right ';'T_SLF-I_left ';'T_SLF-I_right ';'T_SO_left ';'T_SO_right ';'T_SP_left ';'T_SP_right ';'T_Sup-FP_left ';'T_Sup-FP_right ';'T_Sup-F_left ';'T_Sup-F_right ';'T_Sup-OT_left ';'T_Sup-OT_right ';'T_Sup-O_left ';'T_Sup-O_right ';'T_Sup-PO_left ';'T_Sup-PO_right ';'T_Sup-PT_left ';'T_Sup-PT_right ';'T_Sup-P_left ';'T_Sup-P_right ';'T_Sup-T_left ';'T_Sup-T_right ';'T_TF_left ';'T_TF_right ';'T_TO_left ';'T_TO_right ';'T_TP_left ';'T_TP_right ';'T_UF_left ';'T_UF_right '};

for i=1:length(subjects);
   dta=readtable(strcat(subjects{i},'_ses-01_anatomical_tracts.csv'), 'delimiter', ',');
   %l(i)=length(dta.Name);
   FA(i,:)=dta.tensor1_FractionalAnisotropy_Mean';
   QC_NumPoints(i,:)=dta.Num_Points';
   QC_NumFibers(i,:)=dta.Num_Fibers';
   QC_Length(i,:)=dta.Mean_Length';
end %FA=readtable('FA_2023.csv');FA=FA{:,3:75};
histogram(QC_NumFibers(:,1))
histogram(QC_NumFibers, 100)
%FA(QC_NumFibers<400)=NaN; %imagesc(sum(QC_NumFibers<400)./265*100)
%FA(QC_NumPoints<3000)=NaN;
wm_qc=readtable('White Matter QC.xlsx'); wm_qc=wm_qc{:,2:74};
FA(62,:)=[];subjects_short(62)=[];subjects(62)=[];
FA(wm_qc==2)=NaN;
%FA(QC_NumFibers<100)=NaN; 
FA(1:43,:)=[];subjects(1:43)=[];subjects_short(1:43)=[];
FA_cut=FA(:, sum(isnan(FA))./221*100<3); tracts_included=tracts(sum(isnan(FA))./221*100<3);
figure;imagesc(FA_cut)
FA_cut(:,[26 27])=[]; tracts_included([26 27])=[]; 
%ix = [3 4 5 14 15 23]; FA_cut(:,[ix])=[]; tracts_included([ix])=[];

%d=readtable('C:\Users\peter\Documents\OPT\OPT\data\ds_ON_BASELINE_DATA_April2022n_for_cleaning_ID_6_5_22.xlsx');
d=readtable('C:\Users\peter\Documents\OPT\OPT\data\ON_DATASET_08302024.xlsx'); d.ID=d.ID_complete;

%%
d.CWI3CSSFinal_01(d.CWI3CSSFinal_01>80)=NaN;
d.CWI4CSSFinal_01(d.CWI4CSSFinal_01>800)=NaN;
d.DTMTS4_01(d.DTMTS4_01>80)=NaN;
d.DIFFSS_01(d.DIFFSS_01>80)=NaN;
d.DTMTS5_01(d.DTMTS5_01>80)=NaN;


sum(~isnan(d.DTMTS4_01))
sum(~isnan(d.CWI3CSSFinal_01))
d.DTMTS4_01(isnan(d.DTMTS4_01))=nanmean(d.DTMTS4_01);
d.DTMTS5_01(isnan(d.DTMTS5_01))=nanmean(d.DTMTS5_01);
d.DIFFSS_01(isnan(d.DIFFSS_01))=nanmean(d.DIFFSS_01);

d.CWI3CSSFinal_01(isnan(d.CWI3CSSFinal_01))=nanmean(d.CWI3CSSFinal_01);
d.CWI4CSSFinal_01(isnan(d.CWI4CSSFinal_01))=nanmean(d.CWI4CSSFinal_01);

d.EXEC_01=(d.DTMTS4_01+d.CWI3CSSFinal_01)/2;
d.EXEC_01=(d.DIFFSS_01+d.CWI3CSSFinal_01)/2;

%corr(d.EXEC_01, d.CWI3CSSFinal_01)
d(d.SITE_MRI_1_WU_2_UT_3_UP_4_LA_5_CU==5,:)=[];



%% 
for i=1:length(FA_cut(1,:))
    try
FA_cut(isnan(FA_cut(:,i)), i)=nanmean(FA_cut(:,i));
    end
end
%ix=sum(isnan(FA_cut)')'>0;
%FA_cut(ix,:)=[];subjects_short(ix)=[];

subjects_short=table(subjects_short, 'VariableNames',{'ID'});
subjects_short.dti=FA_cut;
subjects_short.subs=subjects; subjects_short.scanner= ~cellfun(@isempty,(strfind(subjects_short.subs, 'CU'))) |  ~cellfun(@isempty,(strfind(subjects_short.subs, 'UT1UT1')))  |  ~cellfun(@isempty,(strfind(subjects_short.subs, 'UT1UT3'))) ;

demo_OPT=innerjoin(d, subjects_short);
FA_cut=demo_OPT.dti;
Y=[demo_OPT.AIS_01, demo_OPT.IMIS_01, demo_OPT.MDMIS_01, demo_OPT.LIS_01, demo_OPT.MVCIS_01, demo_OPT.EXEC_01];
figure; for i=1:6;     subplot(2,3,i);histogram(Y(:,i)); end

%%
batch = demo_OPT.SITE_MRI_1_WU_2_UT_3_UP_4_LA_5_CU';%batch = demo_OPT.scanner';
dat = FA_cut';
age = demo_OPT.AGE;
sex = demo_OPT.GENDER;
sex = dummyvar(sex);
mod = [demo_OPT.RACE demo_OPT.HL demo_OPT.ED age sex(:,2)];
FA_harmonized = combat(dat, batch, mod, 1);
FA_harmonized=FA_harmonized';

%% PLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y=[demo_OPT.AIS_01, demo_OPT.IMIS_01, demo_OPT.MDMIS_01, demo_OPT.LIS_01, demo_OPT.MVCIS_01, demo_OPT.EXEC_01];
Y=[demo_OPT.AIS_01, demo_OPT.IMIS_01, demo_OPT.MDMIS_01, demo_OPT.LIS_01, demo_OPT.MVCIS_01, ...
    demo_OPT.CWI3CSSFinal_01,demo_OPT.CWI4CSSFinal_01, demo_OPT.DIFFSS_01, demo_OPT.DTMTS5_01, demo_OPT.DTMTS4_01];
X=FA_harmonized;
figure;imagesc(corr(X,Y)); colormap jet
%%% remove nans
naninx=sum(isnan(X)')'>0| sum(isnan(Y)')'>0 ; Y=Y(naninx==0,:);  X=X(naninx==0,:);  d_OPT=demo_OPT(naninx==0,:);
Y=zscore(Y); X=zscore(X);x=X;
ncomp=3
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,Y,ncomp);PCTVAR
%% permutation testing
permutations=5000;  
allobservations=Y;
for ncomp=1:1
   
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
[c,pval]=corr(XS, Y); %c(3,:)=c(3,:)*-1; % c(2,:)=c(2,:)*-1;   reshape(mafdr(reshape(pval, [1 36]), 'BHFDR', 'True'), [4 9])
figure;imagesc(c); colormap bone; colorbar
l=length(Y(1,:)); combs=allcomb([1], [1:l]);figure(13);
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
%% bootstrapping to get the func connectivity weights for PLS1, 2 and 3
dim=1
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,Y,dim);PCTVAR
%PLS4w=stats.W(:,4);
%PLS3w=stats.W(:,3);
%PLS2w=stats.W(:,2);
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
     
    %newW=stats.W(:,3);%extract PLS2 weights
    %if corr(PLS3w,newW)<0 % the sign of PLS components is arbitrary - make sure this aligns between runs
    %    newW=-1*newW;
    %end
    %PLS3weights=[PLS3weights,newW]; %store (ordered) weights from this bootstrap run    
   
   % newW=stats.W(:,2);%extract PLS2 weights
   % if corr(PLS2w,newW)<0 % the sign of PLS components is arbitrary - make sure this aligns between runs
   %     newW=-1*newW;
   % end
  %  PLS2weights=[PLS2weights,newW];
   
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
%PLS3sw=std(PLS3weights');
%plsweights3=PLS3w./PLS3sw';
%PLS2sw=std(PLS2weights');
%plsweights2=PLS2w./PLS2sw';
PLS1sw=std(PLS1weights');
plsweights1=PLS1w./PLS1sw';
%%
tracts_included(plsweights1<-3)
%tracts_included(plsweights2<-3)
%tracts_included(plsweights2>3)
figure; imagesc((plsweights1<-3)'); colormap bone
%%
figure; scatter(XS(:,1), YS(:,1), 'filled', 'k');lsline;
corr(XS(:,1), YS(:,1))
%corr(XS(:,3), YS(:,3))

figure; histogram(Rsq); xline(PCTVAR(2));


%%
%%
%%
%% Longitudinal relationship with MADRS and PHQ
%madrs=readtable('C:\Users\peter\Documents\OPT\OPT\data\OPTIMUMMainDatabaseF_DATA_2023-07-03_0248_ALL_MADRS.csv');
madrs=readtable('C:\Users\peter\Documents\OPT\OPT\data\MADRS_Longitudinally_20230711_updated');
subs=table(d_OPT.ID, 'VariableNames',{'ID'});subs.NPDateDays=d_OPT.NPDateDaysKaylaUPDATE;
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
catch; b(i)=NaN; end; end %madrs_cut.days_bl=days;madrs_cut.Properties.VariableNames{1} = 'ID';

madrs_cut.time_in_d=double(dayss); madrs_cut=outerjoin(madrs_cut, d_OPT);madrs_cut(cellfun(@isempty,madrs_cut.ID_complete), :)=[];
madrs_cut.CDRSCORE_01( madrs_cut.CDRSCORE_01>1.5)=NaN;madrs_cut.CDRSUMBOX_01( madrs_cut.CDRSUMBOX_01>=2)=NaN;

madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'month'))) ;madrs_cut(~madrs_cut.remission==0,:)=[];
madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'mth'))) ;madrs_cut(~madrs_cut.remission==0,:)=[];
madrs_cut.remission=madrs_cut.madrs_tot_scr<=10;
%% LME
madrs_cut.T_EmC_right=madrs_cut.dti(:,19);madrs_cut.T_SF_right=madrs_cut.dti(:,32);madrs_cut.T_CB_right=madrs_cut.dti(:,4);
madrs_cut.T_SLFIII_right=madrs_cut.dti(:,35);madrs_cut.T_SLFII_left=madrs_cut.dti(:,34);madrs_cut.T_UF_left=madrs_cut.dti(:,61);

fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d*T_EmC_right+(1+time_in_d|ID_madrs_cut)' )
fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d*T_SF_right+(1+time_in_d|ID_madrs_cut)' )

%% 1/0 step 1
madrs_cut.madrs_tot_scr( ~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'end'))) )
madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'end'))) ;
madrs_cut(madrs_cut.remission==0,:)=[];
madrs_cut.remission=madrs_cut.madrs_tot_scr<=10;

madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'step_2'))) ;madrs_cut(madrs_cut.remission==1,:)=[];
madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'arm_7'))) ;madrs_cut(madrs_cut.remission==1,:)=[];
madrs_cut.remission=madrs_cut.madrs_tot_scr<=10;


%anova1(madrs_cut.T_EmC_right, madrs_cut.remission);
Xlogist=[madrs_cut.dti, madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr,madrs_cut.RC_Z, madrs_cut.DTMT4_Scaled]; 
varnames=vertcat(tracts_included,{'age'},{'sex'}, {'baseline_madrs_scr'}, 'coding', 'tmt4');
%Xlogist=[madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr,madrs_cut.RC_Z, madrs_cut.DTMT4_Scaled]; varnames=vertcat({'age'},{'sex'}, {'baseline_madrs_scr'}, 'coding', 'tmt4');
Xlogist=[madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr]; varnames=vertcat({'age'},{'sex'}, {'baseline_madrs_scr'});
Ylogist=madrs_cut.remission;

[B,FitInfo] = lassoglm(Xlogist,Ylogist,'binomial','alpha',0.5,'CV',10, 'PredictorNames', varnames);
%lassoPlot(B,FitInfo,'PlotType','CV'); legend('show','Location','best') % show legend
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
MinModelPredictors = FitInfo.PredictorNames(B(:,idxLambdaMinDeviance)~=0)
idxLambda1SE = FitInfo.Index1SE;
sparseModelPredictors = FitInfo.PredictorNames(B(:,idxLambda1SE)~=0)
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; B(:,idxLambdaMinDeviance)];
yhat = glmval(coef,Xlogist,'logit');[X,Y,T,AUC] = perfcurve(Ylogist,yhat, 1);AUC

figure;coef_all=[];confusionmatrices=[];yhat_all=[];ytest_all=[];
for i=1:100
%if(i==1); ix=zeros([294,1]); ix(1:20)=1; elseif (i==10); ix=zeros([294,1]); ix(20*(i-1)+1:196)=1; 
%else; ix=zeros([294,1]); ix(20*(i-1)+1:20*(i-1)+20)=1;end
permutation_index = randperm(length(Ylogist));ix=zeros([length(madrs_cut.ID_madrs_cut), 1]);
ix(permutation_index(1:32))=1;  %ix(permutation_index(31:304))=0;  
XTest=Xlogist(ix==1,:);XTrain=Xlogist(ix==0,:); 
yTest=Ylogist(ix==1);yTrain=Ylogist(ix==0);

[B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','CV',10, 'PredictorNames', varnames,'alpha', 0.5);
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; B(:,idxLambdaMinDeviance)]; coef_all=[coef_all,coef];
%predicted vs observed
yhat = glmval(coef,XTest,'logit');yhat_all=[yhat_all; yhat];ytest_all=[ytest_all; yTest];
yhatBinom = (yhat>=0.35);
if i<11;    subplot(2,5,i); c=confusionchart(yTest,yhatBinom); end
c=confusionchart(yTest,yhatBinom);
confusionmatrices(i,:,:)=c.NormalizedValues;
[X,Y,T,AUC] = perfcurve(yTest,yhat, 1);AUC_test(i)=AUC;
end
mean(AUC_test)
[X,Y,T,AUC] = perfcurve(ytest_all,yhat_all, 1);AUC %sum(confusionmatrices)

[sensitivity, specificity, accuracy, F1score]=gofmeasures_3d_square(confusionmatrices)


load('DTI_holdoutAUC_rerun100.mat');figure(2);hold off;plot(X,Y,'Color',[0/255 180/255 0/255]);AUC_all(1)=AUC
AUC_test_all(3,:)=AUC_test;
load('COGMADRS_only_noDTI_holdoutAUC_rerun100.mat');figure(2);hold on;plot(X,Y,'Color',[204/255 0 204/255]); AUC_all(2)=AUC
AUC_test_all(2,:)=AUC_test;
load('MADRS_only_noDTI_holdoutAUC100.mat');figure(2);hold on;plot(X,Y,'Color',[255/255 0 127/255]);set(gca,'box','off'); AUC_all(3)=AUC
AUC_test_all(1,:)=AUC_test; %figure; b = bar(AUC_all,'k');
figure; yyaxis right; set(gca, 'color', 'none'); b = bar(mean(AUC_test_all'));
SEM = std(AUC_test_all')/sqrt(100);
hold on; [ngroups,nbars] = size(mean(AUC_test_all'));
tmp = nan(nbars, ngroups); for i = 1:nbars    tmp(i) = b.XEndPoints(i); end
errorbar(tmp',mean(AUC_test_all'), 2*SEM,'k','linestyle','none');ylim([0.5 0.8]);hold on
b = bar([1; 2; 3],diag(mean(AUC_test_all')),'stacked');
b(1).FaceColor=[255/255 0 127/255];b(2).FaceColor=[204/255 0 204/255]; b(3).FaceColor=[0/255 180/255 0/255];


madrs_cut.T_EmC_right=madrs_cut.dti(:,19);madrs_cut.T_SF_right=madrs_cut.dti(:,32);madrs_cut.T_CB_right=madrs_cut.dti(:,4);
madrs_cut.T_SLFIII_right=madrs_cut.dti(:,35);madrs_cut.T_SLFII_left=madrs_cut.dti(:,34);madrs_cut.T_UF_left=madrs_cut.dti(:,61);
model = fitglm(madrs_cut,['remission ~ T_EmC_right+T_CB_right+T_SF_right+T_SLFIII_right+T_SLFII_left+baseline_madrs_scr' ...
    '+AGE+GENDER+ED'],'link','logit','Distribution','binomial') %lme4/nlme
scores = model.Fitted.Probability; [X,Y,T,AUC] = perfcurve(madrs_cut.remission,scores, 1);AUC

model = fitglm(madrs_cut,['remission ~ DTMT4_Scaled+DTMT5_Scaled+CWI4CSSfinal+CWI3CSSfinal' ...
    '+AGE+GENDER+ED+baseline_madrs_scr'],'link','logit','Distribution','binomial') %lme4/nlme
scores = model.Fitted.Probability; [X,Y,T,AUC] = perfcurve(madrs_cut.remission,scores, 1);AUC
[B,dev,stats] = mnrfit(double(Ylogist), Xlogist)

%% old longitudinal - not as useful
madrs=readtable('C:\Users\peter\Documents\OPT\OPT\data\MADRS_Longitudinally_20230711_updated');
subs=table(d_OPT.ID, 'VariableNames',{'ID'});subs.NPDateDays=d_OPT.NPDateDaysKaylaUPDATE;

for i=1:length(subs.ID);try
    sub=subs.ID{i};
    data=madrs(strcmp(madrs.ID, sub),:);
    MADRS_BL(i)=data.madrs_tot_scr(1);
    data.DaysDiff=data.MADRSDateDays-d_OPT.NPDateDaysKaylaUPDATE(strcmp(d_OPT.ID, sub));
    ix=abs(data.DaysDiff)==min(abs(data.DaysDiff));
    MADRS_NP(i)=data.madrs_tot_scr(ix);MADRS_DateDiffNP(i)=data.DaysDiff(ix);%MADRS_TRUE_BL(i)=(ix(1)==1);
    startstudydate=min(min([data.start_step1days, data.start_step2days]));
    MADRS_length_studyNP(i)=data.MADRSDateDays(ix)-startstudydate;MADRS_TRUE_BL(i)=MADRS_length_studyNP(i)<42; %d_OPT.NPDateDays(strcmp(d_OPT.ID, sub))
    catch
    MADRS_NP(i)=NaN;MADRS_BL(i)=NaN;MADRS_DateDiff(i)=NaN;MADRS_length_studyNP(i)=NaN;
end; end

MADRS_NP(abs(MADRS_DateDiffNP)>60)=NaN;

%figure; histogram((MADRS_BL-MADRS_NP)./MADRS_BL);
%figure; histogram(MADRS_NP);sum(MADRS_NP<11)

responders=((MADRS_BL-MADRS_NP)./MADRS_BL>0.5);
sum(isnan(MADRS_NP))
sum(MADRS_NP<11)
MADRS_groups=ones([length(subs.ID),1]);MADRS_groups(MADRS_groups==1)=NaN;
MADRS_groups(MADRS_TRUE_BL==1 &  ~isnan(MADRS_NP) )=1;
MADRS_groups(MADRS_TRUE_BL==0 &  MADRS_NP<11)=0; % responders==1
MADRS_groups(MADRS_TRUE_BL==0 &  MADRS_NP>=11)=2;
[p,t,stats]=anova1(YS(:,1)*-1, MADRS_groups); [c,m,h,gnames] = multcompare(stats, 'CriticalValueType','hsd');
figure; violinplot(YS(:,1)*-1, MADRS_groups);xlim([0.5 3.5])
gscatter(d_OPT.AGE, YS(:,1), MADRS_groups);lsline

%% relationship with PHQ
edu=readtable('C:\Users\peter\Documents\OPT\OPT\data\OPTIMUMMainDatabaseF_DATA_2023-02-01_PZedit.csv');
madrs=readtable('C:\Users\peter\Documents\OPT\OPT\data\OPT_MADRS_Score_Closest_to_NP_Date.csv');
subs=table(d_OPT.ID, 'VariableNames',{'ID'});
clinical_OPT=innerjoin(subs, madrs);
edu=innerjoin(subs, edu);edu=outerjoin(subs, edu);

clinical_OPT.madrs_tot_scr(clinical_OPT.datediff>60)=NaN;

[rr,pp]=corr(YS, clinical_OPT.madrs_tot_scr, 'rows', 'pairwise')
[rr,pp]=corr(XS, clinical_OPT.madrs_tot_scr, 'rows', 'pairwise')
[rr,pp]=corr(d_OPT.ED, YS, 'rows', 'pairwise')
[rr,pp]=corr(d_OPT.ED, XS,'rows', 'pairwise')
figure; scatter(d_OPT.ED, -1*YS(:,1), 15, 'k', 'filled'); lsline
figure; scatter(d_OPT.ED, -1*XS(:,1), 15, 'k', 'filled'); lsline; xlim([7 21])

figure; scatter(clinical_OPT.madrs_tot_scr, -1*YS(:,1), 15, 'k', 'filled'); lsline


d_OPT.CDRSCORE( d_OPT.CDRSCORE>=1.5)=NaN;d_OPT.CDRSUMBOX( d_OPT.CDRSUMBOX>=2)=NaN;
anova1(YS(:,1)*-1, d_OPT.CDRSCORE)
anova1(YS(:,2), d_OPT.CDRSCORE)

anova1(YS(:,2), clinical_OPT.madrs_tot_scr<10)

anova1(YS(:,3), d_OPT.CDRSCORE)
anova1(XS(:,2), d_OPT.CDRSCORE)
anova1(XS(:,3), d_OPT.CDRSCORE)


edu.demo_race(edu.demo_race>2)=NaN;
anova1(XS(:,2), edu.demo_ethnicity)
anova1(YS(:,2), edu.demo_ethnicity)
anova1(XS(:,2), edu.demo_race)


clinical_OPT.datediff

%% relationship with brain age / centile
centile=readtable('C:\Users\peter\Documents\OPT\OPT\data\brain_age\data_centiles_2023-05-29.csv');
subs=table(d_OPT.ID, 'VariableNames',{'ID'});

centile=innerjoin(subs, centile);
[rr,pp]=corr(YS, centile.centile, 'rows', 'pairwise')
[rr,pp]=corr(XS, centile.centile, 'rows', 'pairwise')
corr(centile.centile, edu.demo_edu, 'rows', 'pairwise')

figure; scatter(centile.centile, -1*YS(:,1), 15, 'k', 'filled'); lsline
anova1(centile.centile, d_OPT.CDRSCORE)

%% relationship with treatment resistance severity
athf=readtable('C:\Users\peter\Documents\OPT\OPT\data\OPTN_ATHF_Summary_Scores_N453_May23.csv');
subs=table(d_OPT.ID, 'VariableNames',{'ID'});
clinical_OPT=innerjoin(subs, athf);
clinical_OPT=outerjoin(subs, clinical_OPT);

[rr,pp]=corr(clinical_OPT.athf_total_score_v2, YS, 'rows', 'pairwise')
[rr,pp]=corr(clinical_OPT.athf_total_score_v2, XS,'rows', 'pairwise')
anova1(YS(:,2),clinical_OPT.athf_highest_trial_v2)

figure; scatter(clinical_OPT.athf_total_score_v2, -1*YS(:,1), 15, 'k', 'filled'); lsline
%% holdout prediction analysis
%[~,scores]=pca(Y(:, pval<0.05/12)); Ycomb(:,1)=scores(:,1);
%[~,scores]=pca(Y(:,9:12));Ycomb(:,2)=scores(:,1);
Ycomb(:,1)=Y(:,9);
clear accuracy*

l=length(X(:,1)); 
%randomsample=randperm(l); X=X(randomsample,:);Y=Y(randomsample,:);
l=round(0.25*(l),0); 
%for h=1:5 %ixtest=d_OPT.SITE_MRI_1_WU_2_UT_3_UP_4_LA_5_CU==h;
figure(11);

for h=1:4
xstart=(h-1)*l+1;
xend=h*l;
ixtest=zeros(1,length(X(:,1)));
ixtest(xstart:xend)=1;

ixtest=d_OPT.SITE_MRI_1_WU_2_UT_3_UP_4_LA_5_CU==h;
X_train=x(ixtest==0,:); X_test=x(ixtest==1 ,:);
Y_train=Ycomb(ixtest==0,:); Y_test=Ycomb(ixtest==1 ,:);
%l=length(X(:,1)); l=round(0.25*l,0); X_test=x(1:l,:); X_train=x(l+1:length(X(:,1)) ,:);Y_test=Y(1:l,:); Y_train=Y(l+1:length(X(:,1)) ,:);
dim=1;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X_train,Y_train,dim);PCTVAR;
r=corr(XS, Y_train); accuracytrain(h,:)=r(1,:);
y_pred = [ones(size(X_test ,1),1) X_test]*BETA;
r=corr(Y_test, y_pred);accuracyholdout(h,:)=r(eye(1)==1);
subplot(4,1,h); scatter(y_pred, Y_test, 'x', 'k'); p = polyfit(y_pred,Y_test,1); pred = polyval(p,y_pred); hold on; plot(y_pred,pred,'r','LineWidth',3); set(gca,'xtick',[]); set(gca,'ytick',[]); 
end
mean(accuracyholdout)
accuracyholdout

%%%%% holdout by GE/prisma scanner
[~,scores]=pca(Y(:, pval<0.05/12)); Ycomb(:,1)=scores(:,1);
%[~,scores]=pca(Y(:,9:12));Ycomb(:,2)=scores(:,1);
clear accuracy*
ix=d_OPT.scanner==1; %permutation_index = randperm(length(Ycomb));ix=zeros([length(Ycomb), 1]); ix(permutation_index(1:150))=1;
X_train=x(ix==0,:); X_test=x(ix==1 ,:);
Y_train=Ycomb(ix==0,:); Y_test=Ycomb(ix==1 ,:);
dim=2;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X_train,Y_train,dim);PCTVAR;
r=corr(XS, Y_train)
y_pred = [ones(size(X_test ,1),1) X_test]*BETA;
r=corr(Y_test, y_pred)
figure; scatter(y_pred, Y_test, 'x', 'k'); p = polyfit(y_pred,Y_test,1); pred = polyval(p,y_pred); hold on; plot(y_pred,pred,'r','LineWidth',3); set(gca,'xtick',[]); set(gca,'ytick',[]); 

%% wmh
aseg=readtable('C:\Users\peter\Documents\OPT\OPT\data\brain_age\aseg_stats.txt');

aseg=innerjoin(aseg, d_OPT);
aseg.Ventricles=aseg.Left_Lateral_Ventricle+aseg.Left_Inf_Lat_Vent+aseg.x3rd_Ventricle+aseg.x4th_Ventricle+aseg.x5th_Ventricle+aseg.Right_Lateral_Ventricle+aseg.Left_choroid_plexus+aseg.Right_choroid_plexus;

aseg.WMH=aseg.WM_hypointensities./aseg.EstimatedTotalIntraCranialVol;d_OPT.WMH=aseg.WMH;
[rr,pp]=corr(YS, sqrt(aseg.WMH), 'rows', 'pairwise')
[rr,pp]=corr(XS, sqrt(aseg.WMH), 'rows', 'pairwise')


%% CIRSG
cirsg=readtable('C:\Users\peter\Documents\OPT\OPT\data\CIRS-G_N_496.xlsx');
cirsg=innerjoin(cirsg, d_OPT);

[rr,pp]=corr(YS, cirsg.cirsgtotal, 'rows', 'pairwise')
[rr,pp]=corr(XS, cirsg.cirsgtotal, 'rows', 'pairwise')


%% compare 2021 vs 2023
fa2021=readtable("FA_2021.csv");
fa2023=readtable("FA_2023.csv");

corr(facombined.T_AF_left_fa2021,facombined.T_AF_left_fa2023)

facombined=innerjoin(fa2021,fa2023,'LeftKeys',1,'RightKeys',1);
facombined.Properties.VariableNames(75)
r=corr(facombined{:,2:74}, facombined{:,75:147}, 'rows','pairwise');

r(eye(73)==1)


%%

load('step2_DTI_holdout100.mat');figure(2);hold off;plot(X,Y,'Color',[0/255 180/255 0/255]);AUC_all(1)=AUC
AUC_test_all(3,:)=AUC_test;
load('step2_COGMADRS_noDTI_holdout100.mat');figure(2);hold on;plot(X,Y,'Color',[204/255 0 204/255]); AUC_all(2)=AUC
AUC_test_all(2,:)=AUC_test;
load('step2_MADRS_noDTI_holdout100.mat');figure(2);hold on;plot(X,Y,'Color',[255/255 0 127/255]);set(gca,'box','off'); AUC_all(3)=AUC
AUC_test_all(1,:)=AUC_test; %figure; b = bar(AUC_all,'k');
figure; yyaxis right; set(gca, 'color', 'none'); b = bar(mean(AUC_test_all'));
SEM = std(AUC_test_all')/sqrt(100);
hold on; [ngroups,nbars] = size(mean(AUC_test_all'));
tmp = nan(nbars, ngroups); for i = 1:nbars    tmp(i) = b.XEndPoints(i); end
errorbar(tmp',mean(AUC_test_all'), 2*SEM,'k','linestyle','none');ylim([0.5 0.8]);hold on
b = bar([1; 2; 3],diag(mean(AUC_test_all')),'stacked');
b(1).FaceColor=[255/255 0 127/255];b(2).FaceColor=[204/255 0 204/255]; b(3).FaceColor=[0/255 180/255 0/255];
