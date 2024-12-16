%% OPT rsfmri analysis
clear
%OPT data import
%OPT NEURO
load('C:\Users\peter\Documents\OPT\OPT\reports\rsfc_2023\OPT_rsfmri_281_12.mat');
%load('C:\Users\peter\Documents\OPT\OPT\reports\rsfc_2021\feat.mat');
cd C:\Users\peter\Documents\OPT\OPT\reports\rsfc_2023

for sub=1:length(subjects)
rmat=reshape(Fnetmats_all(sub, :,:), [21 21]);D=21;
full_icad25(sub,:)=rmat(triu(ones(D),1)==1)';
rmat=reshape(Pnetmats_all(sub, :,:), [21 21]);D=21;
part_icad25(sub,:)=rmat(triu(ones(D),1)==1)';
%rmat=reshape(Rmat_all(sub, :,:), [21 21]);D=21; %rmat=r2z(rmat);
%rmat_icad25(sub,:)=rmat(triu(ones(D),1)==1)';
end; clear rmat Fnetmats_all Pnetmats_all %% reshape the netmats to one line from the square form
%%% Additional bl subjects
load('C:\Users\peter\Documents\OPT\OPT\reports\rsfc_2023\OPT_rsfmri_5_12.mat');


ix=isnan(full_icad25(:,1));ix=isnan(mean_fd');
subjects(ix==1)=[];full_icad25(ix,:)=[]; part_icad25(ix,:)=[];mean_fd(ix)=[];
subjects=table(subjects, 'VariableNames',{'subs'});
try; subjects.ID_complete={'CU10001';'CU10003';'CU10007';'CU10009';'CU10010';'CU10012';'CU10014';'CU10021';'CU10022';'CU10023';'CU10024';'CU10026';'CU10031';'CU10033';'CU10034';'CU10038';'CU10045';'CU10056';'CU10057';'CU10060';'CU10070';'CU10074';'CU10076';'CU10078';'CU10079';'CU10081';'CU10083';'CU10084';'CU10088';'CU10089';'CU10091';'CU10098';'CU10099';'CU10101';'CU10104';'CU10109';'CU10110';'CU10114';'CU10119';'CU10141';'CU10146';'CU20002';'CU20003';'CU20005';'CU20006';'CU20011';'CU20015';'CU20016';'LA10006';'LA10008';'LA10026';'LA10032';'LA10035';'LA10038';'LA10040';'LA10044';'LA10048';'LA10054';'LA10056';'LA10074';'LA10099';'LA10104';'LA10105';'LA20002';'LA20012';'LA20019';'LA20021';'LA20032';'LA20039';'LA20043';'UP10004';'UP10007';'UP10044';'UP10047';'UP10057';'UP10071';'UP10074';'UP10076';'UP10077';'UP10092';'UP10094';'UP10110';'UP10124';'UP10155';'UP10266';'UP10280';'UP10005';'UP10021';'UP10046';'UP10053';'UP10061';'UP10062';'UP10087';'UP10109';'UP10111';'UP10133';'UP10137';'UP10148';'UP10165';'UP10203';'UP10206';'UP10231';'UP10258';'UP10261';'UP10001';'UP10003';'UP10006';'UP10009';'UP10020';'UP10026';'UP10032';'UP10049';'UP10054';'UP10058';'UP10066';'UP10080';'UP10081';'UP10090';'UP10096';'UP10098';'UP10101';'UP10112';'UP10113';'UP10114';'UP10125';'UP10126';'UP10128';'UP10130';'UP10135';'UP10136';'UP10151';'UP10156';'UP10158';'UP10161';'UP10163';'UP10171';'UP10173';'UP10175';'UP10184';'UP10187';'UP10188';'UP10196';'UP10201';'UP10204';'UP10209';'UP10210';'UP10229';'UP10244';'UP10250';'UT10006';'UT10018';'UT10023';'UT10066';'UT10067';'UT10072';'UT10073';'UT10075';'UT10078';'UT10079';'UT10080';'UT10083';'UT10091';'UT10109';'UT10111';'UT10115';'UT10116';'UT10125';'UT10126';'UT10128';'UT10130';'UT10137';'UT10148';'UT30006';'UT30008';'UT30011';'UT30018';'UT30019';'UT30021';'UT30025';'UT30026';'UT30027';'UT30028';'UT30030';'UT30032';'UT30033';'UT30034';'UT30036';'UT30040';'UT30042';'UT10001';'UT10003';'UT10004';'UT10013';'UT10015';'UT10016';'UT10019';'UT10021';'UT10025';'UT10033';'UT10035';'UT10039';'UT10046';'UT10054';'UT10081';'UT10087';'UT10090';'UT10114';'UT10120';'UT10123';'UT10144';'UT30003';'UT30029';'UT30031';'UT30039';'UT30041';'UT30043';'WU10089';'WU10031';'WU10055';'WU10057';'WU10085';'WU10099';'WU10105';'WU10114';'WU10115';'WU10117';'WU10127';'WU10131';'WU10137';'WU10146';'WU10150';'WU10152';'WU10157';'WU10161';'WU10164';'WU10168';'WU10174';'WU10189';'WU20001';'WU20002';'WU20004';'WU20005';'WU20006';'WU10008';'WU10014';'WU10028';'WU10034';'WU10036';'WU10054';'WU10058';'WU10059';'WU10062';'WU10064';'WU10066';'WU10071';'WU10073';'WU10077';'WU10083';'WU10087';'WU10092';'WU10093';'WU10096';'WU10104';'WU10107';'WU10108';'WU10112';'WU10121';'WU10122';'WU10124';'WU10128';'WU10135';'WU10141';'WU10145';'WU10147';'WU10149';'WU10151';'WU10154';'WU10158';'WU10165';'WU10166';'WU10177'};end
try; subjects.ID_complete={'CU10001';'CU10007';'CU10009';'CU10010';'CU10012';'CU10014';'CU10021';'CU10022';'CU10023';'CU10026';'CU10031';'CU10033';'CU10034';'CU10038';'CU10045';'CU10056';'CU10057';'CU10060';'CU10070';'CU10074';'CU10076';'CU10078';'CU10079';'CU10081';'CU10083';'CU10084';'CU10088';'CU10089';'CU10091';'CU10098';'CU10099';'CU10101';'CU10104';'CU10109';'CU10110';'CU10114';'CU10119';'CU10141';'CU10146';'CU20002';'CU20003';'CU20005';'CU20006';'CU20011';'CU20015';'CU20016';'LA10006';'LA10008';'LA10026';'LA10032';'LA10035';'LA10038';'LA10040';'LA10044';'LA10048';'LA10054';'LA10056';'LA10074';'LA10099';'LA10104';'LA10105';'LA20012';'LA20019';'LA20021';'LA20032';'LA20039';'LA20043';'UP10004';'UP10007';'UP10044';'UP10047';'UP10057';'UP10071';'UP10074';'UP10076';'UP10077';'UP10092';'UP10094';'UP10110';'UP10124';'UP10155';'UP10266';'UP10280';'UP10005';'UP10021';'UP10046';'UP10053';'UP10061';'UP10062';'UP10087';'UP10109';'UP10111';'UP10133';'UP10137';'UP10148';'UP10165';'UP10203';'UP10206';'UP10231';'UP10258';'UP10261';'UP10001';'UP10003';'UP10006';'UP10009';'UP10020';'UP10026';'UP10032';'UP10049';'UP10054';'UP10058';'UP10066';'UP10080';'UP10081';'UP10090';'UP10096';'UP10098';'UP10101';'UP10112';'UP10113';'UP10114';'UP10125';'UP10126';'UP10128';'UP10130';'UP10135';'UP10136';'UP10151';'UP10156';'UP10158';'UP10161';'UP10163';'UP10171';'UP10173';'UP10175';'UP10184';'UP10187';'UP10188';'UP10196';'UP10201';'UP10204';'UP10209';'UP10210';'UP10229';'UP10244';'UP10250';'UT10006';'UT10018';'UT10023';'UT10066';'UT10067';'UT10072';'UT10073';'UT10075';'UT10078';'UT10079';'UT10080';'UT10083';'UT10091';'UT10109';'UT10111';'UT10115';'UT10116';'UT10125';'UT10126';'UT10128';'UT10130';'UT10137';'UT10148';'UT30006';'UT30008';'UT30011';'UT30018';'UT30019';'UT30021';'UT30025';'UT30026';'UT30027';'UT30028';'UT30030';'UT30032';'UT30033';'UT30034';'UT30036';'UT30040';'UT30042';'UT10003';'UT10004';'UT10013';'UT10015';'UT10016';'UT10019';'UT10021';'UT10025';'UT10033';'UT10035';'UT10039';'UT10046';'UT10054';'UT10081';'UT10087';'UT10090';'UT10114';'UT10120';'UT10123';'UT10144';'UT30003';'UT30029';'UT30031';'UT30039';'UT30041';'UT30043';'WU10089';'WU10031';'WU10055';'WU10057';'WU10085';'WU10099';'WU10105';'WU10114';'WU10115';'WU10117';'WU10127';'WU10131';'WU10137';'WU10146';'WU10150';'WU10152';'WU10157';'WU10161';'WU10164';'WU10168';'WU10174';'WU10189';'WU20001';'WU20002';'WU20004';'WU20005';'WU20006';'WU10008';'WU10014';'WU10028';'WU10034';'WU10036';'WU10054';'WU10058';'WU10059';'WU10062';'WU10064';'WU10066';'WU10071';'WU10073';'WU10077';'WU10083';'WU10087';'WU10092';'WU10093';'WU10096';'WU10104';'WU10107';'WU10108';'WU10112';'WU10121';'WU10122';'WU10124';'WU10128';'WU10135';'WU10141';'WU10145';'WU10147';'WU10149';'WU10151';'WU10154';'WU10158';'WU10165';'WU10166';'WU10177'};end
qcfail={'CU10010';'CU10114';'LA10040';'UP10094';'UP10124';'UP10049';'UP10081';'UP10096';'UP10114';'WU10014'};
clear ix; for i=1:length(qcfail)
    ix(:,i)=strcmp(qcfail(i), subjects.ID_complete);
end; ix=sum(ix')';
subjects.rsfc=full_icad25;subjects.part_rsfc=part_icad25;subjects.mean_fd=mean_fd';
subjects(ix==1,:)=[]; subjects.scanner= ~cellfun(@isempty,(strfind(subjects.ID_complete, 'CU'))) |  ~cellfun(@isempty,(strfind(subjects.subs, 'UT1UT1')))  |  ~cellfun(@isempty,(strfind(subjects.subs, 'UT1UT3'))) ;
d=readtable('C:\Users\peter\Documents\OPT\OPT\data\ON_DATASET_08302024.xlsx'); d.ID=d.ID_complete;
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

sum(~isnan(d.AIS_01))
sum(~isnan(d.IMIS_01))
sum(~isnan(d.MDMIS_01))
sum(~isnan(d.LIS_01))
sum(~isnan(d.MVCIS_01))
sum(~isnan(d.EXEC_01))
sum(d.MCI_01==1)
sum(d.DEM_01==1)
sum(d.NCD_01==1)

%corr(Y, 'rows', 'pairwise')
%% 
demo_OPT=innerjoin(d, subjects);
full_icad25=demo_OPT.rsfc;part_icad25=demo_OPT.part_rsfc;mean_fd=demo_OPT.mean_fd;
Y=[demo_OPT.AIS_01, demo_OPT.IMIS_01, demo_OPT.MDMIS_01, demo_OPT.LIS_01, demo_OPT.MVCIS_01, demo_OPT.EXEC_01];
figure; for i=1:6;     subplot(2,3,i);histogram(Y(:,i)); end

%% AFTER ComBatH
batch = demo_OPT.SITE_MRI_1_WU_2_UT_3_UP_4_LA_5_CU';%batch=demo_OPT.scanner';
dat = part_icad25';
age = demo_OPT.AGE;
sex = demo_OPT.GENDER;
sex = dummyvar(sex);
mod = [demo_OPT.RACE demo_OPT.HL age sex(:,2) mean_fd];
part_harmonized = combat(dat, batch, mod, 1);part_harmonized=part_harmonized';
dat = full_icad25';
full_harmonized = combat(dat, batch, mod, 1);
full_harmonized=full_harmonized';
[rval,pval]=corr(full_harmonized, demo_OPT.MDMIS_01, 'rows', 'pairwise'); %figure; scatter(full_harmonized(:,191), demo_OPT.RSM_Z)
min(rval)
min(pval)
%ICs_vector(mafdr(p, 'BHFDR', true)<0.01) %r(mafdr(p, 'BHFDR', true)<0.01)
%% PLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y=[demo_OPT.AIS_01, demo_OPT.IMIS_01, demo_OPT.MDMIS_01, demo_OPT.LIS_01, demo_OPT.MVCIS_01, demo_OPT.EXEC_01];

demo_OPT.DTMTS4_01+demo_OPT.CWI3CSSFinal_01;
X=full_harmonized;
%%% remove nans
naninx=sum(isnan(X)')'>0| sum(isnan(Y)')'>0 | mean_fd>0.7; Y=Y(naninx==0,:);  X=X(naninx==0,:);  d_OPT=demo_OPT(naninx==0,:);
Y=zscore(Y); X=zscore(X);x=X;
ncomp=3
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,Y,ncomp);PCTVAR
%% permutation testing
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
[c,pval]=corr(XS, Y); c(1,:)=c(1,:)*-1; %c(3,:)=c(3,:)*-1; % c(2,:)=c(2,:)*-1;   reshape(mafdr(reshape(pval, [1 36]), 'BHFDR', 'True'), [4 9])
figure;imagesc(c(2,:)); colormap bone; colorbar
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
%%
ICs={'IC1';'IC2';'IC3';'IC4';'IC5';'IC6';'IC7';'IC8';'IC9';'IC10';'IC11';'IC12';'IC13';'IC14';'IC15';'IC16';'IC17';'IC18';'IC19';'IC20';'IC21'};
combs=allcomb(ICs, ICs);
for i=1:441; comb(i)=cellstr(strcat(combs{i,2}, combs{i,1})); end
ICs=reshape(comb, [21,21]); clear comb combs i
ICs_vector=ICs(triu(ones(D),1)==1)';

ICs_vector(plsweights2>3)
ICs_vector(plsweights2<-3)
ICs_vector(plsweights3>3)
ICs_vector(plsweights3<-3)
ICs_vector(plsweights1>3)
ICs_vector(plsweights1<-3)

%%
figure; scatter(XS(:,2), YS(:,2), 'filled', 'k');lsline;
corr(XS(:,2), YS(:,2))
%corr(XS(:,3), YS(:,3))
tmp=sum(PCTVAR');
figure; histogram(Rsq); xline(tmp(2));

%    corr(X(:,plsweights2<-3), Y)
sum(d_OPT.MCI_01==1)
sum(d_OPT.DEM_01==1)
sum(d_OPT.NCD_01==1)

%%
ica2yeo7=readtable('D:\ukb\ica2yeo7.csv');
D=21; ICs={'IC1';'IC7';'IC9';'IC13';'IC14';'IC20';'IC21';'IC5';'IC6';'IC16';'IC10';'IC11';'IC12';'IC17';'IC3';'IC2';'IC4';'IC8';'IC19';'IC15';'IC18'};
combs=allcomb(ICs, ICs);
for i=1:441; comb(i)=cellstr(strcat(combs{i,2}, combs{i,1})); end
ICs=reshape(comb, [21,21]); clear comb combs i
ICs_vector=ICs(triu(ones(D),1)==1)';

ic_mapping=readtable('D:\ukb\ic_order_mapping.csv');
tmp=ic_mapping.original_order_num;

myLabel=ica2yeo7.Yeo7N;myLabel=myLabel([1 7 9 13 14 20 21 5 6 16 10 11 12 17 3 2 4 8 19 15 18]);
myLabel=repmat({''}, [21,1]); figure
upperlim=3; lowerlim=-3; %upperlim=nodes(205,component); lowerlim=nodes(5,component);
Weights=plsweights3(tmp); Weights(Weights<upperlim & Weights>lowerlim)=0;%Weights=stats.W(:,component); 
Weights(Weights<0)=0; Weights_square= zeros(21); Weights_square(triu(ones(21),1)>0) = abs(Weights);
hold on; myColorMap=zeros(21,3);myColorMap(:,3)=1; circularGraph(Weights_square, 'Colormap',myColorMap, 'Label',myLabel);
Weights=plsweights3(tmp); Weights(Weights<upperlim & Weights>lowerlim)=0;
Weights(Weights>0)=0; Weights_square= zeros(21); Weights_square(triu(ones(21),1)>0) = abs(Weights);
hold on; myColorMap=zeros(21,3);myColorMap(:,1)=1; circularGraph(Weights_square, 'Colormap',myColorMap, 'Label',myLabel);

%%
%%
%%
%% Longitudinal relationship with MADRS and PHQ
madrs=readtable('C:\Users\peter\Documents\OPT\OPT\data\OPT N MADRS Longitudinally with Study Meds 20230921');madrs(strcmp(madrs.ID, 'UP10209'),:)=[];
%madrs=readtable('C:\Users\peter\Documents\OPT\OPT\data\OPTIMUMMainDatabaseF_DATA_2023-07-03_0248_ALL_MADRS.csv');


madrs=readtable('C:\Users\peter\Documents\OPT\OPT\data\MADRS_Longitudinally_20230711_updated');
subs=table(d_OPT.ID_complete, 'VariableNames',{'ID'});subs.NPDateDays=d_OPT.NPDateDaysKaylaUPDATE;
%madrs=outerjoin(subs, madrs);

for i=1:length(subs.ID);try
    sub=subs.ID{i};
    data=madrs(strcmp(madrs.ID, sub),:);
    MADRS_BL(i)=data.madrs_tot_scr(1);
    data.DaysDiff=data.MADRSDateDays-d_OPT.NPDateDaysKaylaUPDATE(strcmp(d_OPT.ID_complete, sub));
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
[p,t,stats]=anova1(YS(:,2)*-1, MADRS_groups); [c,m,h,gnames] = multcompare(stats, 'CriticalValueType','hsd');
figure; violinplot(YS(:,2)*-1, MADRS_groups);xlim([0.5 3.5])
figure;gscatter(d_OPT.AGE, YS(:,2), MADRS_groups);lsline

MADRS_groups(MADRS_groups<2)=0; MADRS_groups(MADRS_groups==2)=1; 
t=table(YS(:,2),MADRS_groups, d_OPT.AGE,  MADRS_BL', 'VariableNames',{'YS', 'MADRSresp','AGE', 'MADRS_BL'});
%t(t.MADRSresp==1,:)=[];
model = fitglm(t,'MADRSresp ~ YS+AGE','link','logit','Distribution','binomial')
scores = model.Fitted.Probability;
[X,Y,T,AUC] = perfcurve(t.MADRSresp,scores, 1);AUC
figure;plot(X,Y)
[B,dev,stats] = mnrfit([t.YS, t.AGE], t.MADRSresp+1)

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
madrs_cut.CDRSCORE_01( madrs_cut.CDRSCORE_01>2.5)=NaN;madrs_cut.CDRSUMBOX_01( madrs_cut.CDRSUMBOX_01>=2)=NaN;

madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'month'))) ;madrs_cut(~madrs_cut.remission==0,:)=[];
madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'mth'))) ;madrs_cut(~madrs_cut.remission==0,:)=[];
madrs_cut.remission=madrs_cut.madrs_tot_scr<=10;

%% MLE
%mdl=fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d_delta+MDMIS_01+(1+time_in_d_delta|ID_madrs_cut)' )
%mdl=fitlme(madrs_cut, 'MDMIS_01~AGE+GENDER+time_in_d_delta+remission+(1|ID_madrs_cut)' )
mdl=fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d_delta*MTOTALIS_01+(1+time_in_d_delta|ID_madrs_cut)' )
mdl=fitlme(madrs_cut(madrs_cut.MTOTALIS_01>110,:), 'madrs_tot_scr~AGE+GENDER+time_in_d_delta+(1+time_in_d_delta|ID_madrs_cut)' )
madrs_cut.MTOTALIS_01;
%mdl=fitlme(madrs_cut, 'MTOTALIS_01~AGE+GENDER+remission+time_in_d_delta+(1+time_in_d_delta|ID_madrs_cut)' )

madrs_cut.IC9IC11=madrs_cut.rsfc(:, strcmp(ICs_vector,'IC9IC11'));
madrs_cut.IC14IC20=madrs_cut.rsfc(:, strcmp(ICs_vector,'IC14IC20'));
madrs_cut.IC1IC11=madrs_cut.rsfc(:, strcmp(ICs_vector,'IC1IC11'));
fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d*IC14IC20+(1+time_in_d|ID_madrs_cut)' )
fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d*IC9IC11+(1+time_in_d|ID_madrs_cut)' )


predmadrs=table;
for i=1:210
    t=madrs_cut; t.fmri=t.fmri(:,i);
    mdl=fitlme(t, 'madrs_tot_scr~AGE+GENDER+time_in_d*fmri+(1+time_in_d|ID_madrs_cut)' );
    predmadrs.p(i)=mdl.Coefficients.pValue(6); predmadrs.t(i)=mdl.Coefficients.tStat(6);
    %mdl=fitlm(t, 'fmri~AGE+GENDER+remission' );
    %predmadrs.p(i)=mdl.Coefficients.pValue(4); predmadrs.t(i)=mdl.Coefficients.tStat(4);
end; predmadrs.ICs=ICs_vector';

%% 1/0 prediction -  step 1
madrs_cut.madrs_tot_scr( ~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'end'))) )
madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'end'))) ;
madrs_cut(madrs_cut.remission==0,:)=[];
madrs_cut.remission=madrs_cut.madrs_tot_scr<=10;

madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'step_2'))) ; %step 2 and arm 7 or step 1 and arm 8
madrs_cut(madrs_cut.remission==1,:)=[];
madrs_cut.remission=~cellfun(@isempty,(strfind(madrs_cut.redcap_event_name, 'arm_7'))) ;
madrs_cut(madrs_cut.remission==1,:)=[];
madrs_cut.remission=madrs_cut.madrs_tot_scr<=10;
%madrs_cut(madrs_cut.OnAripiprazole~=1,:)=[];%madrs_cut(madrs_cut.OnBupropion~=1,:)=[];%madrs_cut(madrs_cut.OnNortriptyline~=1,:)=[];%


%anova1(madrs_cut.RC_Z, madrs_cut.remission);
Xlogist=[madrs_cut.rsfc, madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr, madrs_cut.AIS_01,madrs_cut.IMIS_01,madrs_cut.MDMIS_01,madrs_cut.EXEC_01];
varnames=horzcat(ICs_vector,{'age'},{'sex'}, {'baseline_madrs_scr'},{'attn'},{'imMem'},{'delMem'},{'exec'});
%Xlogist=[madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr, madrs_cut.DTMT4_Scaled, madrs_cut.RC_Z]; varnames=horzcat({'age'},{'sex'}, {'baseline_madrs_scr'},{'tmt'},{'coding'});
%Xlogist=[madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr]; varnames=horzcat({'age'},{'sex'}, {'baseline_madrs_scr'});
Ylogist=madrs_cut.remission;

[B,FitInfo] = lassoglm(Xlogist,Ylogist,'binomial','alpha',0.5,'CV',10, 'PredictorNames',varnames);
%lassoPlot(B,FitInfo,'PlotType','CV'); legend('show','Location','best') % show legend
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
MinModelPredictors = FitInfo.PredictorNames(B(:,idxLambdaMinDeviance)~=0)
idxLambda1SE = FitInfo.Index1SE;
sparseModelPredictors = FitInfo.PredictorNames(B(:,idxLambda1SE)~=0)
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; B(:,idxLambdaMinDeviance)];
yhat = glmval(coef,Xlogist,'logit');[X,Y,T,AUC] = perfcurve(Ylogist,yhat, 1);AUC
%Xlogist=Xlogist(:,B(:,idxLambdaMinDeviance)~=0);varnames=varnames(B(:,idxLambdaMinDeviance)~=0); %pasimonious model test > set alpha to very low 0.001

figure;coef_all=[];confusionmatrices=[];yhat_all=[];ytest_all=[];yhatBinom_all=[];
for i=1:10
    %if(i==1); ix=zeros([304,1]); ix(1:30)=1; elseif (i==10)
    %ix=zeros([304,1]); ix(30*(i-1)+1:30*(i-1)+34)=1; else
    %ix=zeros([304,1]); ix(30*(i-1)+1:30*(i-1)+30)=1; end
permutation_index = randperm(length(Ylogist));ix=zeros([length(madrs_cut.ID_madrs_cut), 1]);
ix(permutation_index(1:30))=1;  %ix(permutation_index(31:304))=0;  
XTest=Xlogist(ix==1,:);XTrain=Xlogist(ix==0,:); 
yTest=Ylogist(ix==1);yTrain=Ylogist(ix==0);

[B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','CV',10, 'PredictorNames', varnames,'alpha', 0.5);
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
AUC_test
[X,Y,T,AUC] = perfcurve(ytest_all,yhat_all, 1);AUC %sum(confusionmatrices)

[sensitivity, specificity, accuracy, F1score]=gofmeasures_3d_square(confusionmatrices)
yhatBinom_all=yhat_all>0.28;figure;c=confusionchart(ytest_all,double(yhatBinom_all))
[sensitivity, specificity, accuracy, F1score]=gofmeasures_2d_square(c.NormalizedValues)



load('RSFMRI_holdoutAUC_rerun100.mat');figure(2);hold on;plot(X,Y,'Color',[160/255 160/255 0]);AUC_all(1)=AUC
AUC_test_all(3,:)=AUC_test;
load('COGMADRS_onlyNoRSFMRI_holdoutAUC_rerun100.mat');figure(2);hold on;plot(X,Y,'Color',[204/255 0 204/255]); AUC_all(2)=AUC
AUC_test_all(2,:)=AUC_test;
load('MADRS_only_noRSFMRI_holdoutAUC_100.mat');figure(2);hold on;plot(X,Y,'Color',[255/255 0 127/255]); set(gca,'box','off');AUC_all(3)=AUC
AUC_test_all(1,:)=AUC_test; %figure; b = bar(AUC_all,'k');
figure; yyaxis right; set(gca, 'color', 'none'); b = bar(mean(AUC_test_all'));
SEM = std(AUC_test_all')/sqrt(100);
hold on; [ngroups,nbars] = size(mean(AUC_test_all'));
tmp = nan(nbars, ngroups); for i = 1:nbars    tmp(i) = b.XEndPoints(i); end
errorbar(tmp',mean(AUC_test_all'), 2*SEM,'k','linestyle','none');ylim([0.5 0.8]);hold on
b = bar([1; 2; 3],diag(mean(AUC_test_all')),'stacked');
b(1).FaceColor=[255/255 0 127/255];b(2).FaceColor=[204/255 0 204/255]; b(3).FaceColor=[160/255 160/255 0];
%% NAIVE BAYES
Xlogist=[madrs_cut.rsfc, madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr, madrs_cut.DTMT4_Scaled, madrs_cut.RC_Z];
varnames=horzcat(ICs_vector,{'age'},{'sex'}, {'baseline_madrs_scr'},{'tmt'},{'coding'});
%Xlogist=[madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr, madrs_cut.RC_Z, madrs_cut.DTMT4_Scaled]; varnames=[{'age'}, {'sex'},{'blmadrs'}, 'coding', 'tmt4'];%, 'tmt5','rsm','rsr'
%Xlogist=[madrs_cut.AGE, madrs_cut.GENDER, madrs_cut.baseline_madrs_scr]; varnames=[{'age'}, {'sex'},{'blmadrs'}];%, 'tmt5','rsm','rsr'
Ylogist=madrs_cut.remission; Ylogist=zeros(length(Ylogist),1);
Ylogist(madrs_cut.OnBupropion==1 & madrs_cut.madrs_tot_scr<=10)=2;
Ylogist(madrs_cut.OnAripiprazole==1 & madrs_cut.madrs_tot_scr<=10)=3; Ylogist=categorical(Ylogist);

Mdl = fitcnb(Xlogist,Ylogist);
predLabels = resubPredict(Mdl); %resub is out of sample
ConfusionMat1 = confusionchart(Ylogist,predLabels);
[recall1, recall2, recall3, accuracy]=gofsquare_x3(ConfusionMat1.NormalizedValues)
% explain shaps values for 1 person

%%
model = fitglm(madrs_cut,['remission ~ RSM_Z+RSR_Z+RC_Z+DCSSS+DCFTCSS+DTMT4_Scaled+DTMT5_Scaled+CWI4CSSfinal+CWI3CSSfinal' ...
    '+AGE+GENDER+ED+baseline_madrs_scr'],'link','logit','Distribution','binomial') %lme4/nlme
scores = model.Fitted.Probability; [X,Y,T,AUC] = perfcurve(madrs_cut.remission,scores, 1);AUC
figure;plot(X,Y)

madrs_cut.IC9IC11=madrs_cut.rsfc(:, strcmp(ICs_vector,'IC9IC11'));
madrs_cut.IC3IC4=madrs_cut.rsfc(:, strcmp(ICs_vector,'IC3IC4'));
madrs_cut.IC4IC5=madrs_cut.rsfc(:, strcmp(ICs_vector,'IC4IC5'));
madrs_cut.IC14IC20=madrs_cut.rsfc(:, strcmp(ICs_vector,'IC14IC20'));
madrs_cut.IC1IC11=madrs_cut.rsfc(:, strcmp(ICs_vector,'IC1IC11'));
madrs_cut.IC9IC20=madrs_cut.rsfc(:, strcmp(ICs_vector,'IC9IC20'));

model = fitglm(madrs_cut,['remission ~ IC9IC11+IC14IC20+IC3IC4+IC9IC20' ...
    '+DTMT4_Scaled+RC_Z' ...
    '+AGE+GENDER+ED+baseline_madrs_scr'],'link','logit','Distribution','binomial') %lme4/nlme
scores = model.Fitted.Probability; [X,Y,T,AUC] = perfcurve(madrs_cut.remission,scores, 1);AUC


%[r,p]=corr(madrs_cut.rsfc, madrs_cut.DTMT4_Scaled)

violinplot(madrs_cut.madrs_tot_scr, madrs_cut.CDRSCORE)
figure;scatter(madrs_cut.RSM_Z, madrs_cut.madrs_tot_scr); lsline

fitlme(madrs_cut, 'madrs_tot_scr~AGE+GENDER+time_in_d_delta*RSR_Z+(1+time_in_d_delta|ID_madrs)' )


ylim([-5 50]); figure;histogram(b*90)

figure; scatter(MADRS_NP, YS(:,2), 15, 'k', 'filled'); lsline
corr(MADRS_NP', YS(:,2), 'rows', 'pairwise')
corr(MADRS_NP', XS(:,3), 'rows', 'pairwise')
clear tmp
for i=1:length(subs.ID);try
    sub=subs.ID{i};
    data=madrs(strcmp(madrs.ID, sub),:);
    tmp(i,1)=data.redcap_event_name(1);
catch    
end; end

%% Cross-sectional relationship with MADRS and PHQ
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
figure; scatter(d_OPT.ED, YS(:,2)*-1, 15, 'k', 'filled'); lsline
figure; scatter(d_OPT.ED, XS(:,2)*-1, 15, 'k', 'filled'); lsline; xlim([7 21])

figure; scatter(clinical_OPT.madrs_tot_scr, YS(:,2), 15, 'k', 'filled'); lsline

%%%%%%%% PHQ next
phq9=readtable('C:\Users\peter\Documents\OPT\OPT\data\PHQ-9 Closest to NP 20230625.xlsx');
subs=table(d_OPT.ID, 'VariableNames',{'ID'});
clinical_OPT=innerjoin(subs, phq9);

clinical_OPT.phq9_total_score(clinical_OPT.datediff>60)=NaN;

[rr,pp]=corr(clinical_OPT.phq9_total_score, YS, 'rows', 'pairwise')
[rr,pp]=corr(clinical_OPT.phq9_total_score, XS,'rows', 'pairwise')


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

figure; scatter(centile.centile, YS(:,2)*-1, 15, 'k', 'filled'); lsline
anova1(centile.centile, d_OPT.CDRSCORE)
anova1(YS(:,2), d_OPT.CDRSCORE)

%% relationship with treatment resistance severity
athf=readtable('C:\Users\peter\Documents\OPT\OPT\data\OPTN_ATHF_Summary_Scores_N453_May23.csv');
subs=table(d_OPT.ID, 'VariableNames',{'ID'});
clinical_OPT=innerjoin(subs, athf);
clinical_OPT=outerjoin(subs, clinical_OPT);

[rr,pp]=corr(clinical_OPT.athf_total_score_v2, YS, 'rows', 'pairwise')
[rr,pp]=corr(clinical_OPT.athf_total_score_v2, XS,'rows', 'pairwise')

figure; scatter(clinical_OPT.athf_total_score_v2, YS(:,2)*-1, 15, 'k', 'filled'); lsline

anova1(YS(:,2),clinical_OPT.athf_highest_trial_v2)
anova1(Y(:,4),clinical_OPT.athf_highest_trial_v2)

%% hold out results
[~,scores]=pca(Y(:, pval(2,:)<0.05/24));  %[~,~,~,~, F] = factoran(Y,1,'Rotate','none','maxit',100); Ycomb(:,1)=F;
Ycomb(:,1)=scores(:,1);%Ycomb=Y(:, pval(2,:)<0.05/24); Ycomb(:,1)=mean(Y(:, pval(2,:)<0.05/24)')';
clear accuracy*

l=length(X(:,1)); 
%randomsample=randperm(l); X=X(randomsample,:);Y=Y(randomsample,:);
l=round(0.25*(l),0); 
%for h=1:5 %ixtest=d_OPT.SITE_MRI_1_WU_2_UT_3_UP_4_LA_5_CU==h;
figure(11); hold off

for h=1:4
xstart=(h-1)*l+1;
xend=h*l;if (h==4);xend=length(X(:,1));end
ixtest=zeros(1,length(X(:,1)));
ixtest(xstart:xend)=1;

ixtest=d_OPT.SITE_MRI_1_WU_2_UT_3_UP_4_LA_5_CU==h;
X_train=x(ixtest==0,:); X_test=x(ixtest==1 ,:);
Y_train=Ycomb(ixtest==0,:); Y_test=Ycomb(ixtest==1 ,:);
%l=length(X(:,1)); l=round(0.25*l,0); X_test=x(1:l,:); X_train=x(l+1:length(X(:,1)) ,:);Y_test=Y(1:l,:); Y_train=Y(l+1:length(X(:,1)) ,:);
dim=2;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X_train,Y_train,dim);PCTVAR;
r=corr(XS, Y_train); accuracytrain(h,:)=r(2,:);
y_pred = [ones(size(X_test ,1),1) X_test]*BETA;
r=corr(Y_test, y_pred);accuracyholdout(h,:)=r(eye(length(Ycomb(1,:)))==1)';
subplot(4,1,h); scatter(y_pred, Y_test, 'x', 'k'); p = polyfit(y_pred,Y_test,1); pred = polyval(p,y_pred); hold on; plot(y_pred,pred,'r','LineWidth',3); set(gca,'xtick',[]); set(gca,'ytick',[]); 
end
mean(accuracyholdout)
accuracyholdout

%%%%% holdout by GE/prisma scanner
[~,scores]=pca(Y(:, pval(2,:)<0.05/24)); Ycomb(:,1)=scores(:,1);%Ycomb=Y(:, pval(2,:)<0.05/24);
%[~,scores]=pca(Y(:,9:12));Ycomb(:,2)=scores(:,1);
clear accuracy*
ix=d_OPT.scanner==1; %permutation_index = randperm(length(Ycomb));ix=zeros([length(Ycomb), 1]); ix(permutation_index(1:150))=1;
X_train=x(ix==0,:); X_test=x(ix==1 ,:);
Y_train=Ycomb(ix==0,:); Y_test=Ycomb(ix==1 ,:);
dim=3;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X_train,Y_train,dim);PCTVAR;
r=corr(XS, Y_train)
y_pred = [ones(size(X_test ,1),1) X_test]*BETA;
r=corr(Y_test, y_pred)
figure; scatter(y_pred, Y_test, 'x', 'k'); p = polyfit(y_pred,Y_test,1); pred = polyval(p,y_pred); hold on; plot(y_pred,pred,'r','LineWidth',3); set(gca,'xtick',[]); set(gca,'ytick',[]); 

y_pred_all=[]; y_test_all=[];for i=1:1000
permutation_index = randperm(length(Ycomb));ix=zeros([length(Ycomb), 1]); ix(permutation_index(1:79))=1;
X_train=x(ix==0,:); X_test=x(ix==1 ,:);
Y_train=Ycomb(ix==0,:); Y_test=Ycomb(ix==1 ,:);
dim=2;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X_train,Y_train,dim);PCTVAR;
%r=corr(XS, Y_train);
y_pred = [ones(size(X_test ,1),1) X_test]*BETA;
r(i)=corr(Y_test, y_pred);y_test_all=[y_test_all; Y_test]; y_pred_all=[y_pred_all; y_pred];
end
mean(r)
figure; histogram(r); 
figure; scatter(y_test_all, y_pred_all);lsline;corr(y_test_all, y_pred_all)
%% WMH
aseg=readtable('C:\Users\peter\Documents\OPT\OPT\data\brain_age\aseg_stats.txt');

aseg=innerjoin(aseg, d_OPT);
aseg.Ventricles=aseg.Left_Lateral_Ventricle+aseg.Left_Inf_Lat_Vent+aseg.x3rd_Ventricle+aseg.x4th_Ventricle+aseg.x5th_Ventricle+aseg.Right_Lateral_Ventricle+aseg.Left_choroid_plexus+aseg.Right_choroid_plexus;

aseg.WMH=aseg.WM_hypointensities./aseg.EstimatedTotalIntraCranialVol;
[rr,pp]=corr(YS, aseg.WMH, 'rows', 'pairwise')
[rr,pp]=corr(XS, aseg.WMH, 'rows', 'pairwise')

%% cirsg
cirsg=readtable('C:\Users\peter\Documents\OPT\OPT\data\CIRS-G_N_496.xlsx');
cirsg=innerjoin(cirsg, d_OPT);

[rr,pp]=corr(YS, cirsg.cirsgtotal, 'rows', 'pairwise')
[rr,pp]=corr(XS, cirsg.cirsgtotal, 'rows', 'pairwise')

%% demographic stats
%d_OPT=d;
d_OPT.NC_MCI_DEM=zeros([length(d_OPT.GENDER),1]);d_OPT.NC_MCI_DEM(d_OPT.MCI==1)=1; d_OPT.NC_MCI_DEM(d_OPT.DEM==1)=2; 
d_OPT.SEX=d_OPT.GENDER==2; d.MOCATOTALRAWSCORE(d.MOCATOTALRAWSCORE==95)=NaN;
grpstats(d_OPT, "NC_MCI_DEM", {'mean','std'}, "DataVars",{'AGE', 'ED', 'MOCATOTALRAWSCORE'})
grpstats(d_OPT, "NC_MCI_DEM",  'sum', "DataVars",{'SEX'})

grpstats(d_OPT, "NC_MCI_DEM", {'mean','std'}, "DataVars",{'ED'})

mean(d_OPT.AGE)
std(d_OPT.AGE)
sum(d_OPT.GENDER==2)/length(d_OPT.GENDER)


clinical_OPT.NC_MCI_DEM=d_OPT.NC_MCI_DEM;
grpstats(clinical_OPT, "NC_MCI_DEM", {'mean','std'}, "DataVars",{'madrs_tot_scr'})
clinical_OPT.NC_MCI_DEM=d_OPT.NC_MCI_DEM;
grpstats(clinical_OPT, "NC_MCI_DEM", {'mean','std'}, "DataVars",{'athf_total_score_v2'})

cirsg.NC_MCI_DEM=d_OPT.NC_MCI_DEM;
grpstats(cirsg, "NC_MCI_DEM", {'mean','std'}, "DataVars",{'cirsgtotal'})


d_OPT.HL(d_OPT.HL>3)=NaN;
[tbl, chisq, p]=crosstab(d_OPT.RACE,d_OPT.NC_MCI_DEM)
[tbl, chisq, p]=crosstab(d_OPT.HL,d_OPT.NC_MCI_DEM)


mean(d.AGE)
std(d.AGE)
sum(d.GENDER==2)/length(d.GENDER)
anova1(aseg.EstimatedTotalIntraCranialVol, d_OPT.GENDER)
histogram(d.RACE)
sum(d.HL==0)

d_OPT.ED(d_OPT.ED>40)=NaN;
nanmean(d.ED)
nanstd(d.ED)
nanmean(MADRS_NP)
nanstd(MADRS_NP)

nanmean(athf.athf_total_score_v2)
nanstd(athf.athf_total_score_v2)

nanmean(cirsg.cirsgtotal)
nanstd(cirsg.cirsgtotal)

nanmean(d.MOCATOTALRAWSCORE)
nanstd(d.MOCATOTALRAWSCORE)

%% visual for timeline
figure;
for i=1:length(subs.ID);try
    sub=subs.ID{i};
    data=madrs(strcmp(madrs.ID, sub),:);
    data.DaysDiff=data.MADRSDateDays-d_OPT.NPDateDaysKaylaUPDATE(strcmp(d_OPT.ID, sub));
    ix=abs(data.DaysDiff)==min(abs(data.DaysDiff));
    MADRS_NP(i)=data.madrs_tot_scr(ix);MADRS_DateDiffNP(i)=data.DaysDiff(ix);
    startstudydate=min(min([data.start_step1days, data.start_step2days]));
    MADRS_length_studyNP(i)=data.MADRSDateDays(ix==1)-startstudydate;
    toplot(i,:)=[MADRS_length_studyNP(i), MADRS_length_studyNP(i)+data.DaysDiff(ix==1), MADRS_DateDiffNP(i)];
    %if abs(MADRS_DateDiffNP(i))<=60
    %plot(toplot(i,:),[i;i],'k', 'LineWidth',3); hold on;
    %end
    catch
    MADRS_NP(i)=NaN;MADRS_BL(i)=NaN;MADRS_DateDiff(i)=NaN;MADRS_length_studyNP(i)=NaN;
end; end
toplot(:,4)=MADRS_groups;

toplot=sortrows(toplot,1);
for i=1:length(subs.ID);
if abs(toplot(i,3))<=60
    if toplot(i,4)==0
    plot(toplot(i,1:2),[i;i],'b', 'LineWidth',3); hold on;
    elseif toplot(i,4)==1
    plot(toplot(i,1:2),[i;i],'k', 'LineWidth',3); hold on;
    elseif toplot(i,4)==2
    plot(toplot(i,1:2),[i;i],'r', 'LineWidth',3); hold on;
end; end
end
hold on; xline(42);xlim([-100 500]);ylim([0 270]);%xline(0);
hold on; scatter(toplot(:,1), [1:256], 'x', 'r')
%%
load('step2_RSFMRI_holdout100.mat');figure(2);hold off;plot(X,Y,'Color',[160/255 160/255 0]);AUC_all(1)=AUC
AUC_test_all(3,:)=AUC_test;
load('step2_COGMADRS_noRSFMRI_holdout100.mat');figure(2);hold on;plot(X,Y,'Color',[204/255 0 204/255]); AUC_all(2)=AUC
AUC_test_all(2,:)=AUC_test;
load('step2_MADRS_noRSFMRI_holdout100.mat');figure(2);hold on;plot(X,Y,'Color',[255/255 0 127/255]); set(gca,'box','off');AUC_all(3)=AUC
AUC_test_all(1,:)=AUC_test; %figure; b = bar(AUC_all,'k');
figure; yyaxis right; set(gca, 'color', 'none'); b = bar(mean(AUC_test_all'));
SEM = std(AUC_test_all')/sqrt(100);
hold on; [ngroups,nbars] = size(mean(AUC_test_all'));
tmp = nan(nbars, ngroups); for i = 1:nbars    tmp(i) = b.XEndPoints(i); end
errorbar(tmp',mean(AUC_test_all'), 2*SEM,'k','linestyle','none');ylim([0.5 0.8]);hold on
b = bar([1; 2; 3],diag(mean(AUC_test_all')),'stacked');
b(1).FaceColor=[255/255 0 127/255];b(2).FaceColor=[204/255 0 204/255]; b(3).FaceColor=[160/255 160/255 0];
