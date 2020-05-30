import pandas as pd
import numpy as np
import glob
import os

# Input path to the Industry Occupation Matrix
OccIndMatrix = 'Complete_Industry_Occupation.csv'
# Input path to the MSA Occupation Information
NOLA_EmpInfo= 'OccupationInfo_NOLA.csv'
#Read ZipCode level Employment data from LOEDS
empDataPath = 'zip_sector.csv'
#Risk Data path
riskPath = 'Physical_Proximity.csv'



mdf = pd.read_csv(OccIndMatrix)
occdf = pd.read_csv(NOLA_EmpInfo)
sec = pd.read_csv(empDataPath)
pr = pd.read_csv(riskPath)
id= ['11', '21', '22', '23', '31-33', '42', '44-45', '48-49', '51', '52', '53', '54', '55', '56', '61', '62', '71', '72', '81', '90']


#Filter the dataframe
occdf = occdf[['occ_code','occ_title','tot_emp']]
occdf = occdf[occdf['tot_emp']!='**']
occdf['tot_emp'] = occdf['tot_emp'].str.replace(',','').astype(int)

# Map the employees to the percentages in the Industry Occupation Matrix
mdf2 = pd.merge(mdf, occdf, how='inner', left_on='Occupation Code', right_on='occ_code')
mdf2.drop('occ_code', axis=1, inplace=True)
for i in id:
    mdf2[i] = mdf2[i]*mdf2['tot_emp']/100
    mdf2[i]= mdf2[i].astype(int)

for i in id:
    mdf2[i]=mdf2[i]/mdf2[i].sum()
mdf2.drop(['occ_title','tot_emp'], axis=1, inplace=True)
mdf2.set_index('Occupation Code', inplace=True)


# Matrix Multiply the Industry occupation matrix to the number of employees in each sector at each zip code to get the occupation at each zip code

sec.set_index('ZIP', inplace=True)
mdf2T = mdf2.T
mdf4 = sec.values.dot(mdf2T.values)


# Assign row and columns to the matrix

mdf5 = pd.DataFrame(mdf4, index=sec.index, columns=mdf2T.columns)
mdf5 = mdf5.astype(int)
mdf5 = mdf5.T
mdf5 = mdf5.reset_index()


# Filter the rows from the risk index and rescale the index, then combine the risk score with the occupation data from previous step

pr['Code'] = pr['Code'].str[:7]
pr.drop('Occupation',axis=1, inplace=True)
mdf6 = pd.merge(mdf5, pr, how='left', left_on='Occupation Code', right_on='Code')
mdf6.fillna(0, inplace=True)
mdf6.drop('Code', axis=1, inplace=True)
cols = mdf6.columns.values
cols = cols[cols!='Occupation Code']
cols = cols[cols!='Context']
for col in cols:
    mdf6[col] = mdf6[col]*mdf6['Context']/100

#%% Melt the data for each occupation and each zipcode


mdf6_melt = pd.melt(mdf6, id_vars=['Occupation Code'], var_name='ZIP', value_name='Occupation Risk')
mdf6_melt['Occupation Risk'] = mdf6_melt['Occupation Risk'].apply(lambda x: 0.01 if x==0 else np.log10(x))
x,y = mdf6_melt['Occupation Risk'].min(), mdf6_melt['Occupation Risk'].max()
mdf6_melt['Occupation Risk'] = (mdf6_melt['Occupation Risk']-x)/(y-x)
mdf6_melt.to_csv('OccupationRisk_Melted.csv', index=False)

