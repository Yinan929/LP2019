import pandas as pd
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

import agent_moments_alcohol
from agent_moments_alcohol import estimate_all_lognormals, prep_agent_data, prep_prod_data,hh_alcohol_consumption



# Fit an income distribution for each year for CT using entire Nielsen Panel
fn_hh = 'nielsen_panelist.parquet'
fn_purchases = 'nielsen_purchases.parquet'
fn_prod = 'nielsen_prodinfo.parquet'
fn_bins =  'nielsen_income_bins.xlsx'


# Block 1: Fit lognormal to agent data  

# Can use projection weights or equal weight 
# Can use alcohol purchases in Liters as weight or not

#use projection weights
df = prep_agent_data(fn_hh, fn_prod, fn_purchases, use_projection=True,use_liter_purchased=False)
df_lognormal = estimate_all_lognormals(df)
df_lognormal.to_csv('projection_weight.csv')

#use equal weights
df_eq = prep_agent_data(fn_hh, fn_prod, fn_purchases, use_projection=False,use_liter_purchased=False)
df_lognormal_eq = estimate_all_lognormals(df_eq)
df_lognormal_eq.to_csv('equal_weight.csv')

#use alcohol purchases in Liters x projection weights
df_add_liter = prep_agent_data(fn_hh, fn_prod, fn_purchases, use_projection=True,use_liter_purchased=True)
df_add_liter_lognormal = estimate_all_lognormals(df_add_liter)
df_add_liter_lognormal.to_csv('liter_weight.csv')

###
# Block 2: Micromoments
####
def covar_mat(df, name):
    X = df[['proof','price_per_liter', 'size1_adjusted','income']].values
    y = np.cov(X, rowvar=False, aweights=df.liter_total)
    z = y[:, -1:]
    return {'index': name, 'mm_cov_inc_proof': z[0, 0], 'mm_cov_inc_price': z[1, 0], 'mm_cov_inc_size': z[2, 0]}

def exp_outside(df,group_var):
  x=df[df.ind==0].groupby(group_var)[['income']].mean()
  z=df[df.ind==0][['income']].mean()
  tot=pd.DataFrame({'income':z['income']},index=['all'])
  return tot.append(x)

# # Read in the datasets
df_hh = pd.read_parquet(fn_hh, columns=['household_code', 'panel_year',
                                        'projection_factor', 'household_income'])
df_prod = prep_prod_data(fn_prod, fn_purchases)

# Merge the income bins in and compute median HH income
df_bins = pd.read_excel(fn_bins)
df_hh2 = pd.merge(df_hh, df_bins, on='household_income')

# After 2010 we go from 19-->16 buckets (top coding at 100k instead of 200k)
df_hh2.loc[df_hh2.panel_year < 2010,
           'income'] = df_hh2.loc[df_hh2.panel_year < 2010, 'med_inc_early']/100000
df_hh2.loc[df_hh2.panel_year >= 2010,
           'income'] = df_hh2.loc[df_hh2.panel_year >= 2010, 'med_inc_late']/100000
df_hh3 = df_hh2[['household_code', 'panel_year',
                 'projection_factor', 'income']]

# Merge the household data against the purchase data
# we lose some observations here
# filter on liter_total > 0
df = pd.merge(df_prod,df_hh3, on=[
              'household_code', 'panel_year'])
df = df[df.liter_total != 0].copy()

# Do this by panel_year
# Add a row for overall aggregate
df_out_year = pd.DataFrame.from_dict([covar_mat(df, name='all')]+[covar_mat(
    group, name) for name, group in df.groupby(['panel_year'])]).set_index('index')

df_hh_is_purch = hh_alcohol_consumption(fn_hh,fn_prod, fn_purchases,fn_bins)

exp_inc_no_purchase_quarter = exp_outside(df_hh_is_purch,'quarter')
#exp_inc_no_purchase_quarter['income'] = exp_inc_no_purchase_quarter['income']*100000


###
# Block 3: share of all of the categories by income and groupby race/hispanic
####
hh_dat = pd.read_parquet(fn_hh)[['household_code','panel_year','household_income','race','hispanic_origin']]
#aggregate income bucket to low-mid-high three classes
hh_dat.loc[(hh_dat['household_income'] >=3)&(hh_dat['household_income'] <=19),'income'] = 0
hh_dat.loc[(hh_dat['household_income'] >=21)&(hh_dat['household_income'] <=26),'income'] = 1
hh_dat.loc[(hh_dat['household_income'] >26),'income'] = 2

panel_years = [2007,2008,2009,2010,2011,2012,2013]
income_backet = [0,1,2]

cate = ['BOURBON-BLENDED','BOURBON-STRAIGHT/BONDED','BRANDY/COGNAC','CANADIAN WHISKEY','GIN','IRISH WHISKEY','RUM','SCOTCH','TEQUILA','VODKA','REMAINING WHISKEY']
prod_dat = prep_prod_data(fn_prod, fn_purchases)
rslt = prod_dat.merge(hh_dat, on=['household_code','panel_year'],how='left')
rslt = rslt[rslt['income'] == 2]
rslt_year = rslt.groupby(['panel_year','product_module_descr'],as_index = False)[['liter_total']].sum()


share_by_category = pd.DataFrame(columns = ['panel_year','BOURBON-BLENDED','BOURBON-STRAIGHT/BONDED','BRANDY/COGNAC',
						'CANADIAN WHISKEY','GIN','IRISH WHISKEY','RUM','SCOTCH','TEQUILA','VODKA','REMAINING WHISKEY'])
share_by_category['panel_year'] = [2007,2008,2009,2010,2011,2012,2013]
share_by_category = share_by_category.set_index('panel_year')

for year in panel_years:
	smp = rslt_year[rslt_year['panel_year'] == year]
	cate_list = smp['product_module_descr'].unique()
	smp = smp.set_index('product_module_descr')
	for c in cate:		
		if c in cate_list:
			num = smp.loc[c,'liter_total']
			share_by_category.loc[year,c] = num

#print(share_by_category)
ax = share_by_category.plot.area()
plt.show()


purchase_dat = pd.read_parquet(fn_purchases)
purchase_hh = purchase_dat.merge(hh_dat, on=['household_code','panel_year'])
purchase_hh_per_race = pd.DataFrame(purchase_hh.groupby(['panel_year','race'])['household_code'].nunique())
purchase_hh_hispanic = pd.DataFrame(purchase_hh.groupby(['panel_year','hispanic_origin'])['household_code'].nunique())
print(purchase_hh_hispanic)
print(purchase_hh_per_race)













