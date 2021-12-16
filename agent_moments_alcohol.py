import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from income_gmm import FitNielsenIncome
# Call this first to read and arrange the data

#get proof from upc_descr
def getProof(descr):
    pattern = re.compile("\d\d*\\.*\d*P")    
    if descr is None:
        return 0    
    matches = pattern.findall(descr.replace("PK", "").replace("PT", ""))
    if len(matches) > 1:
        #print("Multiple matches found: " + descr)
        #print(matches)
        return matches[0][:-1]
    elif (len(matches) < 1):
        #print("Zero matches found: " + descr)
        return 0
    else:
        return matches[0][:-1]

def prep_prod_data(fn_prod, fn_purchases):
    # df_prod = pd.read_parquet(fn_prod)[['upc', 'upc_ver_uc', 'upc_descr','product_module_code','product_module_descr',
    #                                   'size1_code_uc','size1_amount','size1_units','multi','upc_descr']]
    df_prod = pd.read_parquet(fn_prod)
    df_prod['proof'] = df_prod['upc_descr'].apply(getProof)
    #manually adjust proof column of a few edge cases
    df_prod.loc[(df_prod.upc_descr=='RG13P RUM PRALINE 80P'),'proof']= 80
    df_prod.loc[(df_prod.upc_descr=='FFLY MSN WHISKEY BKBRY 71.4 P'),'proof']= 71.4
    df_prod.loc[(df_prod.upc_descr=='WLT STRT RYE WHISKEY 117.4 P'),'proof']= 117.4
    #drop (5/1022) products which do not have proof on upc_descr
    df_prod = df_prod.dropna(subset=['proof'])
    df_prod['proof'] = df_prod['proof'].astype(float)
    df_prod = df_prod[df_prod['proof'] > 0]
    #convert size unit
    df_prod['size1_adjusted'] = df_prod['size1_amount'].where((df_prod['size1_units']=='LI'),other=df_prod['size1_amount']/1000)
    df_purchases = pd.read_parquet(fn_purchases)[['trip_code_uc', 'upc', 'upc_ver_uc', 'quantity','panel_year', 'purchase_date',
                                      'household_code','total_price_paid','coupon_value']]
    df_prod_merged = df_purchases.merge(df_prod, on=['upc','upc_ver_uc'],how='left')
    df_prod_merged['liter_total'] = df_prod_merged['quantity'] * df_prod_merged['size1_adjusted']* df_prod_merged['multi']
    #calculate price per liter
    df_prod_merged['price_per_liter'] = (df_prod_merged['total_price_paid'] - df_prod_merged['coupon_value']) / df_prod_merged['liter_total']
    df_prod_merged = df_prod_merged.dropna()
    return df_prod_merged


def hh_alcohol_consumption(fn_hh, fn_prod, fn_purchases,fn_bins):
    # household data
    hh_data = pd.read_parquet(fn_hh)[['household_code', 'panel_year','projection_factor',
                                       'household_income']]
    # Merge the income bins in and compute median HH income
    df_bins = pd.read_excel(fn_bins)
    df_hh2 = pd.merge(hh_data, df_bins, on='household_income')

    # After 2010 we go from 19-->16 buckets (top coding at 100k instead of 200k)
    df_hh2.loc[df_hh2.panel_year < 2010,
            'income'] = df_hh2.loc[df_hh2.panel_year < 2010, 'med_inc_early']/100000
    df_hh2.loc[df_hh2.panel_year >= 2010,
            'income'] = df_hh2.loc[df_hh2.panel_year >= 2010, 'med_inc_late']/100000
    df_hh3 = df_hh2[['household_code', 'panel_year',
                    'projection_factor', 'income']]
                                   
    
    # process prod data
    df_prod = prep_prod_data(fn_prod, fn_purchases)
    df_prod['quarter'] = pd.to_datetime(df_prod['purchase_date'])+ pd.offsets.QuarterEnd(0)

    #aggregate alcohol liters purchased per household per quarter
    hh_alcohol_purchased = df_prod.groupby(['household_code','panel_year','quarter'], as_index=False)['liter_total'].sum()
    x = df_hh3.merge(hh_alcohol_purchased, on=['household_code','panel_year'], how = 'left')
    # #create indicator for purchase_event
    x = x.dropna()
    #create indicator for purchase_event
    x['ind'] = 1
    quarter_date = pd.DataFrame(x['quarter'].unique(),columns=['quarter'])
    quarter_date['panel_year'] = pd.to_datetime(quarter_date['quarter']).dt.year

    all_table = df_hh3.merge(quarter_date, on=['panel_year'],how = 'left')
    rslt = all_table.merge(x, on=['household_code','panel_year','income','projection_factor','quarter'],how = 'left')
    rslt['liter_total'].fillna(0, inplace=True)
    rslt['ind'].fillna(0, inplace=True)
    #histgram of alcohol consumption
    # x['liter_total'].fillna(0, inplace=True)
    # x = x[x['panel_year']==2011]
    # x = x[x['liter_total'] > 0]
    # x['liter_total'] = np.log(x['liter_total'])
    # t = x['liter_total'].value_counts().rename_axis('liter_total').reset_index(name='counts')
    # t["liter_total"].plot(kind="hist",bins = 40,weights=t["counts"])
    # plt.show()
    return rslt



def prep_agent_data(fn_hh, fn_prod, fn_purchases, use_projection=True, use_liter_purchased=False):
    # household data
    hh_data = pd.read_parquet(fn_hh)[['household_code', 'panel_year','projection_factor',
                                       'household_income']]
    # Code everything but 9 as having kids
    #hh_data['kids'] = (hh_data['age_and_presence_of_children'] != 9)

    # process prod data
    df_prod = prep_prod_data(fn_prod, fn_purchases)
    #aggregate alcohol liters purchased per household per panel_year
    hh_alcohol_purchased = df_prod.groupby(['household_code','panel_year'], as_index=False)['liter_total'].sum()
    x = hh_data.merge(hh_alcohol_purchased, on=['household_code','panel_year'], how = 'left')
    x['liter_total'].fillna(0, inplace=True)
    if not use_projection:
        x['projection_factor'] = 1
    if use_liter_purchased:
        x['projection_factor'] = x['projection_factor'] * x['liter_total']
    return x.groupby(['panel_year', 'household_income'])[['projection_factor']].sum()

# Estimate lognormal distribution for each row of the dataframe
# Split the sample:
# 1. Before 2010 (19 bins)
# 2. After 2010 (16 bins)
def estimate_all_lognormals(df):
    x3 = df.pivot_table(index=['panel_year'], columns=[
                        'household_income'], values='projection_factor').fillna(0)
    splitA = x3[x3.index.get_level_values(0) >= 2010]
    splitB = x3[x3.index.get_level_values(0) < 2010]
    lognormals = pd.concat(
        [estimate_params(splitB, alt=False), estimate_params(splitA, alt=True)], axis=0)
    #print(lognormals)
    return lognormals

# Split the sample again
# 1. HH with kids
# 2. HH without kids


def estimate_params(df, alt):
    estimates = np.apply_along_axis(
        fit_income, axis=1, arr=df.values, alt=alt)
    dat = pd.DataFrame(estimates, columns=['mu', 'sig'], index=df.index)
    #print(dat)
    return dat

# Helper to fit income and extract params
# alt =16 bins
# not alt = 19 bins


def fit_income(a, alt):
    if alt:
        a = a[0:16]
    inc = FitNielsenIncome(a[None, :], alternate=alt)
    res = inc.solve()
    return res.x
