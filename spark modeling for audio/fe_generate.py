#********************************************************************
#Copyright (C) 2019-. All rights reserved.

#Author       :   chenxinye
#Environment  :   Python3, PySpark2.4.3
#Date         :   2019-11-06

#Contact      : cclcquant@yahoo.com && https://github.com/chenxinye, if you have any suggestion, just be free to contact me. Thank you!
#*******************************************************************



import numpy as np
import pandas as pd


def drop_stdzero(dfp, dfn):
    dfp_s = dfp.sample(False, 0.1, 2019).toPandas()
    dfn_s = dfn.sample(False, 0.1, 2019).toPandas()
    
    train_std = pd.concat([dfp_s,dfn_s],axis = 0).std()
    stdzero_col = train_std[train_std==0].index.tolist()
    for i in stdzero_col:
        dfp = dfp.drop(i)
        dfn = dfn.drop(i)
    return dfp, dfn, stdzero_col


def check_null(df):
    if len(df.columns) == 0 or len(df.dropna()) == 0:
        print("none!")
        return False
    else:
        return True


def list_minus(list1,list2):
    return [i for i in list1 if i not in list2]
	

def fe_generate(data,ret = True):
    data = data.toPandas()
    col_all = data.columns

    ##1.features of buying behavior
    # pchs_firstcate_columns
    pchs_firstcate_columns = [col for col in data.columns if 'pchs' in col and 'secondcate' not in col and 'attr' not in col]
    #print(pchs_firstcate_columns)

    # pchs_secondcate_columns
    pchs_secondcate_columns = [col for col in data.columns if 'pchs' in col and 'secondcate' in col]
    #print(pchs_secondcate_columns)

    # pchs_columns
    pchs_columns = [col for col in data.columns if 'pchs' in col]
    #print(pchs_columns)

    ##2.features of following behaviour
    flw_secondcate_columns = [col for col in data.columns if 'flw' in col]
    #print(flw_secondcate_columns)

    ##3.features of browser behaviour
    brs_secondcate_columns = [col for col in data.columns if 'brs' in col and 'secondcate' in col]
    #print(brs_secondcate_columns)

    brs_firstcate_columns = [col for col in data.columns if 'brs' in col and 'secondcate' not in col]
    #print(brs_firstcate_columns)

    brs_columns = [col for col in data.columns if 'brs' in col]
    #print(brs_columns)

    ##4.features of ad click
    clk_secondcate_columns = [col for col in data.columns if 'clk' in col and 'secondcate' in col]
    #print(clk_secondcate_columns)

    clk_firstcate_columns = [col for col in data.columns if 'clk' in col and 'secondcate' not in col]
    #print(clk_firstcate_columns)

    clk_columns = [col for col in data.columns if 'clk' in col]
    #print(clk_columns)

    ##5.features of user attribute 
    attr_columns = [col for col in data.columns if 'attr' in col]
    #print(attr_columns)

    ##6.features of add repeat buying behavior
    cart_secondcate_columns = [col for col in data.columns if 'cart' in col and 'secondcate' in col]
    #print(cart_secondcate_columns)

    cart_firstcate_columns = [col for col in data.columns if 'cart' in col and 'secondcate' not in col]
    #print(cart_firstcate_columns)

    cart_columns = [col for col in data.columns if 'cart' in col]
    #print(cart_columns)

    columns_dict = {
        'pchs_firstcate_columns':pchs_firstcate_columns,
        'pchs_secondcate_columns':pchs_secondcate_columns,
        'pchs_columns':pchs_columns,
        'flw_secondcate_columns':flw_secondcate_columns,
        'brs_secondcate_columns':brs_secondcate_columns,
        'brs_firstcate_columns':brs_firstcate_columns,
        'brs_columns':brs_columns,
        'clk_firstcate_columns':clk_firstcate_columns,
        'clk_secondcate_columns':clk_secondcate_columns,
        'clk_columns':clk_columns,
        'attr_columns':attr_columns,
        'cart_secondcate_columns':cart_secondcate_columns,
        'cart_firstcate_columns':cart_firstcate_columns,
        'cart_columns':cart_columns
    }


    attrlist = list()
    for i in columns_dict:
        attrlist += columns_dict[i]

    col_reduce = list_minus(col_all,attrlist)

    ###########################transformation##########################
    attr_product = data[attr_columns]

    pchs_firstcate_product = data[pchs_firstcate_columns]
    pchs_secondcate_product = data[pchs_secondcate_columns]
    pchs_product = data[pchs_columns]

    brs_firstcate_product = data[brs_firstcate_columns]
    brs_secondcate_product = data[brs_secondcate_columns]
    brs_product = data[brs_columns]

    clk_firstcate_product = data[clk_firstcate_columns]
    clk_secondcate_product = data[clk_secondcate_columns]
    clk_product = data[clk_columns]

    cart_firstcate_product = data[cart_firstcate_columns]
    cart_secondcate_product = data[cart_secondcate_columns]
    cart_product = data[cart_columns]

    user_attr_all_sum = pd.DataFrame({'attr_all_sum':attr_product.sum(axis = 1)})
    user_pchs = pd.DataFrame({'pchs_firstcate_sum':pchs_firstcate_product.sum(axis=1),
                              'pchs_secondcate_sum':pchs_secondcate_product.sum(axis=1)
                             })

    user_pchs_all_sum = pd.DataFrame({'pchs_all_sum':pchs_product.sum(axis=1)})
    pchs_firstcate_sum = pd.DataFrame({'pchs_firstcate_sum':pchs_firstcate_product.sum(axis=1)})
    pchs_secondcate_sum = pd.DataFrame({'pchs_secondcate_sum':pchs_secondcate_product.sum(axis=1)})


    user_brs = pd.DataFrame({'brs_firstcate_sum':brs_firstcate_product.sum(axis=1),
                             'brs_secondcate_sum':brs_secondcate_product.sum(axis=1)
                            })

    user_brs_all_sum = pd.DataFrame({'brs_all_sum':brs_product.sum(axis = 1)})
    brs_firstcate_sum = pd.DataFrame({'brs_firstcate_sum':brs_firstcate_product.sum(axis=1)})
    brs_secondcate_sum = pd.DataFrame({'brs_secondcate_sum':brs_secondcate_product.sum(axis=1)})  


    user_clk = pd.DataFrame({'clk_firstcate_sum':clk_firstcate_product.sum(axis=1),
                             'clk_secondcate_sum':clk_secondcate_product.sum(axis=1)
                            })

    user_clk_all_sum = pd.DataFrame({'clk_all_sum':clk_product.sum(axis=1)})
    clk_firstcate_sum = pd.DataFrame({'clk_firstcate_sum':clk_firstcate_product.sum(axis=1)})
    clk_secondcate_sum = pd.DataFrame({'clk_secondcate_sum':clk_secondcate_product.sum(axis=1)})


    user_cart = pd.DataFrame({'cart_firstcate_sum':cart_firstcate_product.sum(axis=1),
                              'cart_secondcate_sum':cart_secondcate_product.sum(axis=1)
                             })

    user_cart_all_sum = pd.DataFrame({'cart_all_sum':cart_product.sum(axis=1)})
    cart_firstcate_sum = pd.DataFrame({'cart_firstcate_sum':cart_firstcate_product.sum(axis=1)})
    cart_secondcate_sum = pd.DataFrame({'cart_secondcate_sum':cart_secondcate_product.sum(axis=1)})


    user_rate = pd.DataFrame({
        'attr_sum':user_attr_all_sum.attr_all_sum,
        'pchs_sum':user_pchs_all_sum.pchs_all_sum,
        'clk_sum':user_clk_all_sum.clk_all_sum,
        'brs_sum':user_brs_all_sum.brs_all_sum,
        'cart_sum':user_cart_all_sum.cart_all_sum,
        'clk_rate':user_pchs_all_sum.pchs_all_sum/user_clk_all_sum.clk_all_sum,
        'clk_firstcate_rate':user_pchs.pchs_firstcate_sum/user_clk.clk_firstcate_sum,
        'clk_secondcate_rate':user_pchs.pchs_secondcate_sum/user_clk.clk_secondcate_sum,
        'brs_rate':user_pchs_all_sum.pchs_all_sum/user_brs_all_sum.brs_all_sum,
        'brs_firstcate_rate':user_pchs.pchs_firstcate_sum/user_brs.brs_firstcate_sum,
        'brs_secondcate_rate':user_pchs.pchs_secondcate_sum/user_brs.brs_secondcate_sum,
        'cart_rate':user_pchs_all_sum.pchs_all_sum/user_cart_all_sum.cart_all_sum,
        'cart_firstcate_rate':user_pchs.pchs_firstcate_sum/user_cart.cart_firstcate_sum,
        'cart_secondcate_rate':user_pchs.pchs_secondcate_sum/user_cart.cart_secondcate_sum,
        'attr_rate':user_pchs_all_sum.pchs_all_sum/user_attr_all_sum.attr_all_sum
    })

    if check_null(user_rate.dropna(how = 'all',axis = 1)):
        print("data has been added into user_rate!")
        user_rate.dropna(how = 'all', axis = 1, inplace = True)
        user_rate[np.isinf(user_rate)] = -1
        data = pd.concat([data, user_rate], axis = 1).fillna(0)
    else:
        print("user_rate is not installded")
    
    if ret:
        return data, columns_dict, col_reduce, user_rate
    else:
        return data


dfp, dfn, dropcol = drop_stdzero(dfp,dfn)

dfp = fe_generate(dfp, False)
dfn = fe_generate(dfn, False)
dftest = fe_generate(dftest, False)
