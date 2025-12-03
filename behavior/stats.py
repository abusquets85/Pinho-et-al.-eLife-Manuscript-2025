
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
import scikit_posthocs as sp
from statsmodels.stats.multicomp import MultiComparison
from outliers import smirnov_grubbs as grubbs
import statsmodels.stats.multitest as smt

# pip install scikit-posthocs
# pip install outlier_utils

def is_normal(meas, p_th=0.05):
    # how to run : is_normal(df['perc_freezing'], p_th=0.05)
    r = stats.kstest(meas, 'norm')
    response = not r.pvalue < p_th
    return response


def equal_variances(df, parameter='perc_freezing', p_th=0.05):
    protocols = df['protocol'].unique()
    values = []
    for prot in protocols:
        c_values = df[df['protocol'] == prot][parameter]
        values.append(c_values)
    s, pval = stats.levene(*values)
    if pval < p_th:
        return False
    else:
        return True
    
def mixed_linear_model(df, parameter='perc_freezing', p_value=0.05):
    model = smf.mixedlm(f"{parameter} ~ C(protocol) * C(drug)", df, groups= df["mouse"])
    model_fit = model.fit()
    return print(model_fit.summary())

def GLM(df, parameter='perc_freezing', p_value=0.05): # attention the distribution to be used fit your data?
    model = smf.glm("perc_freezing ~ C(protocol)", df, family=sm.families.Binomial)
    return model

def anova_oneway(df, parameter='perc_freezing', p_value=0.05):
    model = smf.ols(f"{parameter} ~ C(protocol)", df)
    model_fit = model.fit()
    return print(model_fit.summary())

def kruskal_wallis(df, parameter='perc_freezing', p_value=0.05):
    df= df.replace(["habituation"], 1)
    df= df.replace(["probe test new"], 2)
    df= df.replace(["probe test old"], 3)
    return stats.kruskal(df[parameter], df['protocol'])

def anova_oneway_repeated(df, parameter='perc_freezing', p_value=0.05):
    return print(AnovaRM(df, 'perc_freezing', 'protocol', within=['mouse']).fit())

def friedman(df, parameter='perc_freezing', p_value=0.05):
    df= df.replace(["habituation"], 1)
    df= df.replace(["probe test new"], 2)
    df= df.replace(["probe test old"], 3)
    return stats.friedmanchisquare(df[parameter], df['protocol'], df['mouse'])
    
def anova_twoway_no_interaction(df, parameter='perc_freezing', p_value=0.05):
    model = smf.ols(f"{parameter} ~ C(protocol) + C(drug)", df)
    model_fit = model.fit()
    return print(model_fit.summary())

def anova_twoway_with_interaction(df, parameter='perc_freezing', p_value=0.05):
    model = smf.ols(f"{parameter} ~ C(protocol) * C(drug)", df)
    model_fit = model.fit()
    return print(model_fit.summary())


def tukey_hsd_parametric(df, variable1= 'perc_freezing', variable2='protocol', p_value=0.05):
    posthoc = MultiComparison(df[variable1], df[variable2])
    return print(posthoc.tukeyhsd())

def wilcoxon_non_parametric(df, variable1= 'perc_freezing', variable2='protocol', p_value=0.05):
    posthoc = sp.posthoc_wilcoxon(df, variable1, variable2, p_adjust = 'holm')
    return posthoc

def nemenyi_friedman_non_parametric_repeated(df, variable1= 'perc_freezing', variable2='protocol', repeated = 'mouse', p_value=0.05):
    posthoc = sp.posthoc_nemenyi_friedman(df, variable1, repeated, variable2, melted=True)
    return posthoc

def pairwise_corrected(df, variable1= 'perc_freezing', variable2= 'protocol', p_value=0.05):
    posthoc = sp.posthoc_ttest(df, variable1, variable2, p_adjust= 'holm')
    return posthoc 

def pairwise_non_corrected_bins(df, variable1= 'freezing_1', variable2= 'protocol', p_value=0.05):
    posthoc = sp.posthoc_ttest(df, variable1, variable2)
    return posthoc 

def pairwise_comparisons_non_corrected(df, variable1= 'perc_freezing', variable2= 'protocol', p_value=0.05):
    posthoc = sp.posthoc_ttest(df, variable1, variable2)
    return posthoc # dont forget to correct to the number of comparisons, useful to planned comparisons

def pairwise_comparisons_non_corrected_ON(df, variable1= 'meas_slices2', variable2= 'protocol', p_value=0.05):
    posthoc = sp.posthoc_ttest(df, variable1, variable2)
    return posthoc # dont forget to correct to the number of comparisons, useful to planned comparisons


def t_test(df, variable= 'ptt'):
    filtro= df[df['protocol'] == variable]
    a=filtro['meas_slices1']
    b=filtro['meas_slices2']
    pvalue= stats.ttest_rel(a, b)
    return pvalue


#def outliers(df, parameter= 'perc_freezing', p_value=0.05):
#    new_parameter = grubbs.test(df[parameter], alpha= p_value) 
#    if len(new_parameter) == len(parameter):
#        print ('No Outliers')
#    else:
#        print ('Attention Outliers')
#    return new_parameter

def Grubbs_test(df, parameter= 'perc_freezing', p_value=0.05):
    new_parameter = grubbs.test(df[parameter], alpha= p_value) #this variable is your variable without outliers
    index = grubbs.max_test_indices(df[parameter], alpha=p_value) #this variable is the index of the outliers
    values = grubbs.max_test_outliers(df[parameter], alpha=p_value) #this variable is the values of outliers in your variable
    return new_parameter, index, values

def planned_comparisons(list_pvalue):
    list_pvalue=[]
    pvalue_corrected = smt.multipletests(list_pvalue, alpha=0.05, method='fdr_tsbh', is_sorted=False, returnsorted=False)
    return pvalue_corrected

def correlation(df):
    x = df[df['protocol'] == 'ptt']
    y = df[df['protocol'] == 'ptl']
    Pearson = stats.pearsonr(x['perc_freezing'], y['perc_freezing'])    # Pearson's r
    Spearman = stats.spearmanr(x['perc_freezing'], y['perc_freezing'])   # Spearman's rho
    return Pearson, Spearman
    
# ANOVA
#model = smf.ols("perc_freezing ~ C(protocol)", df)  # Single condition
#model = smf.ols("perc_freezing ~ C(protocol) + C(drug)", df)  # Two conditions
#model = smf.ols("perc_freezing ~ C(protocol) * C(drug)", df)  # with interaction
#model_fit = model.fit()
#sm.stats.anova_lm(model_fit)
# with open('stats.txt', 'a') as fp:
#     fp.write(f'pvalue of the Mann Whitney between....: {pval}')
# Do not use as is # attention to the distribution to be used
# model = smf.glm("perc_freezing ~ C(protocol)", df, family=sm.families.Binomial)
