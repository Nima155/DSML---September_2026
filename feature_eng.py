
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class MyTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.aps_mean_dz_group = 0
        self.sps_mean_dz_group = 0
        self.feature_names = None 
    def fit(self, X: pd.DataFrame, y=None):
        self.aps_mean_dz_group = X.groupby("dzgroup", observed=False, dropna=False)["aps"].mean()
        self.sps_mean_dz_group = X.groupby("dzgroup", observed=False, dropna=False)["sps"].mean()
        self.feature_names = X.columns.values.tolist()
        return self
        
    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, "feature_names")
        X = X.copy()
        X["aps_vs_mean_aps"] = X["aps"] - X["dzgroup"].map(self.aps_mean_dz_group) # type: ignore
        X["sps_vs_mean_sps"] = X["sps"] - X["dzgroup"].map(self.sps_mean_dz_group) # type: ignore

        X["dzgroup"] = X["dzgroup"].astype("category")
        return X

    def get_feature_names_out(self):
        check_is_fitted(self, "feature_names")
        return self.feature_names + ["aps_vs_mean_aps", "sps_vs_mean_sps"] # type: ignore

def income_handler(df: pd.DataFrame):
   df = df.copy(deep=True)
   salary_map = {
      "under $11k": 0,
      "$11-$25k": 1,
      "$25-$50k": 2,
      ">$50k": 3
   }
   df["income"] = df["income"].map(salary_map).fillna(999)
   
   return df




def feat_new_eng(df: pd.DataFrame):
   df = df.copy(deep=True)
   
   df["dnr"] = df["dnr"].fillna("missing")
   df["race"] = df["race"].fillna("missing")
   df["sex"] = df["sex"].astype("category")
   df["medical_center"] = df["medical_center"].astype("category")
   # df["dzgroup"] = df["dzgroup"].astype("category")
   df["dzclass"] = df["dzclass"].astype("category")
   df["race"] = df["race"].astype("category")
   df["dnr"] = df["dnr"].astype("category")


   df["age"] = df["age"].round() # new 
   df["edu"] = df["edu"].round()

   ca_mappings = {
      "no": 1,
      "yes": 2,
      "metastatic": 3
   }

   df["ca"] = df["ca"].map(ca_mappings)
   df = df.drop(columns=["temp_f", "Id", "adls"])


   # https://archive.ics.uci.edu/dataset/880/support2 -- I pulled the imputations from here! 
   df.loc[df["alb"].isna(), "alb"] = 3.5
   df.loc[df["pafi"].isna(), "pafi"] = 333.3
   df.loc[df["bili"].isna(), "bili"] = 1.01
   df.loc[df["crea"].isna(), "crea"] = 1.01
   df.loc[df["bun"].isna(), "bun"] = 6.51
   df.loc[df["wblc"].isna(), "wblc"] = 9
   df.loc[df["urine"].isna(), "urine"] = 2502

   month_map = {
   'Sep': 9, 'Feb': 2, 'Jul': 7, 'Oct': 10, 'Jan': 1, 'Jun': 6, 'May': 5, 'Aug': 8, 'Nov': 11, 'Mar': 3,
      'Apr': 4, 'Dec': 12
   }


   df["admission_month"] = df["admission_month"].map(month_map)
   df["month_sin"] = np.sin(2 * np.pi * df["admission_month"] / 12)
   df["month_cos"] = np.cos(2 * np.pi * df["admission_month"] / 12)
   df = df.drop(columns=["admission_month"])

   df["mod_shock_index"] = df["hrt"] / (df["meanbp"] + 0.001)
   df["bun_crea"] = df["bun"] / (df["crea"] + 0.001)
   
   df["prognostic_delta6m"] = df["prg6m"] - df["surv6m"]  
   df["prognostic_delta2m"] = df["prg2m"] - df["surv2m"]  
   df["adl_delta"] = df["adlsc"] - df["adlp"]  


   df["sirs_score"] = ((df["temp"] > 38.0) | (df["temp"] < 36.0)).astype(np.int8) + (df["hrt"] > 90).astype(np.int8) + (df["resp"] > 20).astype(np.int8) + \
   ((df["wblc"] > 12.0) | (df["wblc"] < 4.0)).astype(np.int8)

   # renal_failure_index
   df["rfi"] = df["urine"] / (df["crea"] + 0.001)
   

   return df



transformer_w_cat = make_column_transformer(
                        (OrdinalEncoder(categories=[["under $11k", "$11-$25k", "$25-$50k", ">$50k"]], 
                        handle_unknown="use_encoded_value", encoded_missing_value=-1, unknown_value=-1), ["income"]),
                        (OneHotEncoder(handle_unknown="ignore"), ["sex", "medical_center", "dzgroup", "ca", "dzclass", "race", "dnr"]), 
                        remainder="passthrough")




pipe_non_tree_new_eng = make_pipeline(FunctionTransformer(feat_new_eng), MyTransformer(), transformer_w_cat,  SimpleImputer(), StandardScaler()) # for non tree based models


pipe_random_tree_new_eng = make_pipeline(FunctionTransformer(feat_new_eng), MyTransformer(), transformer_w_cat) # for more simple tree-based models

pipe_boosted_tree_new_neg = make_pipeline(FunctionTransformer(feat_new_eng), MyTransformer(), FunctionTransformer(income_handler)) # for tree based models that can handle missing values as well as categories on their own
 