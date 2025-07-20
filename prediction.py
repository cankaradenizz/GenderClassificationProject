import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score)
import os
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv("IE425_Spring25_train_data.csv")
test_df = pd.read_csv("IE425_Spring25_test_data.csv")

print("Train shape:", train_df.shape)
print(train_df.columns)
train_df.info()
print("Test shape:", test_df.shape)
print(test_df.columns)
test_df['price_missing'] = test_df['sellingprice'].isna().astype(int)
test_df.info()
test_df.isnull().sum().sort_values(ascending=False)
test_df.head(50)
train_df.isnull().sum().sort_values(ascending=False)
cat_cols = [
    'product_gender', 'brand_name', 'brand_id', 'businessunit',
    'category_id',
    'Level1_Category_Id', 'Level1_Category_Name',
    'Level2_Category_Id', 'Level2_Category_Name',
    'Level3_Category_Id', 'Level3_Category_Name',
    'product_name'
]

for col in cat_cols:
    train_df[col] = train_df[col].fillna('Unknown')
train_df['price_missing'] = train_df['sellingprice'].isnull().astype(int)
price_median = train_df['sellingprice'].median()
train_df['sellingprice'] = train_df['sellingprice'].fillna(price_median)
train_df = train_df.dropna(subset=['contentid']).reset_index(drop=True)
train_df['time_stamp'] = pd.to_datetime(train_df['time_stamp'])
campaign_days = pd.to_datetime([
    "2020-11-09", "2020-11-10", "2020-11-11", "2020-11-12",
    "2020-11-27", "2020-11-28", "2020-11-29"
]).tz_localize("UTC")  

train_df['is_campaign_day'] = train_df['time_stamp'].dt.normalize().isin(campaign_days).astype(int)
test_df['time_stamp'] = pd.to_datetime(test_df['time_stamp'])
campaign_days = pd.to_datetime([
    "2020-11-09", "2020-11-10", "2020-11-11", "2020-11-12",
    "2020-11-27", "2020-11-28", "2020-11-29"
]).tz_localize("UTC")  

test_df['is_campaign_day'] = test_df['time_stamp'].dt.normalize().isin(campaign_days).astype(int)
train_df.isnull().sum().sort_values(ascending=False)
train_df.head(50)

for col in cat_cols:
    test_df[col] = test_df[col].fillna('Unknown')

test_df['price_missing'] = test_df['sellingprice'].isnull().astype(int)

test_df['sellingprice'] = test_df['sellingprice'].fillna(price_median)

test_df = test_df.dropna(subset=['contentid']).reset_index(drop=True)

test_df.isnull().sum().sort_values(ascending=False)

train_df['unique_id'].nunique()

train_df.head(50)

brand_check = train_df.groupby('brand_id')['brand_name'].nunique().sort_values(ascending=False)
print(brand_check.head(10))
print(brand_check.tail(10))

sorgulanacaklar = [
    'JwCD9VMsi6',
    'HVxYi1f09I',
    'WU9dr15iJS',
    'vH4IJknPUH',
    'qL9NXl3mqy'
]

for bid in sorgulanacaklar:
    print(f"\n🔍 brand_id: {bid}")
    alt_df = train_df[train_df['brand_id'] == bid]
    
    brand_names = alt_df['brand_name'].dropna().unique()
    
    if len(brand_names) == 0:
        print("⚠️ Hiçbir brand_name tanımlı değil (NaN).")
    else:
        print(f"📦 Brand Name listesi ({len(brand_names)} adet):")
        for bname in brand_names:
            print(f"   - {bname}")
    
    print(f"📊 Toplam kayıt sayısı: {len(alt_df)}")
    print("-" * 50)

for col in train_df.columns:
    print(train_df[col].value_counts())
print(train_df['time_stamp'].min())
print(train_df['time_stamp'].max())

train_df[train_df['is_campaign_day'] == 1].head(100)

import pandas as pd

def create_user_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df['hour'] = df['time_stamp'].dt.hour
    df['dayofweek'] = df['time_stamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].between(0, 6).astype(int)

    df['is_female_product'] = (df['product_gender'] == 'Kadın').astype(int)
    df['is_male_product'] = (df['product_gender'] == 'Erkek').astype(int)
    df['is_unisex_product'] = (df['product_gender'] == 'Unisex').astype(int)
    df_sorted = df.sort_values(by=['unique_id', 'contentid', 'time_stamp'])
    
    first_interaction = df_sorted[df_sorted['user_action'].isin(['visit', 'search'])] \
        .groupby(['unique_id', 'contentid'])['time_stamp'].first().rename('first_time')
    
    order_time = df_sorted[df_sorted['user_action'] == 'order'] \
        .groupby(['unique_id', 'contentid'])['time_stamp'].first().rename('order_time')
    
    decision_times = pd.concat([first_interaction, order_time], axis=1).dropna()
    
    decision_times['decision_time_sec'] = (decision_times['order_time'] - decision_times['first_time']).dt.total_seconds()
    
    avg_decision_time = decision_times.groupby('unique_id')['decision_time_sec'].mean().rename('avg_purchase_decision_time')
    action_counts = df.pivot_table(index='unique_id',
                                   columns='user_action',
                                   values='contentid',
                                   aggfunc='count',
                                   fill_value=0)
    action_counts.columns = [f"n_{col}" for col in action_counts.columns]

    agg_df = df.groupby('unique_id').agg(
        n_total_actions=('contentid', 'count'),
        n_unique_brand_id=('brand_id', 'nunique'),
        n_unique_businessunit=('businessunit', 'nunique'),
        n_unique_product_gender=('product_gender', 'nunique'),
        n_female_products=('is_female_product', 'sum'),
        n_male_products=('is_male_product', 'sum'),
        n_unisex_products=('is_unisex_product', 'sum'),
        mean_price=('sellingprice', 'mean'),
        max_price=('sellingprice', 'max'),
        min_price=('sellingprice', 'min'),
        std_price=('sellingprice', 'std'),
        n_weekend_actions=('is_weekend', 'sum'),
        n_night_actions=('is_night', 'sum'),
        n_price_missing=('price_missing', 'sum'),
        n_lvl1_cats=('Level1_Category_Name', 'nunique'),
        n_lvl2_cats=('Level2_Category_Name', 'nunique'),
        n_lvl3_cats=('Level3_Category_Name', 'nunique'),
        n_distinct_products=('contentid', 'nunique')
    )

    op_brands = df['brand_id'].value_counts().head(7).index
    for brand in op_brands:
        agg_df[f'brand_used_{brand}'] = df[df['brand_id'] == brand].groupby('unique_id')['brand_id'].count()
    agg_df[[f'brand_used_{b}' for b in op_brands]] = agg_df[[f'brand_used_{b}' for b in op_brands]].fillna(0)

    agg_df['basket_rate'] = action_counts.get('n_basket', 0) / action_counts.get('n_visit', 1)
    agg_df['order_rate'] = action_counts.get('n_order', 0) / (action_counts.get('n_basket', 1))
    agg_df['search_to_visit_rate'] = action_counts.get('n_search', 0) / (action_counts.get('n_visit', 1))
    agg_df['price_missing_ratio'] = agg_df['n_price_missing'] / agg_df['n_total_actions']
    agg_df['female_product_ratio'] = agg_df['n_female_products'] / agg_df['n_total_actions']
    agg_df['male_product_ratio'] = agg_df['n_male_products'] / agg_df['n_total_actions']

    first_action_hour = df.sort_values(by='time_stamp').groupby('unique_id')['hour'].first().rename('first_action_hour')
    first_action_night = (first_action_hour <= 6).astype(int).rename('first_action_night')

    df_sorted = df.sort_values(by=['unique_id', 'time_stamp'])
    df_sorted['time_diff'] = df_sorted.groupby('unique_id')['time_stamp'].diff().dt.total_seconds()
    avg_time_between_actions = df_sorted.groupby('unique_id')['time_diff'].mean().rename('avg_time_between_actions')

    action_diversity = (action_counts > 0).sum(axis=1).rename('action_diversity')

    most_common_category = (
        df.groupby(['unique_id', 'Level1_Category_Name'])
        .size()
        .reset_index(name='cat_count')
        .sort_values(['unique_id', 'cat_count'], ascending=[True, False])
        .drop_duplicates('unique_id')
        .set_index('unique_id')['Level1_Category_Name']
        .rename('most_common_category')
    )

    most_common_cat_ohe = pd.get_dummies(most_common_category).groupby('unique_id').sum().add_prefix('topcat_')

    extra_features = pd.concat([
        first_action_hour, first_action_night,
        avg_time_between_actions, action_diversity
    ], axis=1)

    agg_df = pd.concat([agg_df, extra_features, most_common_cat_ohe], axis=1)

    top15_lvl1 = df['Level1_Category_Name'].value_counts().head(15).index
    df['lvl1_top'] = df['Level1_Category_Name'].where(df['Level1_Category_Name'].isin(top15_lvl1), 'Other')
    lvl1_ohe = pd.get_dummies(df[['unique_id', 'lvl1_top']], columns=['lvl1_top']) \
                   .groupby('unique_id').sum().add_prefix('cat1_')

    top15_lvl2 = df['Level2_Category_Name'].value_counts().head(15).index
    df['lvl2_top'] = df['Level2_Category_Name'].where(df['Level2_Category_Name'].isin(top15_lvl2), 'Other')
    lvl2_ohe = pd.get_dummies(df[['unique_id', 'lvl2_top']], columns=['lvl2_top']) \
                   .groupby('unique_id').sum().add_prefix('cat2_')

    top15_lvl3 = df['Level3_Category_Name'].value_counts().head(15).index
    df['lvl3_top'] = df['Level3_Category_Name'].where(df['Level3_Category_Name'].isin(top15_lvl3), 'Other')
    lvl3_ohe = pd.get_dummies(df[['unique_id', 'lvl3_top']], columns=['lvl3_top']) \
                   .groupby('unique_id').sum().add_prefix('cat3_')

    top15_brand = df['brand_id'].value_counts().head(15).index
    df['brand_top'] = df['brand_id'].where(df['brand_id'].isin(top15_brand), 'Other')
    brand_ohe = pd.get_dummies(df[['unique_id', 'brand_top']], columns=['brand_top']) \
                    .groupby('unique_id').sum().add_prefix('brand_')

    top15_bunit = df['businessunit'].value_counts().head(15).index
    df['bunit_top'] = df['businessunit'].where(df['businessunit'].isin(top15_bunit), 'Other')
    bunit_ohe = pd.get_dummies(df[['unique_id', 'bunit_top']], columns=['bunit_top']) \
                    .groupby('unique_id').sum().add_prefix('bunit_')
    n_campaign_day_actions = df.groupby('unique_id')['is_campaign_day'].sum().rename('n_campaign_day_actions')
    
    total_actions = df.groupby('unique_id').size()
    
    campaign_action_ratio = (n_campaign_day_actions / total_actions).rename('campaign_action_ratio')
    
    agg_df = agg_df.join(n_campaign_day_actions).join(campaign_action_ratio)

    agg_df = agg_df.join(avg_decision_time)
    agg_df['avg_purchase_decision_time'] = agg_df['avg_purchase_decision_time'].fillna(0)
    
    features_df = pd.concat([
        agg_df, action_counts,
        lvl1_ohe, lvl2_ohe, lvl3_ohe,
        brand_ohe, bunit_ohe
    ], axis=1)

    return features_df.reset_index()

train_features = create_user_features(train_df)

train_features = train_features.merge(train_df[['unique_id', 'gender']].drop_duplicates(), on='unique_id', how='left')

train_features.head(50)
train_features = train_features.replace([np.inf, -np.inf], 0)
train_features = train_features.fillna(0)

test_features = create_user_features(test_df)
test_features = test_features.merge(test_df[['unique_id', 'gender']].drop_duplicates(), on='unique_id', how='left')
test_features.head(50)
test_features = test_features.replace([np.inf, -np.inf], 0)
test_features = test_features.fillna(0)

train_features.head(50)

train_features[train_features['n_campaign_day_actions'] > 0][['unique_id', 'gender', 'n_campaign_day_actions', 'campaign_action_ratio']].head(500)

low_importance_cols = {
    'cat3_lvl3_top_Jeans',
    'brand_brand_top_wd6Wp39Y6H',
    'brand_used_wd6Wp39Y6H',
    'brand_brand_top_fnIb2yyEyA',
    'cat3_lvl3_top_Pantolon',
    'brand_brand_top_CPlCHUarzb',
    'cat2_lvl2_top_Takı & Mücevher',
    'bunit_bunit_top_Takı',
    'topcat_Süpermarket',
    'brand_brand_top_dfgdVUL1E9',
    'brand_brand_top_mjwJPfvhRy',
    'brand_used_dfgdVUL1E9',
    'cat1_lvl1_top_Unknown',
    'topcat_Yaşam',
    'topcat_Ayakkabı',
    'topcat_Aksesuar',
    'topcat_Kozmetik & Kişisel Bakım',
    'topcat_Anne & Bebek & Çocuk',
    'topcat_Spor & Outdoor',
}

train_features = train_features.drop(columns=low_importance_cols, errors='ignore')
test_features = test_features.drop(columns=low_importance_cols, errors='ignore')

test_features.info()

train_features.info()

print("Test shape:", test_df.shape)

train_df['gender'].value_counts()

train_features.head(10)


X_train = train_features.drop(columns=['unique_id', 'gender'])
y_train = train_features['gender'].map({'F': 1, 'M': 0})  


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import numpy as np

X = train_features.drop(columns=['unique_id', 'gender'])
y = train_features['gender'].map({'F': 1, 'M': 0})

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=1900,
    max_depth=14,
    min_samples_leaf=5,
    min_samples_split=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)


prob_female_val = rf.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.30, 0.71, 0.01)
results = []

for thresh in thresholds:
    preds = (prob_female_val >= thresh).astype(int)
    ber = 1 - balanced_accuracy_score(y_val, preds)
    roc_auc = roc_auc_score(y_val, prob_female_val)
    avg_score = (1 - ber + roc_auc) / 2
    results.append((thresh, ber, roc_auc, avg_score))

# 🔝 En iyi threshold
best = max(results, key=lambda x: x[3])
best_thresh, best_ber, best_auc, best_avg = best

print(f"✅ En iyi threshold: {best_thresh:.2f}")
print(f"📊 Validation BER: {best_ber:.4f}")
print(f"📊 Validation ROC AUC: {best_auc:.4f}")
print(f"📊 Ortalama Skor: {best_avg:.4f}")

X_full = pd.concat([X_train, X_val], axis=0)
y_full = pd.concat([y_train, y_val], axis=0)
rf_final = RandomForestClassifier(
    n_estimators=1900,
    max_depth=14,
    min_samples_leaf=5,
    min_samples_split=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
rf_final.fit(X_full, y_full)

X_test = test_features.drop(columns=['unique_id'], errors='ignore')

train_columns = X_train.columns

X_test = X_test.reindex(columns=train_columns, fill_value=0)
prob_female = rf_final.predict_proba(X_test)[:, 1]

pred_labels = np.where(prob_female >= 0.61, 'F', 'M')

submission = pd.DataFrame({
    'unique_id': test_features['unique_id'],
    'probability_female': prob_female,
    'gender': pred_labels
})

submission.to_csv("test_prediction.csv", index=False)

rf.get_params()

importances = rf.feature_importances_
feat_names = X_train.columns
sorted_idx = np.argsort(importances)[::-1]
for i in range(110):  
    print(f"{feat_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")
