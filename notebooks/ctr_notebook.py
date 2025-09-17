# %% Cell 1
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import preprocessing
import pandas as pd
import gzip
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from operator import itemgetter
import xgboost as xgb


# %% Cell 2
n = 40428967 
sample_size = 1000000
skip_values = sorted(random.sample(range(1,n), n-sample_size))


# %% Cell 3
parse_date = lambda val : pd.datetime.strptime(val, '%y%m%d%H')

with gzip.open('/Users/wmeikle/Downloads/avazu-ctr-prediction-3/train.gz') as f:
    train = pd.read_csv(f, parse_dates = ['hour'], date_parser = parse_date, skiprows = skip_values)

train.head()


# %% Cell 4
with gzip.open('/Users/wmeikle/Downloads/avazu-ctr-prediction-3/test.gz') as f:
    test = pd.read_csv(f, parse_dates = ['hour'], date_parser = parse_date)

test.head()


# %% Cell 5
train.dtypes


# %% Cell 6
train.shape


# %% Cell 7
test.shape


# %% Cell 8
train.describe()


# %% Cell 9
train.info()


# %% Cell 10
sns.pairplot(train, height = 1);
plt.title('Pairplot')
plt.show()


# %% Cell 11
cor = train.corr()
plt.figure(figsize=(15,15))
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.title('Heatmap')
plt.show()


# %% Cell 12
sns.countplot(x='click',data=train)
plt.title('Total Number of Clicks');
plt.show();


# %% Cell 13
train['click'].value_counts()/len(train)


# %% Cell 14
train['hour'].describe()


# %% Cell 15
train.groupby('hour').agg({'click':'sum'}).plot(figsize=(12,6))
plt.ylabel('Number of clicks')
plt.xlabel('Hour')
plt.title('Number of Clicks by Hour');


# %% Cell 16
train['hour_of_day'] = train.hour.apply(lambda x: x.hour)
train.groupby('hour_of_day').agg({'click':'sum'}).plot(figsize=(12,6))
plt.ylabel('Number of Clicks')
plt.xlabel('Hour of Day')
plt.title('Click Trends by Hour of Day');


# %% Cell 17
train.groupby(['hour_of_day', 'click']).size().unstack().plot(kind='bar', title="Hour of Day", figsize=(12,6))
plt.ylabel('Count')
plt.xlabel('Hour of Day')
plt.title('Hourly Impressions vs. Clicks');


# %% Cell 18
df_click = train[train['click'] == 1]
df_hour = train[['hour_of_day','click']].groupby(['hour_of_day']).count().reset_index()
df_hour = df_hour.rename(columns={'click': 'impressions'})
df_hour['clicks'] = df_click[['hour_of_day','click']].groupby(['hour_of_day']).count().reset_index()['click']
df_hour['CTR'] = df_hour['clicks']/df_hour['impressions']*100

plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='hour_of_day', data=df_hour, palette = 'tab10')
plt.xlabel('Hour of Day');
plt.title('Hourly CTR');


# %% Cell 19
train['my_dates'] = pd.to_datetime(train['hour'])
train['day_of_week'] = train['my_dates'].dt.day_name()
cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
train.groupby('day_of_week').agg({'click':'sum'}).reindex(cats).plot(figsize=(12,6))
ticks = list(range(0, 7, 1)) 
labels = "Mon Tues Weds Thurs Fri Sat Sun".split()
plt.xticks(ticks, labels)
plt.xlabel('Day of Week')
plt.ylabel('Clicks')
plt.title('Click Trends by Day of Week');


# %% Cell 20
train.groupby(['day_of_week','click']).size().unstack().reindex(cats).plot(kind='bar', title="Day of the Week", figsize=(12,6))
ticks = list(range(0, 7, 1)) 
labels = "Mon Tues Weds Thurs Fri Sat Sun".split()
plt.xticks(ticks, labels)
plt.ylabel('Clicks')
plt.xlabel('Day of Week')
plt.title('Impressions vs. Clicks by Day of Week');


# %% Cell 21
df_click = train[train['click'] == 1]
df_dayofweek = train[['day_of_week','click']].groupby(['day_of_week']).count().reset_index()
df_dayofweek = df_dayofweek.rename(columns={'click': 'impressions'})
df_dayofweek['clicks'] = df_click[['day_of_week','click']].groupby(['day_of_week']).count().reset_index()['click']
df_dayofweek['CTR'] = df_dayofweek['clicks']/df_dayofweek['impressions']*100

plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='day_of_week', data=df_dayofweek, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.ylabel('CTR')
plt.xlabel('Day of Week')
plt.title('Day of Week CTR');


# %% Cell 22
train.head()


# %% Cell 23
print(train.C1.value_counts()/len(train))


# %% Cell 24
train.groupby(['C1', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='C1 Histogram');
plt.ylabel('Clicks')
plt.xlabel('C1')
plt.show()


# %% Cell 25
C1_values = train.C1.unique()
C1_values.sort()
ctr_avg_list=[]
for i in C1_values:
    ctr_avg=train.loc[np.where((train.C1 == i))].click.mean()
    ctr_avg_list.append(ctr_avg)
    print("for value in C1: {},  click through rate: {}".format(i,ctr_avg))


# %% Cell 26
df_c1 = train[['C1','click']].groupby(['C1']).count().reset_index()
df_c1 = df_c1.rename(columns={'click': 'impressions'})
df_c1['clicks'] = df_click[['C1','click']].groupby(['C1']).count().reset_index()['click']
df_c1['CTR'] = df_c1['clicks']/df_c1['impressions']*100

plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='C1', data=df_c1)
plt.title('CTR by C1');


# %% Cell 27
print(train.banner_pos.value_counts()/len(train))


# %% Cell 28
train.groupby(['banner_pos', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Banner Position Histogram');
plt.ylabel('Clicks')
plt.xlabel('Banner Position')
plt.show()


# %% Cell 29
banner_pos = train.banner_pos.unique()
banner_pos.sort()
ctr_avg_list=[]
for i in banner_pos:
    ctr_avg=train.loc[np.where((train.banner_pos == i))].click.mean()
    ctr_avg_list.append(ctr_avg)
    print("for banner position: {},  click through rate: {}".format(i,ctr_avg))


# %% Cell 30
df_banner = train[['banner_pos','click']].groupby(['banner_pos']).count().reset_index()
df_banner = df_banner.rename(columns={'click': 'impressions'})
df_banner['clicks'] = df_click[['banner_pos','click']].groupby(['banner_pos']).count().reset_index()['click']
df_banner['CTR'] = df_banner['clicks']/df_banner['impressions']*100
sort_banners = df_banner.sort_values(by='CTR',ascending=False)['banner_pos'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='banner_pos', data=df_banner, order=sort_banners)
plt.ylabel('CTR')
plt.xlabel('Banner Position')
plt.title('CTR by Banner Position');


# %% Cell 31
train.site_id.nunique()


# %% Cell 32
print('The top 10 site ids that have the most impressions')
print((train.site_id.value_counts()/len(train))[0:10])


# %% Cell 33
top10_sites = train[(train.site_id.isin((train.site_id.value_counts()/len(train))[0:10].index))]
top10_sites_click = top10_sites[top10_sites['click'] == 1]
top10_sites.groupby(['site_id', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 Site IDs Histogram');
plt.ylabel('Clicks')
plt.xlabel('Site ID')


# %% Cell 34
top10_ids = (train.site_id.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_ids:
    click_avg=train.loc[np.where((train.site_id == i))].click.mean()
    click_avg_list.append(click_avg)
    print("for site id value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 35
df_site = top10_sites[['site_id','click']].groupby(['site_id']).count().reset_index()
df_site = df_site.rename(columns={'click': 'impressions'})
df_site['clicks'] = top10_sites_click[['site_id','click']].groupby(['site_id']).count().reset_index()['click']
df_site['CTR'] = df_site['clicks']/df_site['impressions']*100
sort_site = df_site.sort_values(by='CTR',ascending=False)['site_id'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='site_id', data=df_site, order=sort_site)
plt.ylabel('CTR')
plt.xlabel('Site ID')
plt.title('CTR by Top 10 Site ID');


# %% Cell 36
train.site_domain.nunique()


# %% Cell 37
print('The top 10 site domains that have the most impressions')
print((train.site_domain.value_counts()/len(train))[0:10])


# %% Cell 38
top10_domain = train[(train.site_domain.isin((train.site_domain.value_counts()/len(train))[0:10].index))]
top10_domain_click = top10_domain[top10_domain['click'] == 1]
top10_domain.groupby(['site_domain', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 Site Domains Histogram');
plt.xlabel('Site Domain')
plt.ylabel('Clicks')


# %% Cell 39
top10_domains = (train.site_domain.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_domains:
    click_avg=train.loc[np.where((train.site_domain == i))].click.mean()
    click_avg_list.append(click_avg)
    print("for site domain value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 40
df_domain = top10_domain[['site_domain','click']].groupby(['site_domain']).count().reset_index()
df_domain = df_domain.rename(columns={'click': 'impressions'})
df_domain['clicks'] = top10_domain_click[['site_domain','click']].groupby(['site_domain']).count().reset_index()['click']
df_domain['CTR'] = df_domain['clicks']/df_domain['impressions']*100
sort_domain = df_domain.sort_values(by='CTR',ascending=False)['site_domain'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='site_domain', data=df_domain, order=sort_domain)
plt.title('CTR by top 10 site domain');


# %% Cell 41
train.site_category.nunique()


# %% Cell 42
print('The top 10 site categories that have the most impressions')
print((train.site_category.value_counts()/len(train))[0:10])


# %% Cell 43
top10_category = train[(train.site_category.isin((train.site_category.value_counts()/len(train))[0:10].index))]
top10_category_click = top10_category[top10_category['click'] == 1]
top10_category.groupby(['site_category', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 Site Categories Histogram');
plt.xlabel('Site Category')
plt.ylabel('Clicks')


# %% Cell 44
top10_domains = (train.site_category.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_domains:
    click_avg=train.loc[np.where((train.site_category == i))].click.mean()
    click_avg_list.append(click_avg)
    print("for site domain value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 45
df_category = top10_category[['site_category','click']].groupby(['site_category']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_category_click[['site_category','click']].groupby(['site_category']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['site_category'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='site_category', data=df_category, order=sort_category)
plt.xlabel('Site Category')
plt.title('CTR by Top 10 Site Category');


# %% Cell 46
train.device_id.nunique()


# %% Cell 47
print('The top 10 devices that have the most impressions')
print((train.device_id.value_counts()/len(train))[0:10])


# %% Cell 48
top10_device = train[(train.device_id.isin((train.device_id.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['device_id', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 Device IDs Histogram');
plt.ylabel('Clicks')
plt.xlabel('Device ID')


# %% Cell 49
top10_devices = (train.device_id.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.device_id == i))].click.mean()
    click_avg_list.append(click_avg)
    print("for device id value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 50
df_category = top10_device[['device_id','click']].groupby(['device_id']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['device_id','click']].groupby(['device_id']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['device_id'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='device_id', data=df_category, order=sort_category)
plt.xlabel('Device ID')
plt.title('CTR by Top 10 Device ID');


# %% Cell 51
print(str(train.device_ip.nunique()) + " = number of device ips in the dataset")


# %% Cell 52
print('The impressions by device ips')
print((train.device_ip.value_counts()/len(train)))


# %% Cell 53
top10_device = train[(train.device_ip.isin((train.device_ip.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['device_ip', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 Device IPs Histogram');
plt.ylabel('Clicks')
plt.xlabel('Device IP')


# %% Cell 54
top10_devices = (train.device_ip.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.device_ip == i))].click.mean()
    click_avg_list.append(click_avg)
    print("for device id value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 55
df_category = top10_device[['device_ip','click']].groupby(['device_ip']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['device_ip','click']].groupby(['device_ip']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['device_ip'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='device_ip', data=df_category, order=sort_category)
plt.xlabel('Device IP')
plt.title('CTR by Top 10 Device IP');


# %% Cell 56
print(str(train.device_type.nunique()) + " = number of device types in the dataset")


# %% Cell 57
print('The impressions by device types')
print((train.device_type.value_counts()/len(train)))


# %% Cell 58
train[['device_type','click']].groupby(['device_type','click']).size().unstack().plot(kind='bar', title='Device Types');
plt.xlabel('Click')
plt.ylabel('Device Type')


# %% Cell 59
top10_devices = (train.device_type.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.device_type == i))].click.mean()
    click_avg_list.append(click_avg)
    print("for device id value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 60
df_category = top10_device[['device_type','click']].groupby(['device_type']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['device_type','click']].groupby(['device_type']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['device_type'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='device_type', data=df_category, order=sort_category)
plt.xlabel('Device Type')
plt.title('CTR by Top 10 Device Type');


# %% Cell 61
print(str(train.device_model.nunique()) + " = number of device models in the dataset")


# %% Cell 62
print('The impressions by device models')
print((train.device_model.value_counts()/len(train)))


# %% Cell 63
top10_device = train[(train.device_model.isin((train.device_model.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['device_model', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 Device Models Histogram');
plt.ylabel('Clicks')
plt.xlabel('Device Model')


# %% Cell 64
top10_devices = (train.device_model.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.device_model== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for device model value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 65
df_category = top10_device[['device_model','click']].groupby(['device_model']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['device_model','click']].groupby(['device_model']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['device_model'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='device_model', data=df_category, order=sort_category)
plt.xlabel('Device Model')
plt.title('CTR by Top 10 Device Model');


# %% Cell 66
print(str(train.device_conn_type.nunique()) + " = number of device connection types in the dataset")


# %% Cell 67
print('The impressions by device connection')
print((train.device_conn_type.value_counts()/len(train)))


# %% Cell 68
top10_device = train[(train.device_conn_type.isin((train.device_conn_type.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['device_conn_type', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 Device Connection Types Histogram');
plt.ylabel('Clicks')
plt.xlabel('Device Connection Type')


# %% Cell 69
top10_devices = (train.device_conn_type.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.device_conn_type== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for device connection type: {},  click through rate: {}".format(i,click_avg))


# %% Cell 70
df_category = top10_device[['device_conn_type','click']].groupby(['device_conn_type']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['device_conn_type','click']].groupby(['device_conn_type']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['device_conn_type'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='device_conn_type', data=df_category, order=sort_category)
plt.xlabel('Device Model')
plt.title('CTR by Top 10 Device Connection Type');


# %% Cell 71
train.app_id.nunique()


# %% Cell 72
print('The impressions by app ID')
print((train.app_id.value_counts()/len(train)))


# %% Cell 73
top10_device = train[(train.app_id.isin((train.app_id.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['app_id', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 App IDs');
plt.ylabel('Clicks')
plt.xlabel('App ID')


# %% Cell 74
top10_devices = (train.app_id.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.app_id== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for app ID value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 75
df_category = top10_device[['app_id','click']].groupby(['app_id']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['app_id','click']].groupby(['app_id']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['app_id'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='app_id', data=df_category, order=sort_category)
plt.xlabel('Device Model')
plt.title('CTR by Top 10 App ID');


# %% Cell 76
train.app_domain.nunique()


# %% Cell 77
print('The impressions by App Domain')
print((train.app_domain.value_counts()/len(train)))


# %% Cell 78
top10_device = train[(train.app_domain.isin((train.app_domain.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['app_domain', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 App Domains');
plt.ylabel('Clicks')
plt.xlabel('App Domain')


# %% Cell 79
top10_devices = (train.app_domain.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.app_domain== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for App Domain value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 80
df_category = top10_device[['app_domain','click']].groupby(['app_domain']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['app_domain','click']].groupby(['app_domain']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['app_domain'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='app_domain', data=df_category, order=sort_category)
plt.xlabel('Device Model')
plt.title('CTR by Top 10 App Domain');


# %% Cell 81
train.app_category.nunique()


# %% Cell 82
print('The impressions by app categories')
print((train.app_category.value_counts()/len(train)))


# %% Cell 83
top10_device = train[(train.app_category.isin((train.app_category.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['app_category', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 App Category');
plt.ylabel('Clicks')
plt.xlabel('App Category')


# %% Cell 84
top10_devices = (train.app_category.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.app_category== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for App Category value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 85
df_category = top10_device[['app_category','click']].groupby(['app_category']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['app_category','click']].groupby(['app_category']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['app_category'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='app_category', data=df_category, order=sort_category)
plt.xlabel('App Category')
plt.title('CTR by Top 10 App Category');


# %% Cell 86
train.C14.nunique()


# %% Cell 87
print('The impressions by C14 values')
print((train.C14.value_counts()/len(train)))


# %% Cell 88
top10_device = train[(train.C14.isin((train.C14.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['C14', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 C14 Values');
plt.ylabel('Clicks')
plt.xlabel('C14')


# %% Cell 89
top10_devices = (train.C14.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.C14== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for C14 value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 90
df_category = top10_device[['C14','click']].groupby(['C14']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['C14','click']].groupby(['C14']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['C14'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='C14', data=df_category, order=sort_category)
plt.xlabel('C14')
plt.title('CTR by Top 10 C14');


# %% Cell 91
train.C15.nunique()


# %% Cell 92
print('The impressions by C15 values')
print((train.C15.value_counts()/len(train)))


# %% Cell 93
top10_device = train[(train.C15.isin((train.C15.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['C15', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 C15 Values');
plt.ylabel('Clicks')
plt.xlabel('C15')


# %% Cell 94
top10_devices = (train.C15.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.C15== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for C15 value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 95
df_category = top10_device[['C15','click']].groupby(['C15']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['C15','click']].groupby(['C15']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['C15'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='C15', data=df_category, order=sort_category)
plt.xlabel('C15')
plt.title('CTR by Top 10 C15');


# %% Cell 96
train.C16.nunique()


# %% Cell 97
print('The impressions by C16 values')
print((train.C16.value_counts()/len(train)))


# %% Cell 98
top10_device = train[(train.C16.isin((train.C16.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['C16', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 C16 Values');
plt.ylabel('Clicks')
plt.xlabel('C16')


# %% Cell 99
top10_devices = (train.C16.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.C16== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for C16 value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 100
df_category = top10_device[['C16','click']].groupby(['C16']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['C16','click']].groupby(['C16']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['C16'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='C16', data=df_category, order=sort_category)
plt.xlabel('C16')
plt.title('CTR by Top 10 C16');


# %% Cell 101
train.C17.nunique()


# %% Cell 102
print('The impressions by C17 values')
print((train.C17.value_counts()/len(train)))


# %% Cell 103
top10_device = train[(train.C17.isin((train.C17.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['C17', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 C17 Values');
plt.ylabel('Clicks')
plt.xlabel('C17')


# %% Cell 104
top10_devices = (train.C17.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.C17== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for C17 value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 105
df_category = top10_device[['C17','click']].groupby(['C17']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['C17','click']].groupby(['C17']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['C17'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='C17', data=df_category, order=sort_category)
plt.xlabel('C17')
plt.title('CTR by Top 10 C17');


# %% Cell 106
train.C18.nunique()


# %% Cell 107
print('The impressions by C18 values')
print((train.C18.value_counts()/len(train)))


# %% Cell 108
top10_device = train[(train.C18.isin((train.C18.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['C18', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 C18 Values');
plt.ylabel('Clicks')
plt.xlabel('C18')


# %% Cell 109
top10_devices = (train.C18.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.C18== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for C18 value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 110
df_category = top10_device[['C18','click']].groupby(['C18']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['C18','click']].groupby(['C18']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['C18'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='C18', data=df_category, order=sort_category)
plt.xlabel('C18')
plt.title('CTR by Top 10 C18');


# %% Cell 111
train.C19.nunique()


# %% Cell 112
print('The impressions by C19 values')
print((train.C19.value_counts()/len(train)))


# %% Cell 113
top10_device = train[(train.C19.isin((train.C19.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['C19', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 C19 Values');
plt.ylabel('Clicks')
plt.xlabel('C19')


# %% Cell 114
top10_devices = (train.C19.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.C19== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for C19 value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 115
df_category = top10_device[['C19','click']].groupby(['C19']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['C19','click']].groupby(['C19']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['C19'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='C19', data=df_category, order=sort_category)
plt.xlabel('C19')
plt.title('CTR by Top 10 C19');


# %% Cell 116
train.C20.nunique()


# %% Cell 117
print('The impressions by C20 values')
print((train.C20.value_counts()/len(train)))


# %% Cell 118
top10_device = train[(train.C20.isin((train.C20.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['C20', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 C20 Values');
plt.ylabel('Clicks')
plt.xlabel('C20')


# %% Cell 119
top10_devices = (train.C20.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.C20== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for C20 value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 120
df_category = top10_device[['C20','click']].groupby(['C20']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['C20','click']].groupby(['C20']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['C20'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='C20', data=df_category, order=sort_category)
plt.xlabel('C20')
plt.title('CTR by Top 10 C20');


# %% Cell 121
train.C21.nunique()


# %% Cell 122
print('The impressions by C21 values')
print((train.C21.value_counts()/len(train)))


# %% Cell 123
top10_device = train[(train.C21.isin((train.C21.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['C21', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 C21 Values');
plt.ylabel('Clicks')
plt.xlabel('C21')


# %% Cell 124
top10_devices = (train.C21.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.C21== i))].click.mean()
    click_avg_list.append(click_avg)
    print("for C21 value: {},  click through rate: {}".format(i,click_avg))


# %% Cell 125
df_category = top10_device[['C21','click']].groupby(['C21']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_device_click[['C21','click']].groupby(['C21']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['C21'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='C21', data=df_category, order=sort_category)
plt.xlabel('C21')
plt.title('CTR by Top 10 C21');


# %% Cell 126
train.drop('id', axis=1, inplace=True)
train.drop('device_ip', axis=1, inplace=True)
train.drop('hour', axis=1, inplace=True)
train.drop('my_dates', axis=1, inplace=True)


# %% Cell 127
test['hour_of_day'] = test.hour.apply(lambda x: x.hour)
test['my_dates'] = pd.to_datetime(test['hour'])
test['day_of_week'] = test['my_dates'].dt.day_name()


# %% Cell 128
test.drop('hour', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)
test.drop('my_dates', axis=1, inplace=True)
test.drop('device_ip', axis=1, inplace=True)


# %% Cell 129
def convert_obj_to_int(self):
    
    object_list_columns = self.columns
    object_list_dtypes = self.dtypes
    new_col_suffix = '_int'
    for index in range(0,len(object_list_columns)):
        if object_list_dtypes[index] == object :
            self[object_list_columns[index]+new_col_suffix] = self[object_list_columns[index]].map( lambda  x: hash(x))
            self.drop([object_list_columns[index]],inplace=True,axis=1)
    return self
train = convert_obj_to_int(train)


# %% Cell 130
def convert_obj_to_int(self):
    
    object_list_columns = self.columns
    object_list_dtypes = self.dtypes
    new_col_suffix = '_int'
    for index in range(0,len(object_list_columns)):
        if object_list_dtypes[index] == object :
            self[object_list_columns[index]+new_col_suffix] = self[object_list_columns[index]].map( lambda  x: hash(x))
            self.drop([object_list_columns[index]],inplace=True,axis=1)
    return self
test = convert_obj_to_int(test)


# %% Cell 131
test.dtypes


# %% Cell 132
train.dtypes


# %% Cell 133
Y = train.click.values


# %% Cell 134
X = train.loc[:, train.columns != 'click']


# %% Cell 135
from sklearn.model_selection import train_test_split


# %% Cell 136
x_test, x_train, y_test, Y_train = train_test_split(X,Y, test_size = 0.3)


# %% Cell 137
clf_LR = LogisticRegression(penalty='none', fit_intercept=True, max_iter=10000, verbose = 2, class_weight = 'balanced')


# %% Cell 138
clf_LR.fit(x_train, Y_train)


# %% Cell 139
log_reg_pred = clf_LR.predict(x_test)


# %% Cell 140
print(classification_report(y_test, log_reg_pred))


# %% Cell 141
print(roc_auc_score(y_test, log_reg_pred))


# %% Cell 142
from sklearn.metrics import confusion_matrix


# %% Cell 143
print(confusion_matrix(y_test, log_reg_pred))


# %% Cell 144
nav_bayes_model = GaussianNB()


# %% Cell 145
nav_bayes_model.fit(x_train, Y_train)


# %% Cell 146
nav_bayes_pred = nav_bayes_model.predict(x_test)


# %% Cell 147
print(classification_report(y_test, nav_bayes_pred))


# %% Cell 148
print(roc_auc_score(y_test, nav_bayes_pred))


# %% Cell 149
print(confusion_matrix(y_test, nav_bayes_pred))


# %% Cell 150
dec_tree_model = DecisionTreeClassifier()


# %% Cell 151
dec_tree_model.fit(x_train, Y_train)


# %% Cell 152
dec_tree_pred = dec_tree_model.predict(x_test)


# %% Cell 153
print(classification_report(y_test, dec_tree_pred))


# %% Cell 154
print(confusion_matrix(y_test, dec_tree_pred))


# %% Cell 155
import lightgbm as lgb


# %% Cell 156
X_train = train.loc[:, train.columns != 'click']
y_target = train.click.values


# %% Cell 157
msk = np.random.rand(len(X_train)) < 0.8
lgb_train = lgb.Dataset(X_train[msk], y_target[msk])
lgb_eval = lgb.Dataset(X_train[~msk], y_target[~msk], reference=lgb_train)


# %% Cell 158
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': { 'binary_logloss'},
    'num_leaves': 31, # defauly leaves(31) amount for each tree
    'learning_rate': 0.03,
    'feature_fraction': 0.7, # will select 70% features before training each tree
    'bagging_fraction': 0.3, #feature_fraction, but this will random select part of data
    'bagging_freq': 5, #  perform bagging at every 5 iteration
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5000,
                valid_sets=lgb_eval,
                early_stopping_rounds=500)

#0.40054547600913726


# %% Cell 159
print(gbm.best_score)
print(gbm.best_iteration)


# %% Cell 160
with gzip.open('/Users/wmeikle/Downloads/avazu-ctr-prediction-3/test.gz') as f:
    test2 = pd.read_csv(f, parse_dates = ['hour'], date_parser = parse_date)

test2.head()


# %% Cell 161
with gzip.open('/Users/wmeikle/Downloads/avazu-ctr-prediction-3/SampleSubmission.gz') as f:
    SampleSubmission = pd.read_csv(f)


# %% Cell 162
test2['hour_of_day'] = test2.hour.apply(lambda x: x.hour)
test2['my_dates'] = pd.to_datetime(test2['hour'])
test2['day_of_week'] = test2['my_dates'].dt.day_name()


# %% Cell 163
test2.drop('hour', axis=1, inplace=True)
test2.drop('id', axis=1, inplace=True)
test2.drop('my_dates', axis=1, inplace=True)
test2.drop('device_ip', axis=1, inplace=True)


# %% Cell 164
def convert_obj_to_int(self):
    
    object_list_columns = self.columns
    object_list_dtypes = self.dtypes
    new_col_suffix = '_int'
    for index in range(0,len(object_list_columns)):
        if object_list_dtypes[index] == object :
            self[object_list_columns[index]+new_col_suffix] = self[object_list_columns[index]].map( lambda  x: hash(x))
            self.drop([object_list_columns[index]],inplace=True,axis=1)
    return self
test2 = convert_obj_to_int(test2)


# %% Cell 165
predictions = gbm.predict(test2)


# %% Cell 166
my_submission = pd.DataFrame({'id': SampleSubmission['id'], 'click': predictions})


# %% Cell 167
my_submission.to_csv('submissionnewestversion.csv',index = False)