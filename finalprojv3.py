import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import thinkstats2
import thinkplot



#importing dataset
pollution_df = pd.read_csv('pollution_us_2000_2016.csv')

#removing unncessary columns
pollution_df.drop(['Unnamed: 0', 'State Code', 'County', 'County Code', 'Site Num', 'Address'], axis=1)

#removing NA from rows
pollution_df.dropna(axis='rows')


#changing date from string to value
#pollution_df['Date Local'] = pd.to_datetime(pollution_df['Date Local'], format='%Y-%m-%d')


#grouping AQI with the date and state
grp_pollution_df = pollution_df.groupby(['State', 'Date Local', 'NO2AQI', 'O3AQI', 'SO2AQI', 'COAQI'])

#taking mean values of data and state so multiple entries for the day
grp_pollution_df = pollution_df.groupby(['State', 'Date Local']).mean()


#creating a histogram of each of the pollution
count,bin_edges = np.histogram(grp_pollution_df['NO2AQI'])
grp_pollution_df['NO2AQI'].plot(kind='hist',xticks=bin_edges)
plt.xlabel('Amount of NO2 AQI')
plt.ylabel('Parts per Billion')
plt.grid()
plt.show()


count,bin_edges = np.histogram(grp_pollution_df['O3AQI'])
grp_pollution_df['O3AQI'].plot(kind='hist',xticks=bin_edges)
plt.xlabel('Amount of O3 AQI')
plt.ylabel('Parts per Million of O3 AQI')
plt.grid()
plt.show()

count,bin_edges = np.histogram(grp_pollution_df['SO2AQI'])
grp_pollution_df['SO2AQI'].plot(kind='hist',xticks=bin_edges)
plt.xlabel('Amount of SO2 AQI')
plt.ylabel('Parts per Billion of SO2 AQI')
plt.grid()
plt.show()


count,bin_edges = np.histogram(grp_pollution_df[np.isfinite(grp_pollution_df['COAQI'])])
grp_pollution_df[np.isfinite(grp_pollution_df['COAQI'])].plot(kind='hist',xticks=bin_edges)
plt.xlabel('Amount of CO AQI')
plt.ylabel('Parts per Billion of CO AQI')
plt.grid()
plt.show()

#Descriptive Statistics
#Creating histogram for Mean, Mode, Spread of each variable 

NO2_mean = pollution_df.NO2AQI.mean()
NO2_mode = pollution_df.NO2AQI.mode()

plt.figure(figsize=(10,5))
plt.hist(pollution_df.NO2AQI,color='grey')
plt.axvline(NO2_mean,color='red',label='Mean')
plt.axvline(NO2_mode[0],color='green',label='Mode')
plt.xlabel('NO2 AQI')
plt.ylabel('Frequency')
plt.legend()
plt.show()
print('The NO2 mean is:', NO2_mean)
print('The NO2 mode is:', NO2_mode)


O3_mean = pollution_df.O3AQI.mean()
O3_mode = pollution_df.O3AQI.mode()

plt.figure(figsize=(10,5))
plt.hist(pollution_df.O3AQI,color='grey')
plt.axvline(O3_mean,color='red',label='Mean')
plt.axvline(O3_mode[0],color='green',label='Mode')
plt.xlabel('O3 AQI')
plt.ylabel('Frequency')
plt.legend()
plt.show()
print('The O3 mean is:', O3_mean)
print('The O3 mode is:', O3_mode)


SO2_mean = pollution_df.SO2AQI.mean()
SO2_mode = pollution_df.SO2AQI.mode()

plt.figure(figsize=(10,5))
plt.hist(pollution_df.SO2AQI,color='grey')
plt.axvline(SO2_mean,color='red',label='Mean')
plt.axvline(SO2_mode[0],color='green',label='Mode')
plt.xlabel('SO2 AQI')
plt.ylabel('Frequency')
plt.legend()
plt.show()
print('The SO2 mean is:', SO2_mean)
print('The SO2 mode is:', SO2_mode)



CO_mean = pollution_df.COAQI.mean()
CO_mode = pollution_df.COAQI.mode()

plt.figure(figsize=(10,5))
plt.hist(pollution_df.COAQI,color='grey')
plt.axvline(CO_mean,color='red',label='Mean')
plt.axvline(CO_mode[0],color='green',label='Mode')
plt.xlabel('CO AQI')
plt.ylabel('Frequency')
plt.show()
print('The CO mean is:', CO_mean)
print('The CO mode is:', CO_mode)




#PMF
#creating a variable for PMF of NO2 AQI & SO2 AQI
no2_pmf = thinkstats2.Pmf(grp_pollution_df['NO2AQI'])
so2_pmf = thinkstats2.Pmf(grp_pollution_df['SO2AQI'])

thinkplot.PrePlot(2, cols=2)
thinkplot.Hist(no2_pmf, label='NO2', align='right', width=0.75)
thinkplot.Hist(so2_pmf, label='SO2', align='left',width=0.75)
thinkplot.Show(xlabel='Parts per Billion', ylabel='Probability', axis=[0, 80, 0, 0.10])


#creating the CDF of O3 AQI
t = (grp_pollution_df['O3AQI'])
cdf = thinkstats2.Cdf(t, label='O3')
thinkplot.Clf()
thinkplot.Cdf(cdf)
thinkplot.Show(xlabel='Parts per Million', ylabel='CDF')



#plotting a complementary CDF (CCDF) of O3
thinkplot.Cdf(cdf, complement=True)
thinkplot.Show(xlabel='minutes',
               ylabel='CCDF',
               yscale='log')


#normal CDF with a range of parameters  
thinkplot.PrePlot(3)

mus = [1.0, 2.0, 3.0] #should change to my own numbers instead
sigmas = [0.5, 0.4, 0.3]

for mu, sigma in zip(mus, sigmas):
    xs, ps = thinkstats2.RenderNormalCdf(mu=mu, sigma=sigma, low=-1.0, high=4.0)
    label = r'$\mu=%g$, $\sigma=%g$' % (mu, sigma)
    thinkplot.Plot(xs, ps, label=label)

thinkplot.Config(title='Normal CDF', xlabel='x', ylabel='CDF',
                 loc='upper left')
thinkplot.Show()


#Scatterplots
thinkplot.Scatter(grp_pollution_df['NO2AQI'], grp_pollution_df['SO2AQI'], alpha=1)
thinkplot.Config(xlabel='NO2 & SO2 AQI',
                 ylabel='Parts per Billion',
                 axis=[0, 120, 0, 220],
                 legend=False)
thinkplot.Show()
    

thinkplot.Scatter(grp_pollution_df['O3AQI'], grp_pollution_df['COAQI'], alpha=1)
thinkplot.Config(xlabel='O3 & CO2 AQI',
                 ylabel='Parts per Million',
                 axis=[0, 120, 0, 220],
                 legend=False)
thinkplot.Show()
    


#Testing a difference in means # this doesn't show up
class DiffMeansPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data
        
 
data = grp_pollution_df['NO2Mean'], grp_pollution_df['SO2AQI']
ht = DiffMeansPermute(data)
pvalue = ht.PValue()

ht.PlotCdf()
thinkplot.Show(xlabel='test statistic',
               ylabel='CDF')



























