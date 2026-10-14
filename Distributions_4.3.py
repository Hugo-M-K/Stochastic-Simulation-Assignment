import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_excel('rotterdam_dataset_02.xlsx')
data = data.sort_values('Arrival_Date')

for vesselType in ['Container', 'Tanker', 'Bulk']:
    ships = data[data['Vessel_Type'] == vesselType]


    arrivalTimes = (ships['Arrival_Date'] - ships['Arrival_Date'].min()).dt.total_seconds() / 3600
    gaps = np.diff(arrivalTimes)

    M1 = np.mean(gaps)
    M2 = np.mean(gaps**2)

    mu = M1
    sigma2 = M2 - M1**2
    fitNormDist = stats.norm(mu, np.sqrt(sigma2))

    lam = 1/M1
    fitExpDist = stats.expon(scale=1/lam)

    s2 = np.log(M2 / M1**2)
    m = np.log(M1) - s2/2
    fitLognormDist = stats.lognorm(np.sqrt(s2), scale=np.exp(m))

    tstNorm = stats.kstest(gaps, fitNormDist.cdf)
    tstExp = stats.kstest(gaps, fitExpDist.cdf)
    tstLognorm = stats.kstest(gaps, fitLognormDist.cdf)

    print(f'{vesselType} inter-arrival times: lambda = {round(lam, 4)} per hour')
    print(f'p-value normal: {round(tstNorm.pvalue, 4)}, exponential: {round(tstExp.pvalue, 4)}, lognormal: {round(tstLognorm.pvalue, 4)}')
    print(f'99.9% quantile data: {round(np.quantile(gaps, 0.999), 1)}, exponential: {round(fitExpDist.ppf(0.999), 1)}')

    xs = np.arange(0.01, np.max(gaps), 0.1)
    plt.figure()
    plt.hist(gaps, bins=50, rwidth=0.8, density=True)
    plt.plot(xs, fitNormDist.pdf(xs), color='orange')
    plt.plot(xs, fitExpDist.pdf(xs), color='g')
    plt.plot(xs, fitLognormDist.pdf(xs), color='red')
    plt.title(vesselType)
    plt.xlabel('Inter-arrival time (hours)')
    plt.legend(['Normal', 'Exponential', 'Lognormal'])

    ecdfx = np.sort(gaps)
    ecdfy = np.arange(1, len(gaps)+1) / len(gaps)
    plt.figure()
    plt.step(ecdfx, ecdfy, where='post')
    plt.plot(xs, fitNormDist.cdf(xs))
    plt.plot(xs, fitExpDist.cdf(xs))
    plt.plot(xs, fitLognormDist.cdf(xs))
    plt.title(vesselType)
    plt.xlabel('Inter-arrival time (hours)')
    plt.legend(['Data', 'Normal', 'Exponential', 'Lognormal'])


    cargo = ships['Cargo_tons']

    M1 = np.mean(cargo)
    M2 = np.mean(cargo**2)

    mu = M1
    sigma2 = M2 - M1**2
    fitNormDist = stats.norm(mu, np.sqrt(sigma2))

    lam = 1/M1
    fitExpDist = stats.expon(scale=1/lam)

    s2 = np.log(M2 / M1**2)
    m = np.log(M1) - s2/2
    fitLognormDist = stats.lognorm(np.sqrt(s2), scale=np.exp(m))

    tstNorm = stats.kstest(cargo, fitNormDist.cdf)
    tstExp = stats.kstest(cargo, fitExpDist.cdf)
    tstLognorm = stats.kstest(cargo, fitLognormDist.cdf)

    top = cargo[cargo > np.quantile(cargo, 0.99)]
    rest = cargo[cargo <= np.quantile(cargo, 0.99)]
    M1rest = np.mean(rest)
    M2rest = np.mean(rest**2)
    s2rest = np.log(M2rest / M1rest**2)
    topShare = np.sum(top) / np.sum(cargo)

    print(f'{vesselType} cargo mean: {round(M1)}, std: {round(np.sqrt(sigma2))}, skewness: {round(stats.skew(cargo), 2)}')
    print(f'lognormal mu: {round(m, 4)}, sigma: {round(np.sqrt(s2), 4)}')
    print(f'p-value normal: {round(tstNorm.pvalue, 4)}, exponential: {round(tstExp.pvalue, 4)}, lognormal: {round(tstLognorm.pvalue, 4)}')
    print(f'99.9% quantile data: {round(np.quantile(cargo, 0.999))}, lognormal: {round(fitLognormDist.ppf(0.999))}')
    print(f'share of cargo in top 1%: {round(topShare, 3)}, sigma without top 1%: {round(np.sqrt(s2rest), 4)}')

    xs = np.arange(1, np.max(cargo), 100)
    plt.figure()
    plt.hist(cargo, bins=50, rwidth=0.8, density=True)
    plt.plot(xs, fitNormDist.pdf(xs), color='orange')
    plt.plot(xs, fitExpDist.pdf(xs), color='g')
    plt.plot(xs, fitLognormDist.pdf(xs), color='red')
    plt.title(vesselType)
    plt.xlabel('Cargo (tons)')
    plt.legend(['Normal', 'Exponential', 'Lognormal'])

    ecdfx = np.sort(cargo)
    ecdfy = np.arange(1, len(cargo)+1) / len(cargo)
    plt.figure()
    plt.step(ecdfx, ecdfy, where='post')
    plt.plot(xs, fitNormDist.cdf(xs))
    plt.plot(xs, fitExpDist.cdf(xs))
    plt.plot(xs, fitLognormDist.cdf(xs))
    plt.title(vesselType)
    plt.xlabel('Cargo (tons)')
    plt.legend(['Data', 'Normal', 'Exponential', 'Lognormal'])


plt.figure()
for vesselType in ['Container', 'Tanker', 'Bulk']:
    cargo = data[data['Vessel_Type'] == vesselType]['Cargo_tons']
    plt.step(np.sort(cargo), np.arange(1, len(cargo)+1) / len(cargo), where='post')
plt.title('Cargo per vessel type')
plt.xlabel('Cargo (tons)')
plt.legend(['Container', 'Tanker', 'Bulk'])

plt.show()
