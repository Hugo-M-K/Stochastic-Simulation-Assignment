import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_excel('rotterdam_dataset_02.xlsx', header=0)
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

    print()
    print(vesselType + ' - inter-arrival times (hours)')
    print('  mean = ' + str(round(M1, 2)) + ', std = ' + str(round(np.sqrt(sigma2), 2)))
    print('  exponential: lambda = ' + str(round(lam, 4)) + ' per hour')

    tst1 = stats.kstest(gaps, fitNormDist.cdf)
    print('  p-value normal      = ' + str(round(tst1.pvalue, 4)))
    tst1 = stats.kstest(gaps, fitExpDist.cdf)
    print('  p-value exponential = ' + str(round(tst1.pvalue, 4)))
    tst1 = stats.kstest(gaps, fitLognormDist.cdf)
    print('  p-value lognormal   = ' + str(round(tst1.pvalue, 4)))

    print('  99.9% quantile: data = ' + str(round(np.quantile(gaps, 0.999), 1)) + ', exponential = ' + str(round(fitExpDist.ppf(0.999), 1)))

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
    plt.step(ecdfx, ecdfy, color='blue', where='post')
    plt.plot(xs, fitNormDist.cdf(xs), color='orange')
    plt.plot(xs, fitExpDist.cdf(xs), color='g')
    plt.plot(xs, fitLognormDist.cdf(xs), color='red')
    plt.title(vesselType)
    plt.xlabel('Inter-arrival time (hours)')
    plt.legend(['Data', 'Normal', 'Exponential', 'Lognormal'])


    cargo = ships['Cargo_tons'].values

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

    print()
    print(vesselType + ' - cargo (tons)')
    print('  mean = ' + str(round(M1)) + ', std = ' + str(round(np.sqrt(sigma2))) + ', skewness = ' + str(round(stats.skew(cargo), 2)))
    print('  lognormal: mu = ' + str(round(m, 4)) + ', sigma = ' + str(round(np.sqrt(s2), 4)))

    tst1 = stats.kstest(cargo, fitNormDist.cdf)
    print('  p-value normal      = ' + str(round(tst1.pvalue, 4)))
    tst1 = stats.kstest(cargo, fitExpDist.cdf)
    print('  p-value exponential = ' + str(round(tst1.pvalue, 4)))
    tst1 = stats.kstest(cargo, fitLognormDist.cdf)
    print('  p-value lognormal   = ' + str(round(tst1.pvalue, 4)))

    print('  99.9% quantile: data = ' + str(round(np.quantile(cargo, 0.999))) + ', lognormal = ' + str(round(fitLognormDist.ppf(0.999))))

    top = cargo[cargo > np.quantile(cargo, 0.99)]
    print('  top 1% of ships carry ' + str(round(100 * np.sum(top) / np.sum(cargo), 1)) + '% of all cargo')

    rest = cargo[cargo <= np.quantile(cargo, 0.99)]
    M1rest = np.mean(rest)
    M2rest = np.mean(rest**2)
    s2rest = np.log(M2rest / M1rest**2)
    print('  lognormal sigma without the top 1% = ' + str(round(np.sqrt(s2rest), 4)))

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
    plt.step(ecdfx, ecdfy, color='blue', where='post')
    plt.plot(xs, fitNormDist.cdf(xs), color='orange')
    plt.plot(xs, fitExpDist.cdf(xs), color='g')
    plt.plot(xs, fitLognormDist.cdf(xs), color='red')
    plt.title(vesselType)
    plt.xlabel('Cargo (tons)')
    plt.legend(['Data', 'Normal', 'Exponential', 'Lognormal'])


plt.figure()
plt.hist(data[data['Vessel_Type'] == 'Container']['Cargo_tons'], bins=50, rwidth=0.8, density=True, alpha=0.5)
plt.hist(data[data['Vessel_Type'] == 'Tanker']['Cargo_tons'], bins=50, rwidth=0.8, density=True, alpha=0.5)
plt.hist(data[data['Vessel_Type'] == 'Bulk']['Cargo_tons'], bins=50, rwidth=0.8, density=True, alpha=0.5)
plt.title('Cargo per vessel type')
plt.xlabel('Cargo (tons)')
plt.legend(['Container', 'Tanker', 'Bulk'])

plt.show()
