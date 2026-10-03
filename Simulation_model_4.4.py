import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_excel('rotterdam_dataset_02.xlsx')
data = data.sort_values('Arrival_Date')
data['Day'] = data['Arrival_Date'].dt.date

dailyCargo = data.groupby('Day')['Cargo_tons'].sum()
nrDays = len(dailyCargo)


lam = {}
meanCargo = {}
fitLognormDist = {}

for vesselType in ['Container', 'Tanker', 'Bulk']:
    ships = data[data['Vessel_Type'] == vesselType]
    cargo = ships['Cargo_tons'].values

    arrivalTimes = (ships['Arrival_Date'] - ships['Arrival_Date'].min()).dt.total_seconds() / 86400
    interArrivalTimes = np.diff(arrivalTimes)
    lam[vesselType] = 1 / np.mean(interArrivalTimes)

    M1 = np.mean(cargo)
    M2 = np.mean(cargo**2)
    s2 = np.log(M2 / M1**2)
    m = np.log(M1) - s2/2
    meanCargo[vesselType] = M1
    fitLognormDist[vesselType] = stats.lognorm(np.sqrt(s2), scale=np.exp(m))

    print(vesselType)
    print('  lambda:', round(lam[vesselType], 4))
    print('  lognormal mu and sigma:', round(m, 4), round(np.sqrt(s2), 4))


meanS = 0
varArrivals = 0
varCargo = 0
for vesselType in ['Container', 'Tanker', 'Bulk']:
    meanS += lam[vesselType] * fitLognormDist[vesselType].mean()
    varArrivals += lam[vesselType] * fitLognormDist[vesselType].mean()**2
    varCargo += lam[vesselType] * fitLognormDist[vesselType].var()

print('mean and std of daily cargo:', round(meanS), round(np.sqrt(varArrivals + varCargo)))
print('  part of the variance from the number of ships:', round(varArrivals / (varArrivals + varCargo), 3))
print('  part of the variance from the cargo per ship:', round(varCargo / (varArrivals + varCargo), 3))


def simPoissonProcess(lam, t, cargoDist):
    expDist = stats.expon(scale=1/lam)
    time = expDist.rvs()
    nT = 0
    totalCargo = 0
    while time < t:
        nT += 1
        totalCargo += cargoDist.rvs()
        time += expDist.rvs()
    return nT, totalCargo


fitNormDist = stats.norm(np.mean(dailyCargo), np.std(dailyCargo))

nrRuns = 10000
simNormal = fitNormDist.rvs(nrRuns)
simAverage = np.zeros(nrRuns)
simCompound = np.zeros(nrRuns)
for i in range(nrRuns):
    for vesselType in ['Container', 'Tanker', 'Bulk']:
        nT, totalCargo = simPoissonProcess(lam[vesselType], 1, fitLognormDist[vesselType])
        simAverage[i] += nT * meanCargo[vesselType]
        simCompound[i] += totalCargo

print('Mean, std and P(S > 650000) of daily cargo')
print('  data:            ', round(np.mean(dailyCargo)), round(np.std(dailyCargo)), round(np.mean(dailyCargo > 650000), 4))
print('  model A normal:  ', round(np.mean(simNormal)), round(np.std(simNormal)), round(np.mean(simNormal > 650000), 4))
print('  model B average: ', round(np.mean(simAverage)), round(np.std(simAverage)), round(np.mean(simAverage > 650000), 4))
print('  model C compound:', round(np.mean(simCompound)), round(np.std(simCompound)), round(np.mean(simCompound > 650000), 4))


bins = np.arange(0, 1100000, 25000)
xs = np.arange(0, 1100000, 1000)

plt.figure()
plt.hist(dailyCargo, bins=bins, rwidth=0.8, density=True)
plt.plot(xs, fitNormDist.pdf(xs), color='red')
plt.axvline(x=650000, color='black')
plt.title('Model A: normal distribution')
plt.xlabel('Cargo (tons per day)')
plt.legend(['Model A', 'Capacity', 'Data'])

plt.figure()
plt.hist(dailyCargo, bins=bins, rwidth=0.8, density=True, alpha=0.5)
plt.hist(simAverage, bins=bins, rwidth=0.8, density=True, alpha=0.5)
plt.axvline(x=650000, color='black')
plt.title('Model B: Poisson arrivals, average cargo')
plt.xlabel('Cargo (tons per day)')
plt.legend(['Capacity', 'Data', 'Model B'])

plt.figure()
plt.hist(dailyCargo, bins=bins, rwidth=0.8, density=True, alpha=0.5)
plt.hist(simCompound, bins=bins, rwidth=0.8, density=True, alpha=0.5)
plt.axvline(x=650000, color='black')
plt.title('Model C: compound Poisson')
plt.xlabel('Cargo (tons per day)')
plt.legend(['Capacity', 'Data', 'Model C'])

plt.figure()
plt.step(np.sort(dailyCargo), np.arange(1, nrDays+1) / nrDays, color='blue', where='post')
plt.plot(xs, fitNormDist.cdf(xs), color='orange')
plt.step(np.sort(simAverage), np.arange(1, nrRuns+1) / nrRuns, color='g', where='post')
plt.step(np.sort(simCompound), np.arange(1, nrRuns+1) / nrRuns, color='red', where='post')
plt.axvline(x=650000, color='black')
plt.xlim(500000, 1000000)
plt.ylim(0.8, 1.01)
plt.title('Upper tail of the daily cargo')
plt.xlabel('Cargo (tons per day)')
plt.legend(['Data', 'Model A', 'Model B', 'Model C', 'Capacity'])

plt.show()
