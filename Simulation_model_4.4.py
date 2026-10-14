import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


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
    cargo = ships['Cargo_tons']

    arrivalTimes = (ships['Arrival_Date'] - ships['Arrival_Date'].min()).dt.total_seconds() / 86400
    interArrivalTimes = np.diff(arrivalTimes)
    lam[vesselType] = 1 / np.mean(interArrivalTimes)

    M1 = np.mean(cargo)
    M2 = np.mean(cargo**2)
    s2 = np.log(M2 / M1**2)
    m = np.log(M1) - s2/2
    meanCargo[vesselType] = M1
    fitLognormDist[vesselType] = stats.lognorm(np.sqrt(s2), scale=np.exp(m))

    print(f'{vesselType} lambda: {round(lam[vesselType], 4)}, mu: {round(m, 4)}, sigma: {round(np.sqrt(s2), 4)}')


meanS = 0
varArrivals = 0
varCargo = 0
for vesselType in ['Container', 'Tanker', 'Bulk']:
    meanS += lam[vesselType] * fitLognormDist[vesselType].mean()
    varArrivals += lam[vesselType] * fitLognormDist[vesselType].mean()**2
    varCargo += lam[vesselType] * fitLognormDist[vesselType].var()

stdS = np.sqrt(varArrivals + varCargo)
shareArrivals = varArrivals / (varArrivals + varCargo)
shareCargo = varCargo / (varArrivals + varCargo)
print(f'Theory mean: {round(meanS)}, std: {round(stdS)}')
print(f'Share of the variance from arrivals: {round(shareArrivals, 3)}, from cargo: {round(shareCargo, 3)}')


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

print(f'Data mean: {round(np.mean(dailyCargo))}, std: {round(np.std(dailyCargo))}, P(S > 650000) = {round(np.mean(dailyCargo > 650000), 4)}')
print(f'Model A mean: {round(np.mean(simNormal))}, std: {round(np.std(simNormal))}, P(S > 650000) = {round(np.mean(simNormal > 650000), 4)}')
print(f'Model B mean: {round(np.mean(simAverage))}, std: {round(np.std(simAverage))}, P(S > 650000) = {round(np.mean(simAverage > 650000), 4)}')
print(f'Model C mean: {round(np.mean(simCompound))}, std: {round(np.std(simCompound))}, P(S > 650000) = {round(np.mean(simCompound > 650000), 4)}')


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
plt.hist(dailyCargo, bins=bins, rwidth=0.8, density=True)
[probs, edges] = np.histogram(simAverage, bins=bins, density=True)
plt.plot(edges[0:len(edges)-1] + 12500, probs, 'ro')
plt.axvline(x=650000, color='black')
plt.title('Model B: Poisson arrivals, average cargo')
plt.xlabel('Cargo (tons per day)')
plt.legend(['Model B', 'Capacity', 'Data'])

plt.figure()
plt.hist(dailyCargo, bins=bins, rwidth=0.8, density=True)
[probs, edges] = np.histogram(simCompound, bins=bins, density=True)
plt.plot(edges[0:len(edges)-1] + 12500, probs, 'ro')
plt.axvline(x=650000, color='black')
plt.title('Model C: compound Poisson')
plt.xlabel('Cargo (tons per day)')
plt.legend(['Model C', 'Capacity', 'Data'])

plt.figure()
plt.step(np.sort(dailyCargo), np.arange(1, nrDays+1) / nrDays, where='post')
plt.plot(xs, fitNormDist.cdf(xs))
plt.step(np.sort(simAverage), np.arange(1, nrRuns+1) / nrRuns, where='post')
plt.step(np.sort(simCompound), np.arange(1, nrRuns+1) / nrRuns, where='post')
plt.axvline(x=650000, color='black')
plt.title('Daily cargo: all models')
plt.xlabel('Cargo (tons per day)')
plt.legend(['Data', 'Model A', 'Model B', 'Model C', 'Capacity'])

plt.show()
