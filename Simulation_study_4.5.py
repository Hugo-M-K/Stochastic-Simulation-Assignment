import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


data = pd.read_excel('rotterdam_dataset_02.xlsx')
data = data.sort_values('Arrival_Date')
data['Day'] = data['Arrival_Date'].dt.date
dailyCargo = data.groupby('Day')['Cargo_tons'].sum()

types = ['Container', 'Tanker', 'Bulk']
lam = {}
fitLognormDist = {}

for vesselType in types:
    vessels = data[data['Vessel_Type'] == vesselType]
    cargo = vessels['Cargo_tons']

    arrivalTimes = (vessels['Arrival_Date'] - vessels['Arrival_Date'].min()).dt.total_seconds() / 86400
    interArrivalTimes = np.diff(arrivalTimes)
    lam[vesselType] = 1 / np.mean(interArrivalTimes)

    M1 = np.mean(cargo)
    M2 = np.mean(cargo**2)
    s2 = np.log(M2 / M1**2)
    m = np.log(M1) - s2/2
    fitLognormDist[vesselType] = stats.lognorm(np.sqrt(s2), scale=np.exp(m))


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


nrYears = 100
T = 365
n = nrYears * T

shipsPerDay = {}
cargoPerDay = {}
for vesselType in types:
    shipsPerDay[vesselType] = np.zeros(n)
    cargoPerDay[vesselType] = np.zeros(n)

for i in range(n):
    for vesselType in types:
        shipsPerDay[vesselType][i], cargoPerDay[vesselType][i] = simPoissonProcess(lam[vesselType], 1, fitLognormDist[vesselType])

totalShips = shipsPerDay['Container'] + shipsPerDay['Tanker'] + shipsPerDay['Bulk']
totalCargo = cargoPerDay['Container'] + cargoPerDay['Tanker'] + cargoPerDay['Bulk']

congestionDays = np.zeros(nrYears)
for year in range(nrYears):
    yearCargo = totalCargo[year*T:(year+1)*T]
    congestionDays[year] = np.sum(yearCargo > 650000)


congestion = totalCargo > 650000
halfWidth = 1.96 * np.std(congestion) / np.sqrt(n)
dataCongestionDays = np.sum(dailyCargo > 650000) / (len(dailyCargo) / 365)

print('Ships per day mean, std:', round(np.mean(totalShips), 2), round(np.std(totalShips), 2))
print('Cargo per day mean, std:', round(np.mean(totalCargo)), round(np.std(totalCargo)))
print('Quantile 95%, 99%, 99.9%:', round(np.quantile(totalCargo, 0.95)), round(np.quantile(totalCargo, 0.99)), round(np.quantile(totalCargo, 0.999)))
print('P(S > 650000) and 95% CI:', round(np.mean(congestion), 4), round(np.mean(congestion) - halfWidth, 4), round(np.mean(congestion) + halfWidth, 4))
print('Congestion days per year mean, std, min, max:', round(np.mean(congestionDays), 1), round(np.std(congestionDays), 1), np.min(congestionDays), np.max(congestionDays))
print('Congestion days per year in the data:', round(dataCongestionDays, 1))


for vesselType in types:
    x = np.arange(0, np.max(shipsPerDay[vesselType]) + 1)
    plt.figure()
    plt.hist(shipsPerDay[vesselType], bins=np.arange(-0.5, np.max(shipsPerDay[vesselType]) + 1.5), rwidth=0.8, density=True)
    plt.plot(x, stats.poisson(lam[vesselType]).pmf(x), 'ro')
    plt.title('Simulated ships per day: ' + vesselType)
    plt.xlabel('Number of ships')
    plt.legend(['Poisson', 'Simulation'])

x = np.arange(0, np.max(totalShips) + 1)
plt.figure()
plt.hist(totalShips, bins=np.arange(-0.5, np.max(totalShips) + 1.5), rwidth=0.8, density=True)
plt.plot(x, stats.poisson(lam['Container'] + lam['Tanker'] + lam['Bulk']).pmf(x), 'ro')
plt.title('Simulated ships per day: total')
plt.xlabel('Number of ships')
plt.legend(['Poisson', 'Simulation'])

for vesselType in types:
    plt.figure()
    plt.hist(cargoPerDay[vesselType], bins=50, rwidth=0.8, density=True)
    plt.title('Simulated cargo per day: ' + vesselType)
    plt.xlabel('Cargo (tons per day)')

plt.figure()
plt.hist(totalCargo, bins=50, rwidth=0.8, density=True)
plt.axvline(x=650000, color='black')
plt.title('Simulated cargo per day: total')
plt.xlabel('Cargo (tons per day)')
plt.legend(['Capacity', 'Simulation'])

plt.figure()
plt.hist(congestionDays, bins=np.arange(-0.5, np.max(congestionDays) + 1.5), rwidth=0.8, density=True)
plt.axvline(x=dataCongestionDays, color='black')
plt.title('Congestion days per year')
plt.xlabel('Days above 650000 tons')
plt.legend(['Data', 'Simulation'])

plt.figure()
plt.plot(np.arange(T), totalCargo[0:T])
plt.plot(np.arange(T), dailyCargo[0:T])
plt.plot([0, T], [650000, 650000], color='black')
plt.title('One simulated year and the data of 2022')
plt.xlabel('Day of the year')
plt.ylabel('Cargo (tons per day)')
plt.legend(['Simulation', 'Data 2022', 'Capacity'])

plt.show()
