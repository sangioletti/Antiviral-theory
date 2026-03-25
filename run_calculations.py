"""Driver script for antiviral NP-virus binding theory parameter sweeps."""

import matplotlib.pyplot as plt
import numpy as np

from parameters import Parameters
from theory import rLimit, averageForce_both, CONV_KBT_TO_PN

# ---------------------------------------------------------------------------
# Base parameters
# ---------------------------------------------------------------------------
base_data = {
    'kEff': 1.0, 'kEffRep': 3.0, 'x0': 0, 'sigma': 0.01,
    'NL': 5, 'Nrep': 10, 'kbT': 1.0, 'DG0': -5.0, 'L0': 1.0,
    'maxDG': 5, 'epsilon_self': 1e-7, 'nIntSamples': 100,
    'verbose': False,
    'cV0': 1e-9,   # molar concentration of viruses
    'cNP0': 1e-6,  # molar concentration of nanoparticles
    'rV': 100,     # virus radius
    'NP_type': 'star_polymer',
    'rNP': 10,     # nanoparticle radius
}

# ---------------------------------------------------------------------------
# Parameter sweep ranges
# ---------------------------------------------------------------------------
mySigma = np.logspace(-2, 0, 5)
myDG = range(-16, 5, 2)
myNrep = [0, 6, 12]
mykEff = [0.025, 0.05, 0.1]
mykEffRep = [0.001, 0.01, 0.1]

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
params = Parameters.from_dict(base_data)

for sigma in mySigma:
    for kEff in mykEff:
        for kEffRep in mykEffRep:
            for Nrep in myNrep:
                force = []
                print(f"Calculation for sigma = {sigma}, kEff = {kEff}, kEffRep = {kEffRep}, NRep = {Nrep}")
                for DG in myDG:
                    params.sigma = sigma
                    params.DG0 = DG
                    params.Nrep = Nrep
                    params.kEff = kEff
                    params.kEffRep = kEffRep
                    rLimit(params)
                    print(f"DG = {DG}")

                    fr, fz, fr0, fz0 = averageForce_both(params)

                    force.append((DG, fr, fz, fr0, fz0))
                    params.clear_caches()

                force = np.array(force)
                fileName = f"RESULTS_sigma_{sigma}_kEff_{kEff}_kEffRep_{kEffRep}_NRep_{Nrep}"

                with open(fileName, "w") as myF:
                    myF.write("#F? takes into account how many viral sites are bond, F?2 instead does not")
                    myF.write("#DG( kbT ) Fr ( kbT / nm ) Fz (kbT / nm ) Fr ( pN ) Fz ( pN ) Fr2( kbT/nm ) Fz2( kbT/nm ) \n")
                    for dg, fr, fz, fr0, fz0 in force:
                        myF.write(f"{dg} {fr} {fz} {fr*CONV_KBT_TO_PN} {fz*CONV_KBT_TO_PN} {fr0} {fz0} \n")

                figure = plt.plot(myDG, force[:, 1], "r.", linewidth=2, label='Fr')
                figure = plt.plot(myDG, force[:, 2], "b.", linewidth=2, label='Fz')
                figure = plt.plot(myDG, force[:, 3], "g-", linewidth=2, label='Fr2')
                figure = plt.plot(myDG, force[:, 4], "k-", linewidth=2, label='Fz2')
                plt.xlabel("DG (kbT)")
                plt.ylabel("Force (kbT)")
                plt.legend()
                plt.savefig(fileName + ".pdf")
                plt.close()
