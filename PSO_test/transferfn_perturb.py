import numpy as np
import matplotlib.pyplot as plt
import TransferFunction_chaitanya as tfc
import scipy
import random


def calc_T(l, sig, rs):
    rs *= 1e-2 # cm to m
    gamma2 = ((rs[1] / rs[2]) ** (2 * l + 1)) * ((1 - sig[1] / sig[2]) / ((l + 1) / l + sig[1] / sig[2]))
    del2 = (sig[2] / sig[3]) * (1 - gamma2 * (l + 1) / l) / (1 + gamma2)
    gamma3 = ((rs[2] / rs[3]) ** (2 * l + 1)) * (1 - del2) / ((l + 1) / l + del2)

    T = lambda r : ((rs[3] / (sig[3] * (l - (1 + l) * gamma3))) * (1 + gamma3 * ((rs[1] / r) ** (2 * l + 1))) *
        ((r / rs[1]) ** l) * (1 + gamma2 * ((rs[2]/rs[1])**(2 * l + 1))) *
        ((1 + gamma3 * ((rs[3] / rs[2]) ** (2 * l + 1))) / (1 + gamma2)))

    return T


def plot_stuff(psample):
    A = sample_ts(8.8, psample)
    plt.semilogy(psample, A, label=[f'radius={8.8}, l={ls}' for ls in l])

    # [plt.plot(ps, sample_ts(i), label=f'{i}, {ls}') for i, ls in np.product((10, 10.01), l)]
    plt.legend()
    plt.xlabel('Perturbation multiplied into conductivities')
    plt.ylabel('Vals in T')
    plt.show()

def plot_2(psample):
    psample = np.random.uniform(.5, 1.5, size=1000)
    A = sample_ts(8.8, psample)

    A = A.T
    print(A.shape, A)
    fig, axs = plt.subplots(5, 1)
    for lsample in range(A.shape[0]):
        print(A[lsample])
        axs[lsample].hist(A[lsample], bins=50, label=str((lsample + 1) * 10))
    
    fig.legend()
    # [plt.plot(ps, sample_ts(i), label=f'{i}, {ls}') for i, ls in np.product((10, 10.01), l)]
    # plt.legend()
    # plt.xlabel('Perturbation multiplied into conductivities')
    # plt.ylabel('Vals in T')
    plt.show()

def plot_one(psample):
    for r in (10, 10.01):
        A = sample_ts(r, psample)
        plt.plot(l, A[1], label=f'radius={r}, perturbation = 0.5')

    # [plt.plot(ps, sample_ts(i), label=f'{i}, {ls}') for i, ls in np.product((10, 10.01), l)]
    plt.legend()
    plt.xlabel('l')
    plt.ylabel('Vals in T')
    plt.show()

# plot_one(0.5 + np.linspace(0, 1, num=100))
psample = np.random.uniform(.5, 1.5, size=1000)
# plot_stuff(psample)
# plot_2(psample)
# plot_stuff(0.9 + np.linspace(0, 0.2, num=100))
# plot_stuff(0.99 + np.linspace(0, 0.02, num=100))


sigma = np.asarray([-np.inf, 1, 1/80, 1])
radii = np.asarray([-np.inf, 10-1.1, 10-.6, 10])
l = np.arange(300) + 1
ts = lambda ps : [calc_T(l, sigma * p, radii) for p in ps]
sample_ts = lambda r, ps : np.asarray([t(r) for t in ts(ps)])


taufn = lambda u : tfc.TransferFunction(sigma[1:] + 
    sigma[1:] * np.random.normal(0, u, sigma[1:].shape), radii[1:])
tau = lambda taufn : lambda u : lambda r : taufn(u).calc_tauL(r = r)[1]

def plot_diffrs(rs, us, taufn=taufn, plotfn=1):
    fig, axs = plt.subplots(len(rs), sharex=True)
    for r in range(len(rs)):
        plobj = axs if len(rs) == 1 else axs[r]
        for u in us:
            f = plobj.plot if plotfn == 1 else plobj.semilogy
            f(l, tau(taufn)(u)(rs[r]), 'o', label=f'r = {rs[r]},' +
                f'normal noise w/ $\sigma ^2$={u}')
        plobj.legend()
        plobj.set_ylabel(f'Tau')
    plt.xlabel('$\ell$')
    toplim = 11 if plotfn == 1 else 301
    plt.xlim(0, toplim)
    plt.show()

# plot_diffrs([8.5, 8.8, 9.1], [0, 0.1, 0.5], plotfn=1)
# plot_diffrs([8.5, 8.8, 9.1], [0, 0.1, 0.5], plotfn=2)

def try_scipy():
    A = np.random.random((50, 300))
    f = lambda u : lambda x : sum((A * x) @ np.vstack(tau(u)(8.8)))
    u = 0
    while u < 3:
        scipy.optimize.minimize(f(u), [3])
        u += 0.1

onehot = lambda n : [1 if i == n else 0 for i in range(len(sigma[1:]))]
unc_sigmaN = lambda n : lambda u : tfc.TransferFunction(sigma[1:] + 
    sigma[1:] * onehot(n) * np.random.normal(0, u, sigma[1:].shape), radii[1:])

# for i in (0, 1, 2):
#     plot_diffrs([8.5, 8.8, 9.1], [0, 0.1, 0.5], taufn=unc_sigmaN(i), plotfn=1)
#     plot_diffrs([8.5, 8.8, 9.1], [0, 0.1, 0.5], taufn=unc_sigmaN(i), plotfn=2)

sample_multi = lambda mus,u,shape : np.random.normal(
    mus[random.randint(0, len(mus) - 1)], u, shape)

unc_sigma_multimodal = lambda mus : lambda n : lambda u : tfc.TransferFunction(sigma[1:] + 
    sigma[1:] * onehot(n) * sample_multi(mus, u, sigma[1:].shape), radii[1:])

# last sigma uncertainty makes it most wonky

'''plot tau for 
    several radii
        several uncertainty
            several l

run scipy opt on tau, adding to uncertainty until no solve
'''

# for i in range(3):
#     plot_diffrs([8.5, 8.8, 9.1], [0, 0.1, 0.5], taufn=unc_sigma_multimodal([0,5])(i), plotfn=1)
''' this made some things not solve. also switched trends w.r.t. radii and Tau sometimes '''

compose = lambda f, g : (lambda x : f(g(x)))

arr_op = np.sum
minfn = compose(arr_op, tfc.TransferFunction(sigma[1:], radii[1:]).calc_tauL)

opt = scipy.optimize.minimize(minfn, x0=8.8, bounds=[[0, radii[1]]])
