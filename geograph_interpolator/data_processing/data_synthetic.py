# from tqdm import tqdm
# import numpy as np

# from SimPEG.electromagnetics import time_domain
# from SimPEG import (
#     optimization,
#     discretize,
#     maps,
#     data_misfit,
#     regularization,
#     inverse_problem,
#     inversion,
#     directives,
#     utils,
# )

# def quick1D(station):

#     basment_i       = [n.params['labl'] for n in station].index(1)
#     basement_depth  = station[basment_i].nloc[2]

#     cs, ncx, ncz, npad = 5.0, 25, 15, 15
#     hx = [(cs, ncx), (cs, npad, 1.3)]
#     hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
#     mesh = discretize.CylMesh([hx, 1, hz], "00C")

#     active = mesh.vectorCCz < 0.0
#     layer = (mesh.vectorCCz < 0.0) & (mesh.vectorCCz >= basement_depth) # bsament depth
#     actMap = maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
#     mapping = maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * actMap
#     sig_half = 3e-3 # basement
#     sig_air = 1e-8
#     sig_layer = 1e-3 # cover
#     sigma = np.ones(mesh.nCz) * sig_air
#     sigma[active] = sig_half
#     sigma[layer] = sig_layer
#     mtrue = np.log(sigma[active])

#     rxOffset = 1e-3
#     rx = time_domain.Rx.PointMagneticFluxTimeDerivative(
#         np.array([[rxOffset, 0.0, 30]]), np.logspace(-5, -3, 31), "z"
#     )
#     src = time_domain.Src.MagDipole([rx], location=np.array([0.0, 0.0, 80]))
#     survey = time_domain.Survey([src])
#     time_steps = [(1e-06, 20), (1e-05, 20), (0.0001, 20)]
#     simulation = time_domain.Simulation3DElectricField(
#         mesh, sigmaMap=mapping, survey=survey, time_steps=time_steps
#     )

#     # create observed data
#     rel_err = 0.05
#     data = simulation.make_synthetic_data(mtrue, relative_error=rel_err)

#     dmisfit = data_misfit.L2DataMisfit(simulation=simulation, data=data)
#     regMesh = discretize.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
#     reg = regularization.Tikhonov(regMesh, alpha_s=1e-2, alpha_x=1.0)
#     opt = optimization.InexactGaussNewton(maxIter=5, LSshorten=0.5)
#     invProb = inverse_problem.BaseInvProblem(dmisfit, reg, opt)

#     # Create an inversion object
#     beta = directives.BetaSchedule(coolingFactor=5, coolingRate=2)
#     betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
#     inv = inversion.BaseInversion(invProb, directiveList=[beta, betaest])
#     m0 = np.log(np.ones(mtrue.size) * sig_half)
#     simulation.counter = opt.counter = utils.Counter()
#     opt.remember("xc")

#     mopt = inv.run(m0)

#     modeled_response    = np.exp(mopt)
#     model_depth         = mesh.vectorCCz[active]

#     stat_depths = [n.nloc[2] for n in station]

#     y_interp = np.interp(stat_depths, model_depth, modeled_response)

#     return(y_interp)