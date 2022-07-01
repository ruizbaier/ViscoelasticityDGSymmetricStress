from fenics import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True

import sympy2fenics as sf
str2exp = lambda s: sf.sympy2exp(sf.str2sympy(s))


'''
Convergence test for transient viscoelasticity. 
DG approximation solely in terms of stress with strong symmetry
Displacements are postprocessed from momentum balance
Newmark scheme

Unit square, manufactured solutions. Convergence of the time discretisation
Pure displacement BCs
'''

def local_project(v, V, u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

def sp_t(s):
    a,b = s.split()
    return tensorify(a), tensorify(b)

def tensorify(s):
    if ndim == 2:
        return as_tensor(((s[0],s[1]),
                          (s[1],s[2])))
    elif ndim ==3:
        return as_tensor(((s[0],s[1],s[2]),
                          (s[1],s[3],s[4]),
                          (s[2],s[4],s[5])))
    
# ********* Model coefficients and parameters ********* #

k      = 2
ndim   = 2; I = Identity(ndim)
rho    = Constant(1.)
alpha  = Constant(10.)
omega  = Constant(2.)

# Elastic component:

EC     = Constant(10.)
nuC    = Constant(0.4)
lmbdaC = Constant(EC*nuC/((1.+nuC)*(1.-2.*nuC)))
muC    = Constant(EC/(2.*(1.+nuC)))

print(" muC, lmbdaC ", float(muC), float(lmbdaC))

# Viscoelastic component (larger than elastic, so that (D-C)^{-1} is positive):

ED     = Constant(20.)
nuD    = Constant(0.45)
lmbdaD = Constant(ED*nuD/((1.+nuD)*(1.-2.*nuD)))
muD    = Constant(ED/(2.*(1.+nuD)))

print(" muD, lmbdaD ", float(muD), float(lmbdaD))

calA = lambda s: 0.5/muC * s - lmbdaC/(2.*muC*(ndim*lmbdaC+2.*muC))*tr(s)*I
calV = lambda s: 0.5/(muD-muC) * s - (lmbdaD-lmbdaC)/(2.*(muD-muC)*(ndim*(lmbdaD-lmbdaC)+2.*(muD-muC)))*tr(s)*I

# ******* Exact solutions for error analysis ****** #

d_dt = Constant(-1)

u_str     = '(exp(-t)*(x*y+x**2/(lmbdaC+lmbdaD)),exp(-t)*(x*y+y**2/(lmbdaC+lmbdaD)))'
dt_u_str  = '(-exp(-t)*(x*y+x**2/(lmbdaC+lmbdaD)),-exp(-t)*(x*y+y**2/(lmbdaC+lmbdaD)))'
dtt_u_str = '(exp(-t)*(x*y+x**2/(lmbdaC+lmbdaD)),exp(-t)*(x*y+y**2/(lmbdaC+lmbdaD)))'


mesh = UnitSquareMesh(64,64)
n = FacetNormal(mesh); he = FacetArea(mesh)

# ********* Finite dimensional spaces ********* #
Pk = VectorElement('DG', mesh.ufl_cell(), k, dim = 3)
Hh = FunctionSpace(mesh, MixedElement([Pk,Pk]))
print('dofs = ', Hh.dim())
print('h = ',mesh.hmax())

Vh = VectorFunctionSpace(mesh,'DG',k-1)
Th = TensorFunctionSpace(mesh,'DG',k)

# ********* test and trial functions for product space ****** #

eta,  tau   = TestFunctions(Hh)
gamma, zeta = TrialFunctions(Hh)
gamma= tensorify(gamma)
zeta = tensorify(zeta)
eta  = tensorify(eta)
tau  = tensorify(tau)

Tfinal = 1; 
dtvec = [1./2, 1./4, 1./8, 1./16, 1./32, 1./64] 
nkmax = 6

eu = []; ru = []
es_0 = []; rs_0 = []; es_div = []; rs_div = []
es = []; rs = []; es_jump = []; rs_jump = []

rs.append(0); rs_0.append(0); rs_div.append(0); rs_jump.append(0); ru.append(0)

# ***** Error analysis ***** #

for nk in range(nkmax):
    dt = dtvec[nk]
    print("....... Refinement level : dt = ", dt)
    
    # ********* instantiation of initial conditions ****** #
    t = 0.
    u_ex     = Expression(str2exp(u_str), t = t, lmbdaD=lmbdaD, lmbdaC=lmbdaC, degree=k+4, domain=mesh)
    u_exM    = Expression(str2exp(u_str), t = -dt, lmbdaD=lmbdaD, lmbdaC=lmbdaC, degree=k+4, domain=mesh)
    dt_u_ex  = Expression(str2exp(dt_u_str), t = t, lmbdaD=lmbdaD, lmbdaC=lmbdaC, degree=k+4, domain=mesh)
    dt_uM_ex = Expression(str2exp(dt_u_str), t = -dt, lmbdaD=lmbdaD, lmbdaC=lmbdaC, degree=k+4, domain=mesh)
    dtt_u_ex = Expression(str2exp(dtt_u_str), t = t, lmbdaD=lmbdaD, lmbdaC=lmbdaC, degree=k+4, domain=mesh)
    

    gamma_ex = 2.*muC*sym(grad(u_ex)) + lmbdaC*div(u_ex)*I
    gamma_exM= 2.*muC*sym(grad(u_exM)) + lmbdaC*div(u_exM)*I

    '''
    zeta_ex is tricky to get. But for this very particular case, only valid 
    for exp(-t) time-dependence, we can do:
    u = exp(-t)*ux and d_dt u = -exp(-t)*ux = - u 
    => the action of the time derivative is 'd_dt () = -1 * ()'  
    '''

    zeta_ex  = 1/(d_dt + 1./omega)*(2.*(muD-muC)*sym(grad(dt_u_ex)) \
                                   + (lmbdaD-lmbdaC)*div(dt_u_ex)*I)

    zeta_exM = 1/(d_dt + 1./omega)*(2.*(muD-muC)*sym(grad(dt_uM_ex)) \
                                   + (lmbdaD-lmbdaC)*div(dt_uM_ex)*I)

    
    Sol_old  = project(as_vector((gamma_ex[0,0], gamma_ex[0,1], gamma_ex[1,1],\
                                  zeta_ex[0,0], zeta_ex[0,1], zeta_ex[1,1])),Hh)
    
    Sol_oold = project(as_vector((gamma_exM[0,0], gamma_exM[0,1], gamma_exM[1,1], \
                            zeta_exM[0,0], zeta_exM[0,1], zeta_exM[1,1])), Hh)

    gamma_old,zeta_old = sp_t(Sol_old) 
    gamma_oold,zeta_oold = sp_t(Sol_oold)

    u_old = interpolate(u_ex,Vh)
    u_oold = interpolate(u_exM,Vh)

    # initialise L^infty(0,T;H) norm (max over the time steps)
    E_s0 = 0; E_sdiv = 0; E_sjump=0; E_u = 0; E_s = 0

    # ********* Time loop ************* # 
    while (t < Tfinal):

        if t + dt > Tfinal:
            dt = Tfinal - t
            t  = Tfinal
        else:
            t += dt
            
        print("t=%.4f" % t)
        u_ex.t = t-dt*0.5; dt_u_ex.t = t-dt*0.5; dtt_u_ex.t = t-dt*0.5;

        f_ex   = rho*dtt_u_ex - div(gamma_ex+zeta_ex)
        dt_gamma_plus_zeta_ex = 2.*muD*sym(grad(dt_u_ex))+lmbdaD*div(dt_u_ex)*I-1/omega*zeta_ex
        gamma_hat = 0.25*(gamma+2*gamma_old+gamma_oold)
        zeta_hat = 0.25*(zeta+2*zeta_old+zeta_oold)
    
        FF = 1./dt**2*inner(calA(gamma-2*gamma_old+gamma_oold),eta)*dx \
             + 1./dt**2*inner(calV(zeta-2*zeta_old+zeta_oold),tau)*dx \
             + 0.5/omega/dt*inner(calV(zeta-zeta_oold),tau)*dx \
             + 1/rho*dot(div(gamma_hat+zeta_hat),div(eta+tau))*dx \
             + 1/rho*dot(f_ex,div(eta+tau)) * dx \
             - 1/rho*dot(avg(f_ex),jump(eta+tau,n))*dS \
             - dot((eta+tau)*n,dtt_u_ex)*ds

        # DG terms
        FF += - 1/rho*dot(avg(div(gamma_hat+zeta_hat)),jump(eta+tau,n))*dS \
              - 1/rho*dot(avg(div(eta+tau)),jump(gamma_hat+zeta_hat,n))*dS \
              + alpha*k**2/avg(he)*dot(jump(gamma_hat+zeta_hat,n),jump(eta+tau,n))*dS
    
        AA,BB = system(FF)
        Sol_h = Function(Hh)
        solve(AA==BB, Sol_h, solver_parameters={'linear_solver':'mumps'})
   
        gamma_h,zeta_h = sp_t(Sol_h)
        gamma_mid = 0.5*(gamma_h+gamma_old); zeta_mid = 0.5*(zeta_h+zeta_old)
        
        u_h = local_project(2*u_old-u_oold+dt**2/rho*(f_ex+div(gamma_h+zeta_h)),Vh)
        u_mid = 0.5*(u_h+u_old)
        dt_gamma_plus_zeta_h = (gamma_h-gamma_old+zeta_h-zeta_old)/dt;

        # compute L^infty(0,T;H) errors of each contribution at time t_{n+1/2} 
        E_s0 =max(E_s0,dt*pow(assemble((gamma_ex+zeta_ex-gamma_mid-zeta_mid)**2*dx),0.5))#according to corollary
        #E_s0=max(E_s0,dt*pow(assemble((dt_gamma_plus_zeta_ex-dt_gamma_plus_zeta_h)**2*dx),0.5))#according to theorem
        E_sdiv=max(E_sdiv,dt*pow(assemble((div(gamma_ex+zeta_ex) \
                                           -div(gamma_mid+zeta_mid))**2*dx),0.5))
        E_sjump=max(E_sjump,dt*pow(assemble(1/avg(he)*(jump(gamma_ex+zeta_ex\
                                                            -gamma_mid-zeta_mid,n))**2*dS),0.5))
        E_u = max(E_u,dt*pow(assemble((u_ex-u_mid)**2*dx),0.5))
        E_s = max(E_s, E_s0 + E_sdiv + E_sjump)

        assign(u_oold,u_old)
        assign(u_old,u_h)
        assign(Sol_oold,Sol_old)
        assign(Sol_old,Sol_h)
    
        gamma_old,zeta_old = sp_t(Sol_old)
        gamma_oold,zeta_oold = sp_t(Sol_oold)
        
    # ********* Storing errors ****** #
    
    es_0.append(E_s0)
    es_div.append(E_sdiv)
    es_jump.append(E_sjump)
    es.append(E_s)
    eu.append(E_u)
    
    if(nk>0):
        ru.append(ln(eu[nk]/eu[nk-1])/ln(dtvec[nk]/dtvec[nk-1]))
        rs_0.append(ln(es_0[nk]/es_0[nk-1])/ln(dtvec[nk]/dtvec[nk-1]))
        rs_div.append(ln(es_div[nk]/es_div[nk-1])/ln(dtvec[nk]/dtvec[nk-1]))
        rs_jump.append(ln(es_jump[nk]/es_jump[nk-1])/ln(dtvec[nk]/dtvec[nk-1]))
        rs.append(ln(es[nk]/es[nk-1])/ln(dtvec[nk]/dtvec[nk-1]))
        

# ********* Generating error history ****** #
print('==================================================================================================================')
print('  dt  &  E(s)   &  R(s) &  e0(s)   & r0(s) & ediv(s)  &rdiv(s)& ejump(s) &rjump(s)&   e(u)   &  r(u) ')
print('==================================================================================================================')
print("k, muC, lmbdaC, muD, lmbdaD ", k, float(muC), float(lmbdaC), float(muD), float(lmbdaD))
print('==================================================================================================================')
for nk in range(nkmax):
    print('{:.6f} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f}'.format(dtvec[nk], es[nk], rs[nk], es_0[nk], rs_0[nk], es_div[nk], rs_div[nk], es_jump[nk], rs_jump[nk], eu[nk], ru[nk]))
print('==================================================================================================================')


'''


'''
