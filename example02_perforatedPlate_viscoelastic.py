from dolfin import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
fileP = XDMFFile("outputs/plate-Viscoelastic-DG-NEW.xdmf")
fileP.parameters['rewrite_function_mesh']=False
fileP.parameters["functions_share_mesh"] = True
fileP.parameters["flush_output"] = True

import matplotlib.pyplot as plt



'''

DG approximation solely in terms of stress with strong symmetry
Displacements are postprocessed from momentum balance
Newmark scheme

'''

def local_project(v, V, u=None):
    """Element-wise projection using LocalSolver"""
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
k      = 1
ndim   = 2
I      = Identity(ndim)
rho    = Constant(1.)
alpha  = Constant(10.)
omega  = Constant(0.1)

# Elastic component:
EC     = Constant(30.)
nuC    = Constant(0.3)
lmbdaC = Constant(EC*nuC/((1.+nuC)*(1.-2.*nuC)))
muC    = Constant(EC/(2.*(1.+nuC)))

ED     = Constant(40.)
nuD    = Constant(0.49)
lmbdaD = Constant(ED*nuD/((1.+nuD)*(1.-2.*nuD)))
muD    = Constant(ED/(2.*(1.+nuD)))

calA = lambda s: 0.5/muC * s - lmbdaC/(2.*muC*(ndim*lmbdaC+2.*muC))*tr(s)*I
calV = lambda s: 0.5/(muD-muC) * s - (lmbdaD-lmbdaC)/(2.*(muD-muC)*(ndim*(lmbdaD-lmbdaC)+2.*(muD-muC)))*tr(s)*I

Tfinal = 7.; dt = 0.05; t = 0.; freqSave = 1; inc = 0;

mesh = Mesh('plate_with_hole.xml')
bdry = MeshFunction("size_t", mesh, "plate_with_hole_facet_region.xml")
top = 31; bot = 30; walls = 32

ds = Measure("ds", domain=mesh, subdomain_data=bdry)
dS = Measure("dS", domain=mesh, subdomain_data=bdry)

n = FacetNormal(mesh); he = FacetArea(mesh)

# ********* Finite dimensional spaces ********* #
Pk = VectorElement('DG', mesh.ufl_cell(), k, dim = 3)
Hh = FunctionSpace(mesh, MixedElement([Pk,Pk]))
print('dofs = ', Hh.dim())

#Vh = VectorFunctionSpace(mesh,'CG',1)
Vh = VectorFunctionSpace(mesh,'DG',k-1)
Th = TensorFunctionSpace(mesh,'DG',1)

# ********* test and trial functions for product space ****** #

eta,  tau   = TestFunctions(Hh)
gamma, zeta = TrialFunctions(Hh)
gamma= tensorify(gamma)
zeta = tensorify(zeta)
eta  = tensorify(eta)
tau  = tensorify(tau)

# ********* Boundary conditions ****** #

intense  = Constant(1.)
traction = Expression(('0','t<=1?-i*sin(pi*t*0.5):0'), i=intense, t= t, degree = 0)

f = Constant((0,0))

# ********** Initial conditions *****
Sol_h = Function(Hh)

Sol_old = Function(Hh) # starting from zero
Sol_oold = Function(Hh) # starting from zero

gamma_old,zeta_old = sp_t(Sol_old) 
gamma_oold,zeta_oold = sp_t(Sol_oold)

u_old = Function(Vh)
u_oold = Function(Vh)

# ********* Weak forms ********* #

AA  = 1/dt**2*inner(calA(gamma),eta)*dx \
    + 1/dt**2*inner(calV(zeta),tau)*dx \
    + 0.5/omega/dt*inner(calV(zeta),tau)*dx \
    + 1/rho*dot(div(gamma+zeta),div(eta+tau))*dx \
    - 1/rho*dot(avg(div(gamma+zeta)),jump(eta+tau,n))*dS \
    - 1/rho*dot(avg(div(eta+tau)),jump(gamma+zeta,n))*dS \
    + alpha*k**2/avg(he)*dot(jump(gamma+zeta,n),jump(eta+tau,n))*dS

# part due to traction BCs
AA += alpha*k**2/he*dot((gamma+zeta)*n,(eta+tau)*n)*ds(top) \
    + alpha*k**2/he*dot((gamma+zeta)*n,(eta+tau)*n)*ds(walls)

# f and ddot_g are zero
BB  = 1/dt**2*inner(calA(2*gamma_old-gamma_oold),eta)*dx \
    + 1/dt**2*inner(calV(2*zeta_old-zeta_oold),tau)*dx \
    + 0.5/omega/dt*inner(calV(zeta_oold),tau)*dx

# part due to traction BCs
BB += alpha*k**2/he*dot(traction,(eta+tau)*n)*ds(top)

lhs = assemble(AA)

solver = LUSolver(lhs, 'mumps')

sig_vec =  []
gam_vec =  []
zet_vec =  []
u_vec = []
time_vec = []

outfile = open("mysol_plate_visco_time.txt","w")

# ********* Time loop ************* # 
while (t < Tfinal):

    t += dt;
    time_vec.append(t)
    print("t=%.2f" % t)
    traction.t = t

    rhs = assemble(BB)
    solver.solve(Sol_h.vector(),rhs)
    gamma,zeta = sp_t(Sol_h)

    u_h = local_project(2*u_old-u_oold+dt**2/rho*(f+div(gamma+zeta)),Vh)
    assign(u_oold,u_old)
    assign(u_old,u_h)

    assign(Sol_oold,Sol_old)
    assign(Sol_old,Sol_h)
    
    gamma_old,zeta_old = sp_t(Sol_old)
    gamma_oold,zeta_oold = sp_t(Sol_oold)

    if (inc % freqSave == 0):
        sigh = local_project(gamma+zeta,Th)
        gamh = local_project(gamma,Th)
        zetah = local_project(zeta,Th)
        u_h.rename("u","u"); fileP.write(u_h, t)
        sigh.rename("sig","sig"); fileP.write(sigh, t)
        gamh.rename("gam","gam"); fileP.write(gamh, t)
        zetah.rename("zet","zet"); fileP.write(zetah, t)

        sig_vec.append(pow(assemble(sigh**2*dx),0.5))
        gam_vec.append(pow(assemble(gamh**2*dx),0.5))
        zet_vec.append(pow(assemble(zetah**2*dx),0.5))
        u_vec.append(pow(assemble(u_h**2*dx),0.5))

        print(time_vec[inc], sig_vec[inc], gam_vec[inc], zet_vec[inc], u_vec[inc], file = outfile)
        
    inc += 1
           
fg1 = plt.figure(figsize=(12,6))
plt.plot(time_vec, gam_vec,'r-',label='gamma')
plt.plot(time_vec, zet_vec,'b--',label='zeta')
plt.plot(time_vec, sig_vec,'k-',label='sigma')
plt.xlabel('time')
plt.ylabel('stresses')
plt.legend()

fg2 = plt.figure(figsize=(10,4)) 
plt.plot(time_vec, u_vec,'k--',label='displ')
plt.xlabel('time')
plt.ylabel('displ')
plt.legend()
plt.show()
