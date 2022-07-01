
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"

fileP = XDMFFile("outputs/CreepTest3D_finer.xdmf")
fileP.parameters['rewrite_function_mesh']=False
fileP.parameters["functions_share_mesh"] = True
fileP.parameters["flush_output"] = True


mesh = BoxMesh(Point(0,0,0),Point(1,0.5,0.5), 24, 12, 12)
n = FacetNormal(mesh); he = FacetArea(mesh)
bdry = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
Left = AutoSubDomain(lambda x: near(x[0], 0))
Right = AutoSubDomain(lambda x: near(x[0], 1))
Bot = AutoSubDomain(lambda x: near(x[1], 0))
Top = AutoSubDomain(lambda x: near(x[1], 0.5))
Front = AutoSubDomain(lambda x: near(x[2], 0))
Back = AutoSubDomain(lambda x: near(x[2], 0.5))
clamp = 31; 
sfree = 33; tract = 34;

Left.mark(bdry, clamp)
Right.mark(bdry, tract)
Bot.mark(bdry, sfree)
Top.mark(bdry, sfree)
Back.mark(bdry, sfree)
Front.mark(bdry, sfree)

ndim = mesh.geometry().dim()
ds = Measure("ds", domain=mesh, subdomain_data=bdry)
dS = Measure("dS", domain=mesh, subdomain_data=bdry)

print('Number of nodes: ',mesh.num_vertices())
print('Number of cells: ',mesh.num_cells())

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

    
### Time stepping parameters
Tmin = 0.3; Tmax = 3.; t = 0; Tfinal = 5.; dt = 0.03; freqSave = 1; inc = 0;

### material parameters

I      = Identity(ndim)
rho    = Constant(1.000)
alpha  = Constant(15.)
omega  = Constant(1./6.)

muC    = Constant(20.)
lmbdaC = Constant(100.)
muD    = Constant(50.)
lmbdaD = Constant(200.)

calA = lambda s: 0.5/muC * s - lmbdaC/(2.*muC*(ndim*lmbdaC+2.*muC))*tr(s)*I
calV = lambda s: 0.5/(muD-muC) * s - (lmbdaD-lmbdaC)/(2.*(muD-muC)*(ndim*(lmbdaD-lmbdaC)+2.*(muD-muC)))*tr(s)*I

k = 1
Pk = VectorElement('DG', mesh.ufl_cell(), k, dim = 6)
Hh = FunctionSpace(mesh, MixedElement([Pk,Pk]))
print('dofs = ', Hh.dim())

Vh = VectorFunctionSpace(mesh,'DG',k-1)
Th = TensorFunctionSpace(mesh,'DG',1)

eta,  tau   = TestFunctions(Hh)
gamma, zeta = TrialFunctions(Hh)
gamma= tensorify(gamma)
zeta = tensorify(zeta)
eta  = tensorify(eta)
tau  = tensorify(tau)

traction = Expression(('tmin<=t-dt && t<=tmax?1:0','0','0'), tmin = Tmin, tmax = Tmax, dt = dt, t= t, degree = 0)
zerov    = Constant((0,0,0))
f        = zerov
ddot_g   = zerov

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

# extra terms on stress boundaries 
#AA += - 1/rho*dot(div(gamma+zeta),(eta+tau)*n)*ds(sfree) \
#    - 1/rho*dot(div(eta+tau),(gamma+zeta)*n)*ds(sfree) \
#    - 1/rho*dot(div(gamma+zeta),(eta+tau)*n)*ds(tract) \
#    - 1/rho*dot(div(eta+tau),(gamma+zeta)*n)*ds(tract)
    
# extra terms on stress boundaries 
AA += alpha*k**2/he*dot((gamma+zeta)*n,(eta+tau)*n)*ds(sfree) \
    + alpha*k**2/he*dot((gamma+zeta)*n,(eta+tau)*n)*ds(tract)
    
# f and ddot_g are zero
BB  = 1/dt**2*inner(calA(2*gamma_old-gamma_oold),eta)*dx \
    + 1/dt**2*inner(calV(2*zeta_old-zeta_oold),tau)*dx \
    + 0.5/omega/dt*inner(calV(zeta_oold),tau)*dx

# extra terms on stress boundaries 
BB += alpha*k**2/he*dot(traction,(eta+tau)*n)*ds(tract)

lhs = assemble(AA)

solver = LUSolver(lhs, 'mumps')

sig_vec =  []
gam_vec =  []
zet_vec =  []
u_vec = []
time_vec = []

outfile = open("mysol_creep_3D_finer.txt","w")

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
fg2 = plt.figure(figsize=(10,4)) 
plt.plot(time_vec, u_vec,'k--',label='displ')
plt.xlabel('time')
plt.ylabel('stress and displ')
plt.legend()
plt.show()
