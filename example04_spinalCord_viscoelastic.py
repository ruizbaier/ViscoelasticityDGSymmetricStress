from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"

fileP = XDMFFile("outputs/SpinalCord_newRHOs.xdmf")
fileP.parameters['rewrite_function_mesh']=False
fileP.parameters["functions_share_mesh"] = True
fileP.parameters["flush_output"] = True


mesh = Mesh("withPia.xml")
subdomains = MeshFunction("size_t", mesh, "withPia_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "withPia_facet_region.xml")

# LABELS:
pia = 0;  white=1; grey= 2; tract= 34;  clamp = 32;  sfree = 31

n = FacetNormal(mesh); he = FacetArea(mesh)

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS = Measure("dS", domain=mesh, subdomain_data=boundaries)


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
Tmin = 0.3; t = 0; Tfinal = 1.; dt = 0.01; freqSave = 10; inc = 0;

### material parameters
ndim = 2
I      = Identity(ndim)
alpha  = Constant(15.)

### discontinuous material parameters
Kh = FunctionSpace(mesh, 'DG', 0)
EC = Function(Kh); nuC = Function(Kh)
ED = Function(Kh); nuD = Function(Kh)
omega = Function(Kh)
rho = Function(Kh)

#           PIA     WHITE   GREY
EC_values = [2300., 840.,  1600.] 
nuC_values = [0.3, 0.479,  0.49]
omega_values = [1/1000., 1/6.7, 1/6.7]

rho_values = [1.133, 1.041, 1.045]

ED_values = [2350., 2030.,   2030.]
nuD_values = [0.33, 0.49,    0.49]

for cell_no in range(len(subdomains.array())):
    subdomain_no = subdomains.array()[cell_no]
    EC.vector()[cell_no] = EC_values[subdomain_no]
    nuC.vector()[cell_no] = nuC_values[subdomain_no]
    ED.vector()[cell_no] = ED_values[subdomain_no]
    nuD.vector()[cell_no] = nuD_values[subdomain_no]
    omega.vector()[cell_no] = omega_values[subdomain_no]
    rho.vector()[cell_no] = rho_values[subdomain_no]

#EC.rename("EC","EC"); fileP.write(EC, 0)


lmbdaC = EC*nuC/((1.+nuC)*(1.-2.*nuC))
muC    = EC/(2.*(1.+nuC))
lmbdaD = ED*nuD/((1.+nuD)*(1.-2.*nuD))
muD    = ED/(2.*(1.+nuD))

calA = lambda s: 0.5/muC * s - lmbdaC/(2.*muC*(ndim*lmbdaC+2.*muC))*tr(s)*I
calV = lambda s: 0.5/(muD-muC) * s - (lmbdaD-lmbdaC)/(2.*(muD-muC)*(ndim*(lmbdaD-lmbdaC)+2.*(muD-muC)))*tr(s)*I

k = 1
Pk = VectorElement('DG', mesh.ufl_cell(), k, dim = 3)
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


P = Constant(650.)
traction = Expression(('0','t<=tmin?-P*sin(pi*t/tmin)*exp(-10*pow(t,2)):0'), tmin = 0.75, P=P, t= t, degree = 0)
zerov    = Constant((0,0))
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
    - 1/avg(rho)*dot(avg(div(gamma+zeta)),jump(eta+tau,n))*dS \
    - 1/avg(rho)*dot(avg(div(eta+tau)),jump(gamma+zeta,n))*dS \
    + alpha*k**2/avg(he)*dot(jump(gamma+zeta,n),jump(eta+tau,n))*dS
    
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
ener1_vec = []
ener2_vec = []
u_vec = []
time_vec = []

outfile = open("mysol_spinal.txt","w")

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


    #if (inc % freqSave == 0):
    sigh = local_project(gamma+zeta,Th)
    gamh = local_project(gamma,Th)
    zetah = local_project(zeta,Th)
    u_h.rename("u","u"); fileP.write(u_h, t)
    sigh.rename("sig","sig"); fileP.write(sigh, t)
    gamh.rename("gam","gam"); fileP.write(gamh, t)
    zetah.rename("zet","zet"); fileP.write(zetah, t)

    ener1_vec.append(assemble(0.5*inner(calA(gamma),gamma+zeta)*dx))
    ener2_vec.append(pow(assemble((gamma+zeta)**2*dx \
                                   + (div(gamma+zeta)**2)*dx \
                                   + 1/avg(he)*(jump(gamma+zeta,n))**2*dS \
                                   ),0.5))
    
    sig_vec.append(pow(assemble((gamma+zeta)**2*dx),0.5))
    gam_vec.append(pow(assemble(gamma**2*dx),0.5))
    zet_vec.append(pow(assemble(zeta**2*dx),0.5))
    u_vec.append(pow(assemble(u_h**2*dx),0.5)) 

    print(time_vec[inc], sig_vec[inc], gam_vec[inc], zet_vec[inc], u_vec[inc], ener1_vec[inc], ener2_vec[inc], file = outfile)
        
    
    inc += 1

    
fg1 = plt.figure(figsize=(12,6))
plt.plot(time_vec, gam_vec,'r-',label='gamma')
plt.plot(time_vec, zet_vec,'b--',label='zeta')
plt.plot(time_vec, sig_vec,'k-',label='sigma')
plt.xlabel('time')
plt.ylabel('stress')
plt.legend()
plt.show()

fg2 = plt.figure(figsize=(10,4)) 
plt.plot(time_vec, ener1_vec,'b-',label='energ1')
plt.plot(time_vec, ener2_vec,'r-',label='energ2')
plt.xlabel('time')
plt.ylabel('displ and energy')
plt.legend()
plt.show()

