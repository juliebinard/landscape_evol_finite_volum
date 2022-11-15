using Plots
using Distributions
using DelimitedFiles
using LinearAlgebra
using NLsolve
using SparseArrays


r = "VFstatio2_K=e20"
Nit = 200000 # Number of iterations
etape = [i*2000 for i in 1:Nit/1000] # intermediary steps
chemin = ""  # path

#Constants (units : meters and seconds)
g = 10 # in meters per square seconds
m = 0.8*2 # exponent erosion water height
n = 2*m # exponent erosion water speed
e = 1/10^3/3600/2 # erosion speed in meter per second
θ = 39*2*pi/360 # angle of the plane
csat = 317*1000 # g/m^3
ρs = 2.17*100^3 # density of the sediments (salt) in g/m^3
c0 = csat/1000 # concentration at entry
α = 0.1 # coefficient erosion in front of nabla z
β = 1 # exponent erosion nabla z
h0 = 0.5/1000 # Characteristic water height
Vc = 1 # Characteristic water speed
cf = (g*h0*tan(θ))/Vc^2 # Friction coefficient
K = e/20 # Creeping coefficient


# Numerical parameters:
Lc = 0.4 # Characteristic lenght in meter
Lx = Lc # Lenght of the domain
Ly = Lc/4 # Width of the domain
dx = Lc/200 # Size of a step of the grid in the x direction
dy = Lc/200 # Size of a step of the grid in the y direction
s = e/2000 # Sedimentation speed
Nx = convert(Int64,Lx/dx) # Number of point in the grid, in the x direction
Ny = convert(Int64,Ly/dy) # Number of point in the grid, in the y direction
erMax = 10000 # Maximal mean erosion heigh in meters

T = 7200 # Final time
dtsol = T/16000 # time step

x = [i*dx for i in 1:Nx] # x grid
y = [j*dy for j in 1:Ny] # y grid

function principal()
    h = [h0 for i in 1:Nx, j in 1:Ny] # initial water height
    c = c0*ones(Nx,Ny) # initial sediment concentration
    grad2z = tan(θ)^2*ones(Nx, Ny) # |nabla z|^2
    v = zeros(Nx, Ny, 2) # water speed
    v[:,:,1] .= Vc
    z0 = readdlm("$(chemin)z_init.txt") # initial sol height
    z = copy(z0)
    n = 0 # number of iterations in time
    println(" ")
    while n < Nit
        n += 1
        # computation of h the solution of div(hv) = 0 :
        for i in 1:2
            h, v = zerofh(h, z)
        end
        # test if h negative :
        if minimum(h) < 0
            println("h negative at step $n")
            vx = v[1:end,1:end,1]
            vy = v[1:end,1:end,2]
            writedlm("$(chemin)h_$(r)_etape$n.txt", h)
            writedlm("$(chemin)c_$(r)_etape$n.txt", c)
            writedlm("$(chemin)z_$(r)_etape$n.txt", z)
            writedlm("$(chemin)vx_$(r)_etape$n.txt", vx)
            writedlm("$(chemin)vy_$(r)_etape$n.txt", vy)
            return h, c, z, v, n
        end
        # test if h exploded:
        if isnan(h[3,3])
          println("NaN")
          vx = v[1:end,1:end,1]
          vy = v[1:end,1:end,2]
          Nfinal = n
          return h, c, z, vit, Nfinal
        end
        # computation of c :
        zbord = bordPente(z) # add the boudary conditions to z
        hbord = bordh(h) # add the boudary conditions to h
        c, z, grad2z = zerofc(hbord, zbord, v, grad2z)
        # test if c negative :
        if minimum(c) < 0
            println("c negative at step $n")
            vx = v[1:end,1:end,1]
            vy = v[1:end,1:end,2]
            writedlm("$(chemin)h_$(r)_etape$n.txt", h)
            writedlm("$(chemin)c_$(r)_etape$n.txt", c)
            writedlm("$(chemin)z_$(r)_etape$n.txt", z)
            writedlm("$(chemin)vx_$(r)_etape$n.txt", vx)
            writedlm("$(chemin)vy_$(r)_etape$n.txt", vy)
            return h, c, z, v, n, nb, canal, densiteCanaux
        end
        # save the intermediary steps :
        if n in etape
          vx = v[1:end,1:end,1]
          vy = v[1:end,1:end,2]
          println("number of iterations: $n")
          writedlm("$(chemin)h_$(r)_etape$n.txt", h)
          writedlm("$(chemin)c_$(r)_etape$n.txt", c)
          writedlm("$(chemin)z_$(r)_etape$n.txt", z)
          writedlm("$(chemin)vx_$(r)_etape$n.txt", vx)
          writedlm("$(chemin)vy_$(r)_etape$n.txt", vy)
      end
    end
    Nfinal = n
    return h, c, z, v, Nfinal
end

# this function solve by finite volum the equation div(h^(n+1)v^n) = 0 where v^n is given
# we solve a sparse system, using sparse array
function zerofh(h, z)
    I = zeros(0) # list of the non zero rows of the matrix
    J = zeros(0) # list of the non zero colums of the matrix
    V = zeros(0) # non zeros values of the matrix
    b = zeros(Nx*Ny)
    for j in 1:Ny
        for i in 1:Nx
            k = j+(i-1)*Ny # we scan the values (i,j) column by column
            if i == 1 # Dirichlet condition at the top
                push!(I,k)
                push!(J,k)
                push!(V,1/dx)
                b[k] = h0/dx
            elseif i == Nx # Neumann condition at the bottom
                push!(I, k, k)
                push!(J, k, k-Ny)
                push!(V, 1/dx, -1/dx)
            else
                vois = [i+1 j; i-1 j; i mod(j+1-1,Ny)+1; i mod(j-1-1,Ny)+1] # neighbours of (i,j)
                dist = [dx dx dy dy]
                per_y1 = 0
                per_y2 = 0
                if j == 1
                    per_y1 = Ny # periodicity in y
                elseif j == Ny
                    per_y2 = - Ny # periodicity in y
                end
                gradz = [(z[i+1,j]-z[i,j])/dx^2-tan(θ)/dx; (z[i-1,j]-z[i,j])/dx^2+tan(θ)/dx; (z[i,mod(j+1-1,Ny)+1]-z[i,j])/dy^2; (z[i,mod(j-1-1,Ny)+1]-z[i,j])/dy^2] # list of (z_L - z_K)/d^2 for L neighbour of K
                push!(I,k,k,k,k,k)
                push!(J, k, k+Ny, k-Ny, k+1+per_y2, k-1+per_y1)
                push!(V, 0, 0, 0, 0, 0)
                for it in 1:4 # loop over the neighbours of (i,j)
                    V[end-4] += -(h[i,j]+h[vois[it,1],vois[it,2]])/(2*dist[it]^2) + gradz[it]/2
                    V[end-4+it] = (h[i,j]+h[vois[it,1], vois[it,2]])/(2*dist[it]^2) + gradz[it]/2
                end
            end
        end
    end
    A = sparse(I,J,V) # creation of the sparse matrix
    h2 = A \ b # resolution of the system (h2 have size Nx*Ny)
    h = zeros(Nx,Ny)
    for i in 1:Nx
        for j in 1:Ny
            h[i,j] = h2[j+(i-1)*Ny] # computation of h
        end
    end
    # computation of v :
    h1 = bordh(h)
    zbord = bordPente(z)
    v = zeros(Nx,Ny,2)
    v[:,:,1] = Vc/tan(θ)*((h1[1:Nx,2:Ny+1]-h1[3:Nx+2,2:Ny+1]+zbord[1:Nx,2:Ny+1]-zbord[3:Nx+2,2:Ny+1])/(2*dx).+tan(θ))
    v[:,:,2] = Vc/tan(θ)*((h1[2:Nx+1,1:Ny]-h1[2:Nx+1,3:Ny+2]+zbord[2:Nx+1,1:Ny]-zbord[2:Nx+1,3:Ny+2])/(2*dy))
    return h, v
end

# this function solve the equation on c by finite volum in 1D (x seen as the time variable)
# it also solve a time step of the equation on z, with an explicit euler scheme
function zerofc(hbord, zbord, v, grad2z)
    c = c0*ones(Nx+1, Ny)
    zcopie = copy(zbord)
    for i in 2:Nx+1
        for j in 1:Ny
            if hbord[i,j+1] != 0
                tmp = 0
                # Scheme upwind:
                if v[i-1,j,2] > 0
                    tmp = (c[i-1,mod(j-1-1,Ny)+1]-c[i-1,mod(j-1,Ny)+1])*
                        v[i-1,j,2]/v[i-1,j,1]*dx/dy
                else
                    tmp = (c[i-1,mod(j-1,Ny)+1]-c[i-1,mod(j+1-1,Ny)+1])*
                        v[i-1,j,2]/v[i-1,j,1]*dx/dy
                end
                c[i,j] = (c[i-1,j] + tmp +ρs*dx/(hbord[i,j+1]*v[i-1,j,1])*e*(hbord[i,j+1]/h0)^m*
                    ((v[i-1,j,1]^2+v[i-1,j,2]^2)/Vc^2)^(n/2))/(1+ρs*s/csat*dx/(hbord[i,j+1]*v[i-1,j,1]))
                zcopie[i,j+1] += dtsol*(s*c[i,j]/csat-e*(1+α*sqrt(grad2z[i-1,j])^β)*(hbord[i,j+1]/h0)^m*
                    ((v[i-1,j,1]^2+v[i-1,j,2]^2)/Vc^2)^(n/2)) + dtsol*K*((zbord[i-1,j+1]-2*zbord[i,j+1]+
                    zbord[i+1,j+1])/(2*dx^2)+(zbord[i,j]-2*zbord[i,j+1]+zbord[i,j+2])/(2*dy^2))
            else
                c[i,j] = 0
            end
        end
    end
    zbord = zcopie
    for i in 2:Nx+1
        for j in 2:Ny+1
            grad2z[i-1,j-1] = ((zbord[i+1,j]-zbord[i-1,j])/(2*dx))^2+((zbord[i,j+1]-zbord[i,j-1])/(2*dy))^2
        end
    end
    return c[2:end,:], zbord[2:end-1, 2:end-1], grad2z
end

function bordPente(u)
  u1 = zeros(Nx+2,Ny+2)
  u1[2:end-1,2:end-1] = u
  for j in 2:Ny+1
    u1[end,j] = u1[end-1,j] # bottom
  end
  for j in 2:Ny+1
    u1[1,j] = u1[2,j] # up
  end
  for i in 1:Nx+2
    u1[i,1] = u1[i,end-1] # left (periodicity)
  end
  for i in 1:Nx+2
    u1[i,end] = u1[i,2] # right (periodicity)
  end
  return u1
end

function bordh(u)
  u1 = zeros(Nx+2,Ny+2)
  u1[2:end-1,2:end-1] = u
  for j in 2:Ny+1
    u1[end,j] = u1[end-1,j] # bottm
  end
  for j in 2:Ny+1
    u1[1,j] = h0 # up
  end
  for i in 1:Nx+2
    u1[i,1] = u1[i,end-1] # left
  end
  for i in 1:Nx+2
    u1[i,end] = u1[i,2] # right
  end
  return u1
end

function interp(v, x, y) # compute v(x,y) = vgrid[i,j] where (x,y) is a point in the box [i,j]
    if y >= Ly
        y = y-Ly # periodicity
    end
    if y < 0
        y = y+Ly
    end
    i = convert(Int64,floor(x/dx))+1
    j = convert(Int64,floor(y/dy))+1
    if x >= Lx || x < 0
        println("x outside ", x)
    end
    return v[i,j]
end

@time h, c, z, vit, n = principal()
Tfinal = n*dtsol
