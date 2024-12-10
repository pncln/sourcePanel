import numpy as np
import matplotlib.pyplot as plt

# Arbitrary V_inf value and radius
Vinf = 100
r = 0.5

# Panel configurations to analyze
panel_configs = [4, 8, 16, 32]
markers = ['o', 's', '^', 'v']  # Different markers for each configuration

plt.figure(figsize=(12, 8))

# Generate analytical solution
angles = np.linspace(-0.5*np.pi, 1.5*np.pi, 10000)
Cp_exact = 1-4*np.sin(angles)**2
plt.plot(angles, Cp_exact, '-k', linewidth=2, label='Analytical')

# Calculate for each panel configuration
for t_panel, marker in zip(panel_configs, markers):
    # Create arrays for panel angles and coordinates
    mid_angle = np.arange(np.pi + np.pi/t_panel, -np.pi + np.pi/t_panel, -2*np.pi/t_panel)
    X = np.append(r * np.cos(mid_angle), r * np.cos(mid_angle)[0])
    Y = np.append(r * np.sin(mid_angle), r * np.sin(mid_angle)[0])

    # Initialize arrays
    phi = np.zeros(t_panel)
    beta = np.zeros(t_panel)
    xcontrol = np.zeros(t_panel)
    ycontrol = np.zeros(t_panel)
    S = np.zeros(t_panel)

    # Calculate panel properties
    for i_panel in range(t_panel):
        phi[i_panel] = np.arctan2(Y[i_panel+1]-Y[i_panel], X[i_panel+1]-X[i_panel])
        beta[i_panel] = phi[i_panel] + np.pi/2
        xcontrol[i_panel] = (X[i_panel+1] + X[i_panel])/2
        ycontrol[i_panel] = (Y[i_panel+1] + Y[i_panel])/2
        S[i_panel] = np.sqrt((Y[i_panel+1]-Y[i_panel])**2 + (X[i_panel+1]-X[i_panel])**2)

    # Initialize matrices
    I_i = np.zeros((t_panel, t_panel))
    I_j = np.zeros((t_panel, t_panel))
    V = np.zeros((t_panel, 1))

    # Source panel method calculations
    for n_panel in range(t_panel):
        neigh = np.concatenate([np.arange(n_panel), np.arange(n_panel+1, t_panel)])
        xi = xcontrol[n_panel]
        yi = ycontrol[n_panel]
        
        for i_panel in range(t_panel-1):
            ne = neigh[i_panel]
            X_j = X[ne]
            Y_j = Y[ne]
            A = -(xi-X_j)*np.cos(phi[ne]) - (yi-Y_j)*np.sin(phi[ne])
            B = (xi-X_j)**2 + (yi-Y_j)**2
            C = np.sin(phi[n_panel]-phi[ne])
            D = (yi-Y_j)*np.cos(phi[n_panel]) - (xi-X_j)*np.sin(phi[n_panel])
            E = np.sqrt(B-A**2)
            Sj = S[ne]
            
            I_i[n_panel,ne] = C/2*np.log((Sj**2+2*A*Sj+B)/B) + \
                (D-A*C)/E*(np.arctan2((Sj+A),E)-np.arctan2(A,E))
            I_j[n_panel,ne] = (D-A*C)/2/E*np.log((Sj**2+2*A*Sj+B)/B) - \
                C*(np.arctan2((Sj+A),E)-np.arctan2(A,E))
        
        V[n_panel,0] = Vinf*np.cos(beta[n_panel])

    lambda_matrix = I_i/(2*np.pi) + np.eye(t_panel)/2
    lambda_val = -np.linalg.inv(lambda_matrix) @ V

    # Calculate velocity and pressure coefficient
    V = Vinf*np.sin(beta) + lambda_val.T[0]/(2*np.pi) @ I_j.T
    Cp = 1-(V/Vinf)**2

    # Sort and plot
    sort_idx = np.argsort(beta)
    beta = beta[sort_idx]
    Cp = Cp[sort_idx]
    
    plt.plot(beta, Cp, '--', marker=marker, linewidth=1.5, markersize=6, 
             label=f'Panel Method ({t_panel} panels)')

plt.legend()
plt.xlim([-2, 5])
plt.ylim([-4, 2])
plt.xlabel('θ [rad]')
plt.ylabel('C_p')
plt.title('Pressure Coefficient Distribution Around Cylinder')
plt.grid(True)
plt.show()