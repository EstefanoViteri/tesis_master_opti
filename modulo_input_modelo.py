# Importación de bibliotecas necesarias
from gurobipy import GRB,Model,quicksum
import pandas as pd

# Funciones de ayuda
def CD(j):
    return 0.599 + 0.1083*j

def ide(k,K_E,K_R,PD,PE_N,PE_D,rho_N):
    if k in K_R:
        return (rho_N * PE_N + (1-rho_N)*PE_D)
    elif k in K_E:
        return PE_N
    else:
        return PD

def tam_bat(i,K_E):
    if i in K_E:
        return int(i.split('_')[3])
    else:
        return 0

def crear_input(datos,params):
    # Parametros generales

    t1      = 2025   # Periodo inicial
    num_per = params['Periodos']#int(param['Cantidad de periodos']) # incluido el inicial
    dr      = 0.03   # Tasa de descuento anual
    beta    = 1/(1 + dr)
    Gamma   = 7.5 # Promedio de vida maxima final años
    F       = 4.053*12 # Cargos por demanda al año (USD por kW)
    B       = params['presupuesto'][1] if params['presupuesto'][0] else 0 # Presupuesto
    T       = range(t1,t1 + num_per)
    t_F     = t1 + num_per - 1
    # Si t1 = 2025 y num_per = 6, entonces T = [2025, 2026, 2027, 2028, 2029, 2030] y T_f = 2030
    
    '''*********************************************************************************'''
    
    # Estado actual de la flota
    buses     = datos['Buses']
    K         = list(buses['Codigo']) # Tipos de buses
    K_E       = [i for i in K if '_EB_' in i] # Tipos de buses eléctricos
    K_R       = [i for i in K if '_EB_' in i and 'CT' in i] # Tipos de buses eléctricos
    J_K       = {i[0]: range(0,i[1]+1) for i in buses[['Codigo','Vida útil (años)']].values} # Vida útil (años)
    mc        = {(i[0],t): i[1] for i in buses[['Codigo','Costo de media vida']].values for t in T} # Costo media vida
    alpha     = {i[0]: i[1] for i in buses[['Codigo','Media vida (años)']].values} # Edad de media vida
    kappa     = {i[0]: i[1] - 1 for i in buses[['Codigo','Vida útil (años)']].values} # Edad máxima o períodos máximos de circulación
    J_K_tilde = {i: [a for a in j if a != (kappa[i]+1)]  for i,j in J_K.items()}
    tipo_bus  = {i[0]: i[1] for i in buses[['Codigo','Tipo']].values} # Eléctrico (EB) o a Diesel (Diesel)
    efic_bus  = {i[0]: i[1] for i in buses[['Codigo','Eficiencia de combustible (km/combustible)']].values} # Eficiencia (km/combustible)
    cost_ini_bus  = {i[0]: i[1] for i in buses[['Codigo','Costo de compra (USD)']].values} # Costo de compra inicial (USD)

    # Edad flota y precios de salvamento
    flota = datos['Flota'].set_index('Edad')
    salva = datos['Flota salvamento'].set_index('Edad')
    a = {(j,i): val if val > 0 else 0 for j in flota.columns for i, val in flota[j].items() } # (bus,edad): # buses
    s = {(j,i): val if val > 0 else 0 for j in salva.columns for i, val in salva[j].items() } # (bus,edad): precio venta
    
    '''*********************************************************************************'''
    
    # Información Cargadores
    carg    = datos['Cargadores']
    CT      = list(carg['Codigo']) # Tipos de cargadores
    c_tilde = [a for a in CT if 'Pila' in a][0] # Cargador tipo Pila
    e       = {i[0]: i[1] for i in carg[['Codigo','Inventario']].values} # Cargadores iniciales
    p       = {i[0]: i[1] for i in carg[['Codigo','Costo de compra (USD)']].values} # Costo de compra por unidad de cargador (USD)
    g       = {i[0]: i[1] for i in carg[['Codigo','Potencia de carga (kW)']].values} # Potencia de carga (kW)
    theta   = {i[0]: int(i[1].split(':')[1])/int(i[1].split(':')[0]) if 'o' not in i[1] else 0 for i in carg[['Codigo','Relacion bus-cargador']].values} # Relación bus-cargador
    cont    = datos['Contenedor'] # se asume tener solo un tipo de contenedor
    id_cont = cont['Codigo'][0] # Codigo del contenedor
    e_tilde = cont['Inventario'][0] # Contenedores iniciales
    p_tilde = cont['Costo de compra (USD)'][0] # Costo contenedor
    theta_tilde =  int(cont['Relacion contenedor-pila'][0].split(':')[1]) # Capacidad contenedor

    '''*********************************************************************************'''
    
    # Compatibilidades bus-cargador y bus-ruta

    comp_bus_carg = datos['Compat buses-cargadores'].set_index('Cargadores\n          Buses')
    comp_bus_rut = datos['Compat buses-rutas'].set_index('Buses\n          Rutas')

    # Cargadores factibles para cada tipo de bus
    C     = {i: [j for j in comp_bus_carg.columns if comp_bus_carg.at[i,j] == 'x'] for i in comp_bus_carg.index} # {bus:[cargadores]}
    # Rutas factibles para cada tipo de bus
    h_aux = {j: [i for i, val in comp_bus_rut[j].items() if val == 'x'] for j in comp_bus_rut.columns  } # {bus: [rutas]}
    
    '''*********************************************************************************'''
    
    # Informacion de rutas

    rutas = datos['Rutas'].set_index('Ruta')
    R = list(rutas.index)
    d = {r : rutas.at[r,'Distancia'] for r in rutas.index } # Distancia por ruta
    m = {(r,h.split(' ')[0]) : rutas.at[r,h] for r in rutas.index for h in rutas.columns if h not in ['Distancia','Bus de 12 metros'] and 'Flota' in h} # Flota requerida
    n = {(r,h.split(' ')[0]) : rutas.at[r,h] for r in rutas.index for h in rutas.columns if h not in ['Distancia','Bus de 12 metros'] and 'Vueltas' in h} # Flota requerida
    tau = {'(L-V)': 236, '(S)': 47, '(D)': 47} # Días de cada horario en un año
    H = tau.keys() # Horarios

    DA = {r: d[r] * sum(tau[h] * (n[(r,h)] / m[(r,h)]) for h in H) for r in R} # Distancia anual de cada ruta

    #ruta_km = {i[0]: i[1] for i in rutas[['Ruta','Distancia']].values}
    q = {r: m[(r,'(L-V)')] for r in R}
    # Se asume que solo hay el tipo de bus de 12 metros
    ruta_tam_bus = {r: '12' for r in rutas.index} # {ruta: [tamaños]}
    
    '''*********************************************************************************'''
    
    # Costo de mantenimiento
    correc = ( sum(num_bus * (CD(j)) for (k,j),num_bus in a.items()) / (sum(num_bus for (k,j),num_bus in a.items()))  )
    gamma = 0.24 / correc
    o_inv = {(k,r,t,j): gamma * CD(j) * (1 if k not in K_E else params['mantenimiento_elec_ahorro']) * DA[r] for k in K for j in J_K[k] for t in T for r in R}
    #print('gamma = 0.24/{}'.format(round(correc,2)))

    # Costos de combustible/energia
    rho_N = params['rho_N']
    PD    = 0.44 # Precio del diesel subsidiado
    PDNS  = 1.08  # Precio del diesel no subsidiado
    PE_N  = params['Precio_Elect_Noche'] # Precio energia noche
    PE_T  = 0.096 # Precio energia tarde
    PE_D  = params['Precio_Elect_Dia'] # Precio energia dia
    epsilon = (PE_D - PE_N) / PE_N # Eficiencia de carga
    PD_param = params['Precio_diesel']
    o_techo = {}
    o_techo = {(k,r,t,j): ide(k,K_E,K_R,PD_param,PE_N,PE_D,rho_N) * (1/efic_bus[k]) * DA[r] for k in K for j in J_K[k] for t in T for r in R}
    
    o = {i: o_inv[i] + o_techo[i] for i in o_inv.keys()}
    pd.DataFrame([[i[0],i[1],i[2],i[3],o_inv[i],o_techo[i],j] for i,j in o.items()],
                 columns = ['Tipo de autobus','Ruta','Periodo','Edad','Costo mantenimiento','Costo combustible','Costo operativo']).to_excel('costos_operativos.xlsx')
    
    '''*********************************************************************************'''
    
    # Precios de compra de autobuses
    # $300 por kHw en 2018 y reducción anual del 12%
    bat_ini = 300
    anio_bat = 2018
    red_anio = params['reduccion_anual_precio_bateria']
    
    costo_kwh = {t: bat_ini * (1-red_anio)**(per) for per,t in enumerate(range(anio_bat,t_F +1))}
    costo_kwh_Schmidt = {2025: 173,2026: 160,2027: 148,2028: 136,2029: 126,2030: 117,2031: 110,2032: 103,
                       2033: 97,2034: 91,2035: 87,2036: 83,2037: 79,2038: 77,2039: 74,2040: 72,2041: 70,
                       2042: 68,2043: 66,2044: 65,2045: 64,2046: 63,2047: 62,2048: 61,2049: 60,2050: 59 }
    
    if len(T) == len(costo_kwh_Schmidt.keys()):
        costo_kwh = {t: costo_kwh_Schmidt[t] for t in T}
        print('Costo de baterias Schmidt')
    else:
        print('Costo de baterias reduccion anual')
    cost_bus_sin_bat = {i: j - (tam_bat(i,K_E) * costo_kwh[t1]) for i,j in cost_ini_bus.items()}
    f = {(k,t): cost_bus_sin_bat[k] + (tam_bat(k,K_E) * costo_kwh[t]) for k in K for t in T}

    h = {(k,r,t,j): 1 if r in h_aux[k] else 0 for k in K for j in J_K[k] for t in T for r in R} 
    
    '''*********************************************************************************'''
    
    # Objetivos de descarbonizacion
    eta_param = params['Tasa_redu_emisiones']
    iota = {'SO2': 0.21, 'PM25': 0.31, 'NOX': 17.9} # contaminante: cant. emisiones (gramos por km)
    L = list(iota.keys())
    Theta_tilde = {l: sum(DA[r] * iota[l] * m[(r,'(L-V)')] for r in R) for l in L}
    var_epsilon = {(k,j,r,l): 0 if k in K_E else (DA[r] * iota[l]) for l in L for r in R for k in K for j in J_K[k] }
    Theta = {(l,t): max(0, 1 - ((t_+1) * eta_param) ) * Theta_tilde[l] for t_,t in enumerate(T) for l in L}
    #print('Emision anual de toda la flota:')
    #for i,j in Theta_tilde.items():
    #    print('{}: {}'.format(i,round(j,2)))

    return L,CT,T,K,J_K,J_K_tilde,K_E,K_R,C,alpha,R,beta,f,s,kappa,mc,p,F,g,p_tilde,num_per,o,o_inv,o_techo,q,h,t1,a,e,e_tilde,t_F,Gamma,theta,theta_tilde,B,Theta,Theta_tilde,var_epsilon,c_tilde,costo_kwh,cost_bus_sin_bat,efic_bus,DA,epsilon,rho_N,PE_N

def crear_escenarios(params,num_escenario):
    parametros = {i: j for i,j in params.items()}
    parametros['escenario'] = num_escenario
    if num_escenario == 1:
        parametros['Precio_diesel'] = 0.76
    if num_escenario == 2:
        parametros['Precio_diesel'] = 1.08
    if num_escenario == 3:
        parametros['Precio_Elect_Noche'] = 0.0803
        parametros['Precio_Elect_Dia']   = 0.0924
    if num_escenario == 4:
        parametros['Precio_Elect_Noche'] = 0.0876
        parametros['Precio_Elect_Dia']   = 0.1008
    if num_escenario == 5:
        parametros['reduccion_anual_precio_bateria'] = 0.08
    if num_escenario == 6:
        parametros['reduccion_anual_precio_bateria'] = 0.16
    if num_escenario == 7: # infactible
        parametros['presupuesto'] = [True,12000000]
    if num_escenario == 8:
        parametros['presupuesto'] = [True,16000000]
    if num_escenario == 9:
        parametros['fin_horizonte'] = True
    if num_escenario == 10:
        parametros['Tasa_redu_emisiones'] = 1/3
        parametros['Periodos'] = 3
    if num_escenario == 11:
        parametros['Tasa_redu_emisiones'] = 0.1
        parametros['Periodos'] = 10
    '''
    # experimentos adicionales correción de revisor
    if num_escenario == 12:
        parametros['Tasa_redu_emisiones'] = 0.05
        parametros['Periodos'] = 20
    if num_escenario == 13:
        parametros['Tasa_redu_emisiones'] = 0.0333
        parametros['Periodos'] = 30
    '''
    return parametros
        
def Transicion_flota_modelo(L,CT,T,K,J_K,J_K_tilde,K_E,K_R,C,alpha,R,beta,f,s,kappa,mc,p,F,g,p_tilde,num_per,o,o_inv,o_techo,q,h,t1,a,e,e_tilde,t_F,Gamma,theta,theta_tilde,B,Theta,var_epsilon,c_tilde,efic_bus,DA,epsilon,rho_N,sol_ini,params:dict):
  
  mo = Model()
  mo.Params.OutputFlag = 1
  # Variables
  u = mo.addVars(CT,T,name='u',vtype='I', lb= 0)
  v = mo.addVars(CT,T,name='v',vtype='I', lb= 0)
  x = mo.addVars( K,T,name='x',vtype='I', lb= 0)
  b = mo.addVars(T   ,name='b',vtype='C', lb= 0)
  w_diesel = {}
  w_energia = {}

  u_tilde = mo.addVars(T,name='u',vtype='I', lb= 0)
  v_tilde = mo.addVars(T,name='v',vtype='I', lb= 0)

  if params['contenedores'] == False:
    u_tilde.ub = 0
    v_tilde.ub = 0

  y = {}; [y.update(mo.addVars([k],T,[j]  , name= 'y',vtype= 'I', lb= 0)) for k in K for j in J_K[k] if j != 0];
  z = {}; [z.update(mo.addVars([k],T,[j-1], name= 'z',vtype= 'I', lb= 0)) for k in K for j in J_K[k] if j != 0];
  w = {}; [w.update(mo.addVars([k],R,T,[j], name= 'w',vtype= 'I', lb =0)) for k in K for j in J_K_tilde[k]    ];
  mo.update()
  vars   = ['u','v','x','b','u_tilde','v_tilde','y','z','w']
  
  if sol_ini != {}:
    for i in u.keys():
      if i in sol_ini['u'].keys():
        u[i].lb = int(round(sol_ini['u'][i],0))
        u[i].ub = int(round(sol_ini['u'][i],0))
      
    for i in v.keys():
      if i in sol_ini['v'].keys():
        v[i].lb = int(round(sol_ini['v'][i],0))
        v[i].ub = int(round(sol_ini['v'][i],0))
      
    for i in x.keys():
      if i in sol_ini['x'].keys():
        x[i].lb = int(round(sol_ini['x'][i],0))
        x[i].ub = int(round(sol_ini['x'][i],0))
      
    #for i in b.keys():
    #  if i in sol_ini['b'].keys():
    #    b[i].lb = sol_ini['b'][i]
    #    b[i].ub = sol_ini['b'][i]
      
    for i in u_tilde.keys():
      if i in sol_ini['u_tilde'].keys():
        u_tilde[i].lb = int(round(sol_ini['u_tilde'][i],0))
        u_tilde[i].ub = int(round(sol_ini['u_tilde'][i],0))
      
    for i in v_tilde.keys():
      if i in sol_ini['v_tilde'].keys():
        v_tilde[i].lb = int(round(sol_ini['v_tilde'][i],0))
        v_tilde[i].ub = int(round(sol_ini['v_tilde'][i],0))
      
    for i in y.keys():
      if i in sol_ini['y'].keys():
        y[i].lb = int(round(sol_ini['y'][i],0))
        y[i].ub = int(round(sol_ini['y'][i],0))
      
    for i in z.keys():
      if i in sol_ini['z'].keys():
        z[i].lb = int(round(sol_ini['z'][i],0))
        z[i].ub = int(round(sol_ini['z'][i],0))
      
    for i in w.keys():
      if i in sol_ini['w'].keys():
        w[i].lb = int(round(sol_ini['w'][i],0))
        w[i].ub = int(round(sol_ini['w'][i],0))
      
  
  # Funcion objetivo
  if params['sensi_combus'][0]:
    if params['sensi_combus'][1]:
      w_diesel = mo.addVars(T,name='w_diesel',vtype='C', lb= 0)
      w_energia = mo.addVars(T,name='w_energia',vtype='C', lb= 0)
    else:
      w_diesel = mo.addVar(name='w_diesel',vtype='C', lb= 0)
      w_energia = mo.addVar(name='w_energia',vtype='C', lb= 0)
    if params['fin_horizonte']: 
      if params['sensi_combus'][1]:
        mo.addConstrs( ( w_diesel[t] == 
        quicksum( 
                (beta**( (t_+1) -1) * (1/efic_bus[k]) * DA[r]) * w[k,r,t,j]
                 
                for k in K if k not in K_E 
                for r in R 
                for j in range(0, (kappa[k]) +1) 
                )
        + 
        quicksum( 
                (beta**( num_per + i - 1 ) * (1/efic_bus[k]) * DA[r]) * w[k,r,t_F,j] * (1 if t == t_F else 0)
                for k in K if k not in K_E 
                for r in R 
                for j in range(0, (kappa[k]-1) +1)
                for i in range(1, (kappa[k]-j) +1)
                )
        for t_,t in enumerate(T) ), name='R-diesel_');
        
        mo.addConstrs( (w_energia[t] == 
        quicksum( 
                beta**( (t_+1) -1) * DA[r] *  
                ( 
                  quicksum( (1/efic_bus[k]) * w[k,r,t,j] for k in K_E if k not in K_R for j in range(0, (kappa[k]) +1) ) +
                  quicksum( (1/efic_bus[k]) * (1 + epsilon - (rho_N*epsilon) ) * w[k,r,t,j] for k in K_R for j in range(0, (kappa[k]) +1) )
                )
                for r in R
                )
        +
        quicksum( DA[r] *  
                ( 
                  quicksum( beta**( num_per + i - 1 ) * (1/efic_bus[k]) * w[k,r,t_F,j] * (1 if t == t_F else 0)
                          for k in K_E if k not in K_R 
                          for j in range(0, (kappa[k]-1) +1)
                          for i in range(1, (kappa[k]-j) +1) )
                  +
                  quicksum( beta**( num_per + i - 1 ) * (1/efic_bus[k]) * (1 + epsilon - (rho_N*epsilon) ) * w[k,r,t_F,j] * (1 if t == t_F else 0)
                          for k in K_R 
                          for j in range(0, (kappa[k]-1) +1)
                          for i in range(1, (kappa[k]-j) +1) )
                )
                for r in R
                )
        for t_,t in enumerate(T) ), name='R-energia_')
        
        mo.setObjective(
            quicksum( beta**( (t_+1) -1) * quicksum( f[k,t] * x[k,t] 
                                          - quicksum( s[k,j] * y[k,t,j]  for j in range(1, (kappa[k]+1) +1))
                                          + quicksum( o_inv[k,r,t,j] * w[k,r,t,j] for r in R for j in range(0, (kappa[k]) +1))
                                          + mc[k,t] * z[k,t,alpha[k]]
                                        for k in K)  
                    for t_,t in enumerate(T) )
          
            +quicksum( beta**( (t_+1) -1) * quicksum( p[c] * u[c,t] + F * g[c] * v[c,t] for c in CT) for t_,t in enumerate(T) )
            +quicksum( beta**( (t_+1) -1) * p_tilde * u_tilde[t] for t_,t in enumerate(T) )
            
            +quicksum( beta**( num_per + i - 1 ) * o_inv[k,r,t_F,j+i] * w[k,r,t_F,j]
                      for k in K for r in R for j in range(0, (kappa[k]-1) +1) for i in range(1, (kappa[k]-j) +1) ) 
            - quicksum( beta**( num_per + kappa[k] - j ) * s[k,kappa[k]+1] * z[k,t_F,j] for k in K for j in range(0, (kappa[k]) +1) ) 
            + quicksum( beta**( num_per + alpha[k] - j - 1 ) * mc[k,t_F] * z[k,t_F,j] for k in K for j in range(0, (alpha[k]-1) +1 ) )
            + quicksum( params['Precio_diesel']*w_diesel[t] + params['Precio_Elect_Noche']* w_energia[t] for t in T )
            , GRB.MINIMIZE)
      else:
        mo.addConstr( w_diesel == 
        quicksum( 
                (beta**( (t_+1) -1) * (1/efic_bus[k]) * DA[r]) * w[k,r,t,j]
                for t_,t in enumerate(T) 
                for k in K if k not in K_E 
                for r in R 
                for j in range(0, (kappa[k]) +1) 
                )
        + 
        quicksum( 
                (beta**( num_per + i - 1 ) * (1/efic_bus[k]) * DA[r]) * w[k,r,t_F,j]
                for k in K if k not in K_E 
                for r in R 
                for j in range(0, (kappa[k]-1) +1)
                for i in range(1, (kappa[k]-j) +1)
                ), name='R-diesel_');
        
        mo.addConstr( w_energia == 
        quicksum( 
                beta**( (t_+1) -1) * DA[r] *  
                ( 
                  quicksum( (1/efic_bus[k]) * w[k,r,t,j] for k in K_E if k not in K_R for j in range(0, (kappa[k]) +1) ) +
                  quicksum( (1/efic_bus[k]) * (1 + epsilon - (rho_N*epsilon) ) * w[k,r,t,j] for k in K_R for j in range(0, (kappa[k]) +1) )
                )
                for t_,t in enumerate(T) 
                for r in R
                )
        +
        quicksum( DA[r] *  
                ( 
                  quicksum( beta**( num_per + i - 1 ) * (1/efic_bus[k]) * w[k,r,t_F,j] 
                          for k in K_E if k not in K_R 
                          for j in range(0, (kappa[k]-1) +1)
                          for i in range(1, (kappa[k]-j) +1) )
                  +
                  quicksum( beta**( num_per + i - 1 ) * (1/efic_bus[k]) * (1 + epsilon - (rho_N*epsilon) ) * w[k,r,t_F,j] 
                          for k in K_R 
                          for j in range(0, (kappa[k]-1) +1)
                          for i in range(1, (kappa[k]-j) +1) )
                )
                for r in R
                ), name='R-energia_')
        
        mo.setObjective(
            quicksum( beta**( (t_+1) -1) * quicksum( f[k,t] * x[k,t] 
                                          - quicksum( s[k,j] * y[k,t,j]  for j in range(1, (kappa[k]+1) +1))
                                          + quicksum( o_inv[k,r,t,j] * w[k,r,t,j] for r in R for j in range(0, (kappa[k]) +1))
                                          + mc[k,t] * z[k,t,alpha[k]]
                                        for k in K)  
                    for t_,t in enumerate(T) )
          
            +quicksum( beta**( (t_+1) -1) * quicksum( p[c] * u[c,t] + F * g[c] * v[c,t] for c in CT) for t_,t in enumerate(T) )
            +quicksum( beta**( (t_+1) -1) * p_tilde * u_tilde[t] for t_,t in enumerate(T) )
            
            +quicksum( beta**( num_per + i - 1 ) * o_inv[k,r,t_F,j+i] * w[k,r,t_F,j]
                      for k in K for r in R for j in range(0, (kappa[k]-1) +1) for i in range(1, (kappa[k]-j) +1) ) 
            - quicksum( beta**( num_per + kappa[k] - j ) * s[k,kappa[k]+1] * z[k,t_F,j] for k in K for j in range(0, (kappa[k]) +1) ) 
            + quicksum( beta**( num_per + alpha[k] - j - 1 ) * mc[k,t_F] * z[k,t_F,j] for k in K for j in range(0, (alpha[k]-1) +1 ) )
            + params['Precio_diesel']*w_diesel + params['Precio_Elect_Noche']* w_energia
            , GRB.MINIMIZE)
    else:
      print('No se implementa (aún) la sensibilidad de combustible en el caso de no comteplar el fin de horizonte')
      mo.setObjective(
          quicksum( beta**( (t_+1) -1) * quicksum( f[k,t] * x[k,t] 
                                        - quicksum( s[k,j] * y[k,t,j]  for j in range(1, (kappa[k]+1) +1))
                                        + quicksum( o[k,r,t,j] * w[k,r,t,j] for r in R for j in range(0, (kappa[k]) +1))
                                        + mc[k,t] * z[k,t,alpha[k]]
                                      for k in K)  
                  for t_,t in enumerate(T) )
        
          +quicksum( beta**( (t_+1) -1) * quicksum( p[c] * u[c,t] + F * g[c] * v[c,t] for c in CT) for t_,t in enumerate(T) )
          +quicksum( beta**( (t_+1) -1) * p_tilde * u_tilde[t] for t_,t in enumerate(T) )
          
          , GRB.MINIMIZE)
    
  else:
    if params['fin_horizonte']: 
      mo.setObjective(
          quicksum( beta**( (t_+1) -1) * quicksum( f[k,t] * x[k,t] 
                                        - quicksum( s[k,j] * y[k,t,j]  for j in range(1, (kappa[k]+1) +1))
                                        + quicksum( o[k,r,t,j] * w[k,r,t,j] for r in R for j in range(0, (kappa[k]) +1))
                                        + mc[k,t] * z[k,t,alpha[k]]
                                      for k in K)  
                  for t_,t in enumerate(T) )
        
          +quicksum( beta**( (t_+1) -1) * quicksum( p[c] * u[c,t] + F * g[c] * v[c,t] for c in CT) for t_,t in enumerate(T) )
          +quicksum( beta**( (t_+1) -1) * p_tilde * u_tilde[t] for t_,t in enumerate(T) )
          
          +quicksum( beta**( num_per + i - 1 ) * o[k,r,t_F,j+i] * w[k,r,t_F,j]
                    for k in K for r in R for j in range(0, (kappa[k]-1) +1) for i in range(1, (kappa[k]-j) +1) ) 
          - quicksum( beta**( num_per + kappa[k] - j ) * s[k,kappa[k]+1] * z[k,t_F,j] for k in K for j in range(0, (kappa[k]) +1) ) 
          + quicksum( beta**( num_per + alpha[k] - j - 1 ) * mc[k,t_F] * z[k,t_F,j] for k in K for j in range(0, (alpha[k]-1) +1 ) )
          , GRB.MINIMIZE)
    else:
        mo.setObjective(
          quicksum( beta**( (t_+1) -1) * quicksum( f[k,t] * x[k,t] 
                                        - quicksum( s[k,j] * y[k,t,j]  for j in range(1, (kappa[k]+1) +1))
                                        + quicksum( o[k,r,t,j] * w[k,r,t,j] for r in R for j in range(0, (kappa[k]) +1))
                                        + mc[k,t] * z[k,t,alpha[k]]
                                      for k in K)  
                  for t_,t in enumerate(T) )
        
          +quicksum( beta**( (t_+1) -1) * quicksum( p[c] * u[c,t] + F * g[c] * v[c,t] for c in CT) for t_,t in enumerate(T) )
          +quicksum( beta**( (t_+1) -1) * p_tilde * u_tilde[t] for t_,t in enumerate(T) )
          
          , GRB.MINIMIZE)

  # Restricciones

  mo.addConstrs( ( quicksum( w[k,r,t,j]  for k in K for j in range(0,kappa[k] + 1)) == q[r]  for r in R for t in T), name='R-2_');

  mo.addConstrs( ( w[k,r,t,j] <= h[k,r,t,j] * z[k,t,j] for k in K for r in R for t in T for j in J_K_tilde[k] ), name='R-3_');

  mo.addConstrs( ( quicksum( w[k,r,t,j]  for r in R) == z[k,t,j] for k in K for t in T for j in J_K_tilde[k] ), name='R-4_');

  mo.addConstrs( ( x[k,t] == z[k,t,0] for k in K for t in T if t != t1), name='R-5_');

  mo.addConstrs( ( x[k,t1] + a[k,0] == z[k,t1,0] for k in K ), name='R-6_');

  mo.addConstrs( ( z[k,t,j] == z[k,t-1,j-1] - y[k,t,j] for k in K for t in T if t != t1 for j in J_K_tilde[k] if j != 0 ), name='R-7_');

  mo.addConstrs( ( z[k,t1,j] == a[k,j] - y[k,t1,j] for k in K for j in J_K_tilde[k] if j != 0), name='R-8_');

  mo.addConstrs( ( y[k,t,(kappa[k]+1)] == z[k,t-1,kappa[k]] for k in K for t in T if t != t1), name='R-9_');

  mo.addConstrs( ( y[k,t1,(kappa[k]+1)] == a[k,kappa[k]+1] for k in K ), name='R-10_');

  mo.addConstrs( ( v[c,t1] == e[c] + u[c,t1] for c in CT ), name='R-11_');

  mo.addConstrs( (  v[c,t] == v[c,t-1] + u[c,t] for t in T if t != t1 for c in CT ), name='R-12_');

  if params['edad_media']:
    mo.addConstr(  quicksum( j * z[k,t_F,j] for k in K for j in range(0,(kappa[k]) +1)) <= 
                      Gamma * quicksum( z[k,t_F,j] for k in K for j in range(0,(kappa[k]) +1)) , name='R-13_');

  mo.addConstrs( ( v[c,t] + v[c_tilde,t] >= theta[c] * quicksum( z[k,t,j] for k in K_E if c in C[k] for j in range(0, (kappa[k]) +1)) 
                  for c in CT if c != c_tilde for t in T ), name='R-14_');

  mo.addConstrs( ( quicksum( f[k,t] * x[k,t] - quicksum( s[k,j] * y[k,t,j] for j in range(1, (kappa[k]+1) + 1)) for k in K)
                + quicksum( p[c] * u[c,t] for c in CT)
                + p_tilde * u_tilde[t] == b[t]   for t in T ), name='R-15_');

  mo.addConstr(    v_tilde[t1] == e_tilde + u_tilde[t1] , name='R-16_');

  mo.addConstrs( ( v_tilde[t] == v_tilde[t-1] + u_tilde[t] for t in T if t != t1 ), name='R-17');

  mo.addConstrs( ( v[c_tilde,t] <= theta_tilde * v_tilde[t] for t in T ), name='R-18');

  if params['presupuesto'][0]:
    mo.addConstr( quicksum( b[t] for t in T ) <= B, name='R-19_');

  if params['emisiones']:
    mo.addConstrs( ( quicksum( var_epsilon[(k,j,r,l)] * w[k,r,t,j]  for k in K for j in range(0,kappa[k] + 1) for r in R) <= Theta[(l,t)] 
                  for t in T for l in L ), name='R-20');

  # mo.addConstrs( ( quicksum( g[c] * v[c,t] for c in DC) <= G for t in T if np.char.isnumeric(G) ), name='R-14_');

  # mo.addConstrs( ( quicksum( v[c,t] for c in DC) <= H for t in T if np.char.isnumeric(H) ), name='R-15_');

  mo.Params.MIPGap = 0
  mo.update()
  mo.optimize()
  if mo.Status == 3:
    print('Infactible')
    mo.computeIIS()
    mo.write('infactible.ilp')
  
  vars_g = [u,v,x,b,u_tilde,v_tilde,y,z,w]
  sol_salida = {vars[idx]: {i: j.x for i,j in var.items()} for idx,var in enumerate(vars_g)}
  
  #print(mo.Status)
  #mo.computeIIS()
  #mo.write('infactible.ilp')
  
  return mo,x,y,z,w,u,v,b,u_tilde,v_tilde,w_diesel,w_energia,sol_salida