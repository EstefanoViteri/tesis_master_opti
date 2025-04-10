import pandas as pd

def Summary_Box(x,u,z,u_tilde,w,v,f,p,p_tilde,o,F,g,mo,kappa,alpha,beta,num_per,K,CT,T,R,t_F,s,mc,cfh):
    mill = 1000000
    Summary = [ sum(beta**( (t_+1) -1) * f[k,t] * int(round(x[k,t].x,0)) for k in K for t_,t in enumerate(T))  ] # costo buses
    Summary.append( sum(beta**( (t_+1) -1) * p[c] * int(round(u[c,t].x,0)) for c in CT for t_,t in enumerate(T)) ) # costo cargadores
    Summary.append( sum(beta**( (t_+1) -1) * p_tilde * int(round(u_tilde[t].x,0))  for t_,t in enumerate(T)) ) # costo contenedores
    Summary.append(sum(beta**( (t_+1) -1) *  o[k,r,t,j] * int(round(w[k,r,t,j].x,0)) for k in K for t_,t in enumerate(T) for r in R for j in range(0, (kappa[k]) +1))) #costo operacion
    Summary.append(sum(beta**( (t_+1) -1) *  F * g[c] * int(round(v[c,t].x,0)) for c in CT for t_,t in enumerate(T) ) ) # costo cargos demanda
    Summary.append(sum(beta**( (t_+1) -1) *  mc[k,t] * int(round(z[k,t,alpha[k]].x,0)) for k in K for t_,t in enumerate(T)))# costo de media vida
    Summary.append(sum(Summary[i] for i in range(0,6)))
    
    val = sum(  beta**( num_per + i - 1 ) * o[k,r,t_F,j+i] * int(round(w[k,r,t_F,j].x,0))  for k in K for r in R for j in range(0, (kappa[k]-1) +1) for i in range(1, (kappa[k]-j) +1) ) # costos f.h.
    - sum( beta**( num_per + kappa[k] - j ) * s[k,kappa[k]+1] * int(round(z[k,t_F,j].x,0)) for k in K for j in range(0, (kappa[k]) +1) ) 
    + sum( beta**( num_per + alpha[k] - j - 1 ) * mc[k,t_F] * int(round(z[k,t_F,j].x,0)) for k in K for j in range(0, (alpha[k]-1) +1 ) )
    
    Summary.append(val)
    Summary.append(mo.ObjVal)
    Summary = [round(i/mill,2) for i in Summary]
    Summary.append(round(mo.runtime,2))
    Summary.append(mo.NumVars)
    Summary.append(mo.NumConstrs)
    Summary.append(mo.NodeCount)

    Rest = pd.DataFrame(Summary)
    Rest.columns = ["Costos"]
    Rest.index   = ['Costos de buses comprados','Costos de cargadores comprados','Costos de contenedores comprados','Costos operativos',
                    'Costos por demanda','Costos de media vida','Costo total','Costos por fin de horizonte','Funcion Objetivo',
                    'Tiempo de solución', 'Número de variables','Numero de restricciones','Número de nodos explorados B&B']
    return Rest

def aux_total(x:dict):
    for i,j in x.items():
        j.loc['Total'] = j.sum(axis=0)
    
    return x

def tablas_exp(x,z,y,u,v,w,K,T,J_K,CT,R,K_E,J_K_tilde):

    bus_comp_per = pd.DataFrame( [ [k] + [int(round(x[k,t].x,0)) for t in list(T)] for k in K], columns=['Tipo de Bus'] + list(T) ).set_index('Tipo de Bus')
    bus_disp_per = pd.DataFrame( [ [k] + [int(round(sum(z[k,t,j-1].x for j in J_K[k] if j != 0),0)) for t in list(T)] for k in K], columns=['Tipo de Bus'] + list(T) ).set_index('Tipo de Bus')
    bus_vend_per = pd.DataFrame( [ [k] + [int(round(sum(y[k,t,j].x for j in J_K[k] if j != 0),0)) for t in list(T)] for k in K], columns=['Tipo de Bus'] + list(T) ).set_index('Tipo de Bus')
    carg_comp_per = pd.DataFrame( [ [c] + [int(round( u[c,t].x  ,0)) for t in list(T)] for c in CT], columns=['Tipo de Cargador'] + list(T) ).set_index('Tipo de Cargador')
    carg_disp_per = pd.DataFrame( [ [c] + [int(round( v[c,t].x  ,0)) for t in list(T)] for c in CT], columns=['Tipo de Cargador'] + list(T) ).set_index('Tipo de Cargador')
    EB_rutas = pd.DataFrame( [ [r] + [int(round( sum( w[k,r,t,j].x for k in K_E for j in J_K_tilde[k])  ,0)) for t in list(T)] for r in R], columns=['Ruta'] + list(T) ).set_index('Ruta')

    tablas = {'Autobuses comprados por per' : bus_comp_per,
            'Autobuses disponibles por per' : bus_disp_per,
            'Autobuses vendidos por per'    : bus_vend_per,  
            'Cargadores comprados por per'  : carg_comp_per,
            'Cargadores disponibles por per': carg_disp_per,
            'Autobuses electricos por ruta' : EB_rutas }
    
    tablas = aux_total(tablas)

    tablas['Autobuses disponibles por per'].loc['Total autobuses electricos'] = bus_disp_per[bus_disp_per.index.str.contains('_EB')].sum(axis=0)
    tablas['Autobuses electricos por ruta'].loc['Total autobuses electricos y diesel'] = bus_disp_per.loc['Total']
    
    return tablas

def aux_exp(vari,nom):
    return [ ([i] if isinstance(i,int) else list(i)) + [int(round(vari[i].x,0)) if nom != 'b' else vari[i].x] for i in vari.keys()]

def exportar_solucion(x,z,y,u,u_tilde,v,v_tilde,w,b,K,T,J_K,CT,R,f,p,o,F,g,mo,kappa,alpha,beta,num_per,p_tilde,K_E,J_K_tilde,t_F,s,mc,var_epsilon,Theta,Theta_tilde,L,params_exp):
    
    variables = {'Cargadores Comprados'   : ['Tipo de cargador','Periodo','u',u],
                'Cargadores Disponibles'  : ['Tipo de cargador','Periodo','v',v],
                'Autobuses Comprados'     : ['Tipo de autobus' ,'Periodo','x',x],
                'Autobuses Retirados'     : ['Tipo de autobus' ,'Periodo','Edad','y',y],
                'Autobuses Disponibles'   : ['Tipo de autobus' ,'Periodo','Edad','z',z],
                'Autobuses Asignados'     : ['Tipo de autobus' ,'Ruta'   ,'Periodo','Edad','w',w],
                'Contenedores Comprados'  : ['Periodo','u_tilde',u_tilde],
                'Contenedores Disponibles': ['Periodo','v_tilde',v_tilde],
                'Inversion'               : ['Periodo','b',b]
                }

    # Exportar variables
    info_exportar = {i: pd.DataFrame( aux_exp(j[-1],j[-2]), columns= j[:-1]) for i,j in variables.items()}
    # Exportar resumen
    Resumen = Summary_Box(x,u,z,u_tilde,w,v,f,p,p_tilde,o,F,g,mo,kappa,alpha,beta,num_per,K,CT,T,R,t_F,s,mc,params_exp['fin_horizonte'])
    info_exportar['Resumen'] = Resumen
    
    # Exportar tablas
    tablas = tablas_exp(x,z,y,u,v,w,K,T,J_K,CT,R,K_E,J_K_tilde)
    [info_exportar.update({i:j}) for i,j in tablas.items()]
    # Exportar Emisiones de contaminantes
    info_exportar['Emisiones'] = pd.DataFrame([[t,l, sum( var_epsilon[(k,j,r,l)] * int(round(w[k,r,t,j].x,0)) for k in K for j in range(0,kappa[k] + 1) for r in R),Theta[(l,t)],Theta_tilde[l] ] for l in L for t in T],columns=['Periodo','Tipo Contaminante','Emisión Flota','Meta de emisión','Emisión Flota antes de la planificación'])
    '''
    with pd.ExcelWriter('Resultado_Trans_Flota_Escenario_{}.xlsx'.format(escenario), engine='openpyxl') as writer:
        for sheet_name, df in info_exportar.items():
            df.to_excel(writer, sheet_name=sheet_name)
    '''
    # Escenario 0 hace referencia al Caso Base
    with pd.ExcelWriter('Res_Trans_Flota_Esc_{}.xlsx'.format(params_exp['escenario'])) as writer:
        for i,j in info_exportar.items():
            j.to_excel(writer, i)
        
    return info_exportar