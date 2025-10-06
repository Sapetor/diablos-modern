"""
functions.py - Contains all the functions associated with the blocks in the simulation
"""

import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp

import logging

logger = logging.getLogger(__name__)

def integrator(time, inputs, params, output_only=False, next_add_in_memory=True, dtime=0.01):
    # Initialization (this step happens only in the first iteration)
    if params.get('_init_start_', True):
        params['dtime'] = dtime
        params['mem'] = np.atleast_1d(np.array(params['init_conds'], dtype=float))
        params['output'] = np.atleast_1d(np.array(params['init_conds'], dtype=float))
        params['mem_list'] = [np.zeros_like(params['mem'])]
        params['mem_len'] = 5.0
        params['_init_start_'] = False
        params['aux'] = np.zeros_like(params['mem'])

        if params['method'] == 'RK45':
            params['nb_loop'] = 0
            params['RK45_Klist'] = [0, 0, 0, 0]

        params['add_in_memory'] = True

    if output_only:
        # The output now returns the value calculated in the previous step,
        # which is stored in 'output'. This fixes the one-step delay.
        result = {0: params.get('output', params['mem'])}
        return result
    
    # Checks if the new input vector dimensions match.
    if isinstance(inputs.get(0), (float, int)):
        inputs[0] = np.atleast_1d(inputs[0])
    
    if params['mem'].shape != inputs.get(0, params['mem']).shape:
        if params['mem'].size == 1:
            logger.warning(f"Expanding initial conditions for {params['_name_']} to match input dimensions.")
            params['mem'] = np.full(inputs[0].shape, params['mem'].item())
        else:
            logger.error(f"Dimension Error in initial conditions in {params['_name_']}")
            params['_init_start_'] = True
            return {'E': True, 'error': f"Dimension mismatch in {params['_name_']}"}

    # Integration process according to chosen method
    if params['method'] == 'FWD_EULER':
        if params['add_in_memory']:
            params['mem'] += params['dtime'] * inputs[0]
        else:
            params['aux'] = np.array(params['mem'] + 0.5 * params['dtime'] * inputs[0])
            return {0: params['aux']}
    elif params['method'] == 'BWD_EULER':
        if params['add_in_memory']:
            params['mem'] += params['dtime'] * params['mem_list'][-1]
        else:
            params['aux'] = np.array(params['mem'] + 0.5 * params['dtime'] * params['mem_list'][-1])
            return {0: params['aux']}
    elif params['method'] == 'TUSTIN':
        if params['add_in_memory']:
            params['mem'] += 0.5*params['dtime'] * (inputs[0] + params['mem_list'][-1])
        else:
            params['aux'] = np.array(params['mem'] + 0.25 * params['dtime'] * (inputs[0] + params['mem_list'][-1]))
            return {0: params['aux']}
    elif params['method'] == 'RK45':
        K_list = params['RK45_Klist']
        K_list[params['nb_loop']] = params['dtime'] * np.array(inputs[0], dtype=float)
        params['RK45_Klist'] = K_list
        K1, K2, K3, K4 = K_list

        if params['nb_loop'] == 0:
            params['nb_loop'] += 1
            params['aux'] = params['mem'] + 0.5 * K1
            return {}
        elif params['nb_loop'] == 1:
            params['nb_loop'] += 1
            params['aux'] = params['mem'] + 0.5 * K2
            return {}
        elif params['nb_loop'] == 2:
            params['nb_loop'] += 1
            params['aux'] = params['mem'] + K3
            return {}
        elif params['nb_loop'] == 3:
            params['nb_loop'] = 0
            params['mem'] += (1 / 6) * (K1 + 2 * K2 + 2 * K3 + K4)
    elif params['method'] == 'SOLVE_IVP':
        def fun(t, y):
            return inputs[0]
        sol = solve_ivp(fun, [time, time + dtime], params['mem'])
        params['mem'] = sol.y[:, -1]
        return {0: params['mem']}
    else:
        logger.error(f"Unknown integration method {params['method']} in {params['_name_']}")
        return {'E': True}

    aux_list = params['mem_list']
    aux_list.append(inputs[0])
    if len(aux_list) > params['mem_len']:
        aux_list = aux_list[-5:]
    params['mem_list'] = aux_list

    result = {0: params['mem']} if params['add_in_memory'] else {}
    return result

def transfer_function(time, inputs, params, output_only=False):
    """
    Transfer function block
    """
    if params.get('_init_start_', True):
        params['_init_start_'] = False
        num = np.array(params['numerator'])
        den = np.array(params['denominator'])
        
        # Convert to state-space
        A, B, C, D = signal.tf2ss(num, den)
        
        # Discretize
        dtime = params['dtime']
        Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), dtime)
        
        params['_Ad_'] = Ad
        params['_Bd_'] = Bd
        params['_Cd_'] = Cd
        params['_Dd_'] = Dd
        
        # State vector initialization
        num_states = Ad.shape[0]
        init_conds = np.atleast_1d(np.array(params.get('init_conds', 0.0), dtype=float))

        if len(init_conds) < num_states:
            # Pad with zeros
            padded_conds = np.zeros(num_states)
            padded_conds[:len(init_conds)] = init_conds
            init_conds = padded_conds
        elif len(init_conds) > num_states:
            # Truncate
            init_conds = init_conds[:num_states]

        params['_x_'] = init_conds.reshape(-1, 1)

    # Get discrete-time system matrices and state
    Ad = params['_Ad_']
    Bd = params['_Bd_']
    Cd = params['_Cd_']
    Dd = params['_Dd_']
    x = params['_x_']
    
    # Get input
    u = 0.0
    if not output_only:
        u = inputs.get(0, 0.0)

    # Compute output
    try:
        y = Cd @ x + Dd * u
    except ValueError as e:
        logger.error(f"Error in transfer function matrix multiplication: {e}")
        logger.error(f"Shapes: Cd: {Cd.shape}, x: {x.shape}, Dd: {Dd.shape}, u: {np.shape(u)}")
        return {'E': True, 'error': f"Matrix multiplication error in {params['_name_']}: {e}"}

    # Update state only if not in output_only mode
    if not output_only:
        try:
            params['_x_'] = Ad @ x + Bd * u
        except ValueError as e:
            logger.error(f"Error in transfer function state update: {e}")
            logger.error(f"Shapes: Ad: {Ad.shape}, x: {x.shape}, Bd: {Bd.shape}, u: {np.shape(u)}")
            return {'E': True, 'error': f"State update error in {params['_name_']}: {e}"}
    
    return {0: y.item()}

def export(time, inputs, params):
    """
    Block to save and export block signals
    """
    # To prevent saving data in the wrong iterations (integration method RK45 in use)
    if '_skip_' in params.keys() and params['_skip_']:
        params['_skip_'] = False
        return {0: np.array([0.0])}
    # Initialization of the saving vector
    if params['_init_start_']:
        aux_vector = np.array([inputs[0]])
        try:
            params['vec_dim'] = len(inputs[0])
        except:
            params['vec_dim'] = 1

        labels = params['str_name']
        if labels == 'default':
            labels = params['_name_'] + '-0'
        labels = labels.replace(' ', '').split(',')
        if len(labels) < params['vec_dim']:
            for i in range(params['vec_dim'] - len(labels)):
                labels.append(params['_name_'] + '-' + str(params['vec_dim'] + i - 1))
        elif len(labels) > params['vec_dim']:
            labels = labels[:params['vec_dim']]
        if len(labels) == params['vec_dim'] == 1:
            labels = labels[0]
        params['vec_labels'] = labels
        params['_init_start_'] = False
    else:
        aux_vector = params['vector']
        aux_vector = np.concatenate((aux_vector, [inputs[0]]))
    params['vector'] = aux_vector
    return {0: np.array([0.0])}

def scope(time, inputs, params):
    """
    Function to plot block signals
    """
    # To prevent saving data in the wrong iterations (integration method RK45 in use)
    if '_skip_' in params.keys() and params['_skip_']:
        params['_skip_'] = False
        return {0: np.array([0.0])}
    # Initialization of the saving vector
    if params['_init_start_']:
        logger.debug(f"Scope {params.get('_name_', 'unknown')} initializing, inputs: {inputs}")
        aux_vector = np.atleast_1d(inputs[0])
        try:
            params['vec_dim'] = len(inputs[0])
        except:
            params['vec_dim'] = 1

        labels = params['labels']
        if labels == 'default':
            labels = params['_name_'] + '-0'
        labels = labels.replace(' ', '').split(',')
        if len(labels) - params['vec_dim'] >= 0:
            labels = labels[:params['vec_dim']]
        elif len(labels) - params['vec_dim'] < 0:
            for i in range(len(labels), params['vec_dim']):
                labels.append(params['_name_'] + '-' + str(i))
        elif len(labels) == params['vec_dim'] == 1:
            labels = labels[0]
        params['vec_labels'] = labels
        params['_init_start_'] = False
        logger.debug(f"Scope {params.get('_name_', 'unknown')} initialized, vec_labels: {params['vec_labels']}")
    else:
        aux_vector = params['vector']
        aux_vector = np.concatenate((aux_vector, np.atleast_1d(inputs[0])))
    params['vector'] = aux_vector
    return {0: np.array([0.0])}

def bode_plot(time, inputs, params):
    """
    Bode plotter block. This function doesn't do anything during simulation.
    The plotting is handled by a special call from the UI.
    """
    return {0: np.array([0.0])}
