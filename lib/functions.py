"""
functions.py - Contains all the functions associated with the blocks in the simulation
"""

import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp

import logging

logger = logging.getLogger(__name__)

class DFunctions:
    """
    Class to contain all the default functions available to work with in the simulation interface
    """

    def step(self, time, inputs, params):
        """
        Step source function

        :purpose: Function that returns a constant value over time.
        :description: This is a source type function, which is piecewise. It can be used to indicate the beginning or end of a branch of a network.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :param params['value']: The value that the function returns. It can be a scalar (float) as well as a vector ([float, ...]).
        :param params['delay']: Indicates a point in time where the piecewise jump occurs.
        :param params['type']: ['up'/'down'/'pulse'/'constant'] Indicates whether the jump is upward ('value'), downward (0), in pulse form or constant ('value').
        :param params['pulse_start_up']: Indicates whether the pulse starts upwards (True) or downwards (False).
        :param params['_init_start_']: Auxiliary parameter used by the system to perform special functions in the first simulation loop.
        :param params['_name_']: Auxiliary parameter delivered by the associated block, for error identification.
        :type time: float
        :type inputs: dict
        :type params['value']: float/numpy.ndarray
        :type params['delay']: float
        :type params['type']: str
        :type params['pulse_start_up']: bool
        :type params['_init_start_']: bool
        :type params['_name_']: str
        :return: The value defined in 'value' or 0.
        :rtype: numpy.ndarray
        :examples: See example in :ref:`examples:vectorial integration`, :ref:`examples:gaussian noise` and :ref:`examples:signal products`.

        """
        if params['_init_start_']:
            params['step_old'] = time
            params['change_old'] = not params['pulse_start_up']
            params['_init_start_'] = False
        
        # Convert delay to float
        delay = float(params['delay'])
        
        if params['type'] == 'up':
            change = True if time < delay else False
        elif params['type'] == 'down':
            change = True if time > delay else False
        elif params['type'] == 'pulse':
            if time - params['step_old'] >= delay:
                params['step_old'] += delay
                change = not params['change_old']
            else:
                change = params['change_old']
        elif params['type'] == 'constant':
            change = False
        else:
            print("ERROR: 'type' not correctly defined in", params['_name_'])
            return {'E': True}

        if change:
            params['change_old'] = True
            return {0: np.atleast_1d(np.zeros_like(np.array(params['value'], dtype=float)))}
        else:
            params['change_old'] = False
            return {0: np.atleast_1d(np.array(params['value'], dtype=float))}


    def ramp(self, time, inputs, params):
        """
        Ramp source function

        :purpose: Function that returns a value that changes linearly over time.
        :description: This is a source type function, which is piecewise. The value changes linearly over time, and can increase or decrease.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :param params['slope']: The value of the slope that the ramp has.
        :param params['delay']: Indicates a point in time where the start of the ramp happens.
        :type time: float
        :type inputs: dict
        :type params['slope']: float
        :type params['delay']: float
        :return: The value of the slope multiplied by the difference between 'time' and 'delay'.
        :rtype: numpy.ndarray
        :examples: See example in :ref:`examples:external derivator (z-process)`.

        """
        slope = float(params['slope'])
        delay = float(params['delay'])
        if slope == 0:
            return {0: np.array(0, dtype=float)}
        elif slope > 0:
            return {0: np.maximum(0, slope * (time - delay))}
        elif slope < 0:
            return {0: np.minimum(0, slope * (time - delay))}


    def sine(self, time, inputs, params):
        """
        Sinusoidal source function

        :purpose: Function that returns a sinusoidal in time.
        :description: This is a source type function. It returns a sinusoidal with variation in the parameters of amplitude, frequency and initial angle.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :param params['amplitude']: Amplitude value taken by the sinusoidal.
        :param params['omega']: Value in rad/s (2*pi*f) of the frequency taken by the sinusoidal.
        :param params['init_angle']: Value in radians of the angle taken by the sinusoidal at time zero.
        :type time: float
        :type inputs: dict
        :type params['amplitude']: float
        :type params['omega']: float
        :type params['init_angle']: float
        :return: A sinusoidal of amplitude 'amplitude', frequency 'omega' and initial angle 'init_angle'.
        :rtype: numpy.ndarray
        :examples: See example in :ref:`examples:sine integration`.

        """
        amplitude = float(params['amplitude'])
        omega = float(params['omega'])
        init_angle = float(params['init_angle'])
        return {0: np.array(amplitude * np.sin(omega * time + init_angle), dtype=float)}


    def noise(self, time, inputs, params):
        """
        Gaussian noise function

        :purpose: Function returns a normal random noise.
        :description: This is a source type function. It produces a gaussian random value of mean mu and variance sigma**2.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :param params['mu']: Mean value of the noise.
        :param params['sigma']: Standard deviation value of the noise.
        :type time: float
        :type inputs: dict
        :type params['sigma']: float
        :type params['mu']: float
        :return: Gaussian random value of mean mu and variance sigma**2.
        :rtype: numpy.ndarray
        :examples: See example in :ref:`examples:gaussian noise`.

        """
        sigma = float(params['sigma'])
        mu = float(params['mu'])
        return {0: np.array(sigma ** 2 * np.random.randn() + mu, dtype=float)}



    def gain(self, time, inputs, params):
        """
        Gain function

        :purpose: Function that scales an input by a factor.
        :description: This is a process type function. It returns the same input, but scaled by a user-defined factor. This input can be either scalar or vector, as well as the scaling factor.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :param params['gain']: Scaling value of the input. Can be a scalar value, or a matrix (only for vector multiplication).
        :type time: float
        :type inputs: dict
        :type params['gain']: float/numpy.ndarray
        :return: The input value, scaled by the 'gain' factor.
        :rtype: numpy.ndarray
        :examples: See example in :ref:`examples:gaussian noise` and :ref:`examples:convergent ode system`.

        """
        try:
            input_value = np.array(inputs[0], dtype=float)
            gain_value = np.array(params['gain'], dtype=float)
            return {0: np.dot(gain_value, input_value)}
        except (ValueError, TypeError):
            print(f"ERROR: Invalid input or gain type in gain block. Expected numeric.")
            return {'E': True}


    def exponential(self, time, inputs, params):
        """
        Exponential function

        :purpose: Function that returns the value of an exponential from an input.
        :description: This is a process type function. It takes the input value, and calculates the exponential of it, with scaling factors for the base as well as the exponent.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :param params['a']: Scaling factor for the base of the exponential.
        :param params['b']: Scaling factor for the exponent of the exponential.
        :type time: float
        :type inputs: dict
        :type params['a']: float
        :type params['b']: float
        :return: The exponential of the input value.
        :rtype: numpy.ndarray

        """
        try:
            input_value = np.array(inputs[0], dtype=float)
            a = float(params['a'])
            b = float(params['b'])
            return {0: np.array(a * np.exp(b * input_value), dtype=float)}
        except (ValueError, TypeError):
            print(f"ERROR: Invalid input or parameter type in exponential block. Expected numeric.")
            return {'E': True}


    def adder(self, time, inputs, params):
        """
        Adder function

        :purpose: Function that returns the addition of two or more inputs.
        :description: This is a process type function. It takes each input value and associates it with a sign (positive or negative), and then adds or subtracts them in an auxiliary variable. The function supports both scalar and vector operations.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :param params['sign']: String that contains all the signs associated to each input value (or vector). It should be noted that in case of having less symbols than vectors, the function will assume that the remaining symbols will add up.
        :param params['_name_']: Auxiliary parameter delivered by the associated block, for error identification.
        :type time: float
        :type inputs: dict
        :type params['sign']: str
        :type params['_name_']: str
        :return: The sum of all inputs.
        :rtype: numpy.ndarray
        :examples: See example in :ref:`examples:gaussian noise`, :ref:`examples:export data` and :ref:`examples:convergent ode system`.
        :notes: This function returns 'Error' if the dimensions of any of the entries are not equal.

        """
        logger.debug(f"Adder input: time={time}, inputs={inputs}, params={params}")
        try:
            suma = 0.0
            for i in sorted(inputs.keys()):
                sign = '+' # Default sign
                if i < len(params['sign']):
                    sign = params['sign'][i]
                
                input_value = np.atleast_1d(inputs[i])

                if sign == '+':
                    suma += input_value
                elif sign == '-':
                    suma -= input_value
            
            logger.debug(f"Adder output: {suma}")
            return {0: suma}
        except (ValueError, TypeError) as e:
            logger.error(f"ERROR: Invalid input type in adder block. Expected numeric. Error: {str(e)}")
            return {'E': True}


    def sigproduct(self, time, inputs, params):
        """
        Element-wise product between signals

        :purpose: Function that returns the multiplication by elements of two or more inputs.
        :description: This is a process type function. It takes each input value and multiplies it with a base value (or vector).
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :type time: float
        :type inputs: dict
        :return: The multiplication of all inputs.
        :rtype: numpy.ndarray
        :examples: See example in :ref:`examples:signal products`.
        :notes: Unlike the sumator function, this one does not check if the inputs have the same dimensions, since there may be occasions where the result needed may be something larger.
        :limitations: The function does not check that the result has the desired dimensions, so it is a job to be done by the user.

        """
        try:
            mult = np.array(1.0, dtype=float)
            for input_value in inputs.values():
                mult *= np.array(input_value, dtype=float)
            return {0: mult}
        except (ValueError, TypeError):
            print(f"ERROR: Invalid input type in sigproduct block. Expected numeric.")
            return {'E': True}


    def mux(self, time, inputs, params):
        """
        Multiplexer function

        :purpose: Function that concatenates several values or vectors.
        :description: This is a process type function. It concatenates each of its entries in such a way as to obtain a vector equal to the sum product of the number of entries by the number of dimensions of each one. The order of the values is given by the order of the block entries.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :type time: float
        :type inputs: dict
        :return: The vector with all values sorted in a single dimension ((a,1) with a>=1).:rtype: numpy.ndarray
        :examples: See example in :ref:`examples:signal products`, :ref:`examples:export data` and :ref:`examples:convergent ode system`.

        """
        try:
            return {0: np.concatenate([np.array(input_value, dtype=float).flatten() for input_value in inputs.values()])}
        except (ValueError, TypeError):
            print(f"ERROR: Invalid input type in mux block. Expected numeric.")
            return {'E': True}


    def demux(self, time, inputs, params):
        """
        Demultiplexer function

        :purpose: Function that splits an input vector into two or more.
        :description: This is a process type function. It takes the input vector and splits it into several smaller equal vectors, depending on the number of outputs.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :param params['output_shape']: Value defining the number of dimensions with which each output will have.
        :param params['_name_']: Auxiliary parameter delivered by the associated block, for error identification.
        :param params['_outputs_']: Auxiliary parameter delivered by the associated block, for identification of available outputs.
        :type time: float
        :type inputs: dict
        :type params['output_shape']: float
        :type params['_name_']: str
        :type params['_outputs_']: float
        :return: A given number of outputs, with each output having equal dimensions.
        :rtype: numpy.ndarray
        :examples: See example in :ref:`examples:gaussian noise`.
        :notes: This function returns 'Error' if the number of values in the input vector is not enough to get all the outputs at the required dimensions. It also returns a 'Warning' if the vector is larger than required, truncating the values that are not taken.

        """
        # Check input dimensions first
        if len(inputs[0]) / params['output_shape'] < params['_outputs_']:
            print("ERROR: Not enough inputs or wrong output shape in", params['_name_'])
            return {'E': True}

        elif len(inputs[0]) / params['output_shape'] > params['_outputs_']:
            print("WARNING: There are more elements in vector for the expected outputs. System will truncate. Block", params['_name_'])

        try:
            input_array = np.array(inputs[0], dtype=float).flatten()
            output_shape = int(params['output_shape'])
            outputs = {}
            for i in range(params['_outputs_']):
                start = i * output_shape
                end = start + output_shape
                outputs[i] = input_array[start:end]
            return outputs
        except (ValueError, TypeError):
            print(f"ERROR: Invalid input type in demux block. Expected numeric.")
            return {'E': True}


    def integrator(self, time, inputs, params, output_only=False, next_add_in_memory=True, dtime=0.01):
        """
        Integrator function

        :purpose: Function that integrates the input signal.
        :description: This is a process type function. It takes the input signal and adds it to an internal variable, weighted by the sampling time. It allows 4 forms of integration, the most complex being the Runge Kutta 45 method.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :param params['init_conds']: Value that contains the initial conditions for the integrator.
        :param params['method']: ['FWD_EULER/BWD_EULER/TUSTIN/RK45/SOLVE_IVP'] String that contains the method of integration to use.
        :param params['dtime']: Auxiliary variable that contains the sampling time that the simulation is using (fixed step integration).
        :param params['mem']: Variable containing the sum of all data, from start to lapse 'time'.
        :param params['mem_list']: Vector containing the last values of 'mem'.
        :param params['mem_len']: Variable defining the number of elements contained in 'mem_list'.
        :param params['nb_loop']: Auxiliary variable indicating the current step of the RK45 method.
        :param params['RK45_Klist']: Auxiliary vector containing the last values of K1,K2,K3,K4 (RK45 method).
        :param params['add_in_memory']: Auxiliary variable indicating when the input value is added to 'mem', as well as returning an auxiliary result (method RK45).
        :param params['aux']: Auxiliary variable containing the sum of 'mem' above, with half a simulation step (method RK45)
        :param params['_init_start_']: Auxiliary parameter used by the system to perform special functions in the first simulation loop.
        :param params['_name_']: Auxiliary parameter delivered by the associated block, for error identification.
        :type time: float
        :type inputs: dict
        :type params['init_conds']: numpy.ndarray
        :type params['method']: str
        :type params['dtime']: float
        :type params['mem']: numpy.ndarray
        :type params['mem_list']: numpy.ndarray
        :type params['mem_len']: float
        :type params['nb_loop']: int
        :type params['RK45_Klist']: numpy.ndarray
        :type params['add_in_memory']: bool
        :type params['aux']: numpy.ndarray
        :type params['_init_start_']: bool
        :type params['_name_']: str
        :return: The accumulated value of all inputs since step zero weighted by the sampling time.
        :rtype: numpy.ndarray
        :examples: See example in :ref:`examples:sine integration` and :ref:`examples:convergent ode system`.
        :notes: The 'init_conds' parameter must be set by the user if the input has more than one dimension. You can define a vector value as [a,b,...], with a and b scalar values.

        """
        logger.debug(f"Integrator function called with time={time}, inputs={inputs}, params={params}")

    # Initialization (this step happens only in the first iteration)
        if params['_init_start_']:
            params['dtime'] = dtime
            params['mem'] = np.atleast_1d(np.array(params['init_conds'], dtype=float))  # Ensure float type
            params['mem_list'] = [np.zeros_like(params['mem'])]
            params['mem_len'] = 5.0
            params['_init_start_'] = False
            params['aux'] = np.zeros_like(params['mem'])  # Initialize 'aux'

            if params['method'] == 'RK45':
                params['nb_loop'] = 0
                params['RK45_Klist'] = [0, 0, 0, 0]  # K1, K2, K3, K4

            params['add_in_memory'] = True

        if output_only:
            old_add_in_memory = params['add_in_memory']
            params['add_in_memory'] = next_add_in_memory  # Update for next loop
            if old_add_in_memory:
                return {0: params['mem']}
            else:
                return {0: params['aux']}
        else:
            # Checks if the new input vector dimensions match.
            if isinstance(inputs[0], (float, int)):
                inputs[0] = np.atleast_1d(inputs[0])
            
            if params['mem'].shape != inputs[0].shape:
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

            logger.debug(f"Integrator function completed. mem={params['mem']}")
            return {0: params['mem']} if params['add_in_memory'] else {}

    def transfer_function(self, time, inputs, params, output_only=False):
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
    

    def derivative(self, time, inputs, params):
        """
        Derivative function

        :purpose: Function that obtains the derivative of a signal.
        :description: This is a process type function. It takes the input value and the value of the current time, then takes the difference of these with their previous and obtains the slope.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :param params['t_old']: Previous value of the variable time.
        :param params['i_old']: Previous value of the entry.
        :param params['_init_start_']: Auxiliary parameter used by the system to perform special functions in the first simulation loop.:type time: float
        :type inputs: dict
        :type params['t_old']: float
        :type params['i_old']: float
        :type params['_init_start_']: bool
        :return: The slope between the previous value and the current value.
        :rtype: numpy.ndarray
        :notes: ...

        """
        if params['_init_start_']:
            params['t_old'] = time
            params['i_old'] = np.array(inputs[0], dtype=float)
            params['didt_old'] = np.zeros_like(params['i_old'])
            params['_init_start_'] = False
            return {0: params['didt_old']}
        
        if time == params['t_old']:
            return {0: np.array(params['didt_old'])}
        
        dt = time - params['t_old']
        di = np.array(inputs[0], dtype=float) - params['i_old']
        didt = di/dt
        
        params['t_old'] = time
        params['i_old'] = np.array(inputs[0], dtype=float)
        params['didt_old'] = didt
        
        return {0: np.array(didt)}


    def terminator(self, time, inputs, params):
        """
        Signal terminator function

        :purpose: Function that terminates with the signal.
        :description: This is a drain type function. It takes any input value and does nothing with it. This function is useful for terminating signals that will not be plotted.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :type time: float
        :type inputs: dict
        :return: A value set in zero.
        :rtype: numpy.ndarray
        :examples: See example in :ref:`examples:signal products`.

        """
        return {0: np.array([0.0])}


    def export(self, time, inputs, params):
        """
        Block to save and export block signals

        :purpose: Function that accumulates values over time for later export to .npz.
        :description: This is a drain type function. It takes the input value and concatenates it to a vector. If the input has more than one dimension, the function concatenates so that the saving vector has the corresponding dimensions as a function of time.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :param params['str_name']: String supplied by the user with the names of the input values separated by comma: ("value1,value2,value3,...")
        :param params['vec_dim']: Value defined by the function that gets the number of dimensions of the input.
        :param params['vec_labels']: Vector produced by the function that gets the name for each element of the saving vector.
        :param params['vector']: Vector that accumulates the input values of the block.
        :param params['_init_start_']: Auxiliary parameter used by the system to perform special functions in the first simulation loop.
        :param params['_skip_']: Auxiliary parameter used by the system to indicate when not to save the input value (RK45 half steps).
        :param params['_name_']: Auxiliary parameter delivered by the associated block, for error identification.:type time: float
        :type inputs: dict
        :type params['str_name']: str
        :type params['vec_dim']: float
        :type params['vec_labels']: numpy.ndarray
        :type params['vector']: numpy.ndarray
        :type params['_init_start_']: bool
        :type params['_skip_']: bool
        :type params['_name_']: str
        :return: A value set in zero.
        :rtype: numpy.ndarray
        :examples: See example in :ref:`examples:export data` and :ref:`examples:convergent ode system`.
        :notes: If not enough labels are detected for 'vec_labels', the function adds the remaining labels using '_name_' and a number depending on the number of missing names.

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


    def scope(self, time, inputs, params):
        """
        Function to plot block signals

        :purpose: Function that accumulates values over time to plot them with pyqtgraph both later and during the simulation.
        :description: This is a drain type function. It takes the input value and concatenates it to a vector. If the input has more than one dimension, the function concatenates in such a way that the saving vector has the corresponding dimensions as a function of time.
        :param time: Value indicating the current period in the simulation.
        :param inputs: Dictionary that provides one or more entries for the function (if applicable).
        :param params['labels']: String supplied by the user with the names of the input values separated by comma: ("value1,value2,value3,...")
        :param params['vec_dim']: Value defined by the function that gets the number of dimensions of the input.
        :param params['vec_labels']: Vector produced by the function that gets the name for each element of the saving vector.
        :param params['vector']: Vector that accumulates the input values of the block.
        :param params['_init_start_']: Auxiliary parameter used by the system to perform special functions in the first simulation loop.
        :param params['_skip_']: Auxiliary parameter used by the system to indicate when not to save the input value (RK45 half steps).
        :param params['_name_']: Auxiliary parameter delivered by the associated block, for error identification.
        :type time: float
        :type inputs: dict
        :type params['labels']: str
        :type params['vec_dim']: float
        :type params['vec_labels']: numpy.ndarray
        :type params['vector']: numpy.ndarray
        :type params['_init_start_']: bool
        :type params['_skip_']: bool
        :type params['_name_']: str
        :return: A value set in zero.
        :rtype: numpy.ndarray
        :examples: See example in :ref:`examples:sine integration`, :ref:`examples:signal products`, :ref:`examples:gaussian noise`.
        :notes: If not enough labels are detected for 'vec_labels', the function adds the remaining ones using '_name_' and a number depending on the number of missing names.

        """
        # To prevent saving data in the wrong iterations (integration method RK45 in use)
        if '_skip_' in params.keys() and params['_skip_']:
            params['_skip_'] = False
            return {0: np.array([0.0])}
        # Initialization of the saving vector
        if params['_init_start_']:
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
        else:
            aux_vector = params['vector']
            aux_vector = np.concatenate((aux_vector, np.atleast_1d(inputs[0])))
        params['vector'] = aux_vector
        return {0: np.array([0.0])}
