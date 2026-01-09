from blocks.base_block import BaseBlock
import numpy as np
from scipy.integrate import solve_ivp
import logging

logger = logging.getLogger(__name__)


class IntegratorBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Integrator"

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def params(self):
        return {
            "init_conds": {"default": 0.0, "type": "float"},
            "method": {"default": "SOLVE_IVP", "type": "string"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return "Integrates the input signal over time."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Integrator uses 1/s text rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        """
        Integrator block with multiple integration methods.
        """
        output_only = kwargs.get('output_only', False)
        next_add_in_memory = kwargs.get('next_add_in_memory', True)
        dtime = kwargs.get('dtime', params.get('dtime', 0.01))
        
        # Initialization
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
            result = {0: params.get('output', params['mem']), 'E': False}
            return result
        
        # Check input dimensions
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

        # Integration by method
        if params['method'] == 'FWD_EULER':
            if params['add_in_memory']:
                params['mem'] += params['dtime'] * inputs[0]
            else:
                params['aux'] = np.array(params['mem'] + 0.5 * params['dtime'] * inputs[0])
                return {0: params['aux'], 'E': False}
        elif params['method'] == 'BWD_EULER':
            if params['add_in_memory']:
                params['mem'] += params['dtime'] * params['mem_list'][-1]
            else:
                params['aux'] = np.array(params['mem'] + 0.5 * params['dtime'] * params['mem_list'][-1])
                return {0: params['aux'], 'E': False}
        elif params['method'] == 'TUSTIN':
            if params['add_in_memory']:
                params['mem'] += 0.5*params['dtime'] * (inputs[0] + params['mem_list'][-1])
            else:
                params['aux'] = np.array(params['mem'] + 0.25 * params['dtime'] * (inputs[0] + params['mem_list'][-1]))
                return {0: params['aux'], 'E': False}
        elif params['method'] == 'RK45':
            K_list = params['RK45_Klist']
            K_list[params['nb_loop']] = params['dtime'] * np.array(inputs[0], dtype=float)
            params['RK45_Klist'] = K_list
            K1, K2, K3, K4 = K_list

            if params['nb_loop'] == 0:
                params['nb_loop'] += 1
                params['aux'] = params['mem'] + 0.5 * K1
                return {'E': False}
            elif params['nb_loop'] == 1:
                params['nb_loop'] += 1
                params['aux'] = params['mem'] + 0.5 * K2
                return {'E': False}
            elif params['nb_loop'] == 2:
                params['nb_loop'] += 1
                params['aux'] = params['mem'] + K3
                return {'E': False}
            elif params['nb_loop'] == 3:
                params['nb_loop'] = 0
                params['mem'] += (1 / 6) * (K1 + 2 * K2 + 2 * K3 + K4)
        elif params['method'] == 'SOLVE_IVP':
            mem_shape = params['mem'].shape
            y0 = np.atleast_1d(params['mem']).flatten()
            
            def fun(t, y):
                return np.atleast_1d(inputs[0]).flatten()
            
            sol = solve_ivp(fun, [time, time + dtime], y0)
            params['mem'] = sol.y[:, -1].reshape(mem_shape)
            return {0: params['mem'], 'E': False}
        else:
            logger.error(f"Unknown integration method {params['method']} in {params['_name_']}")
            return {'E': True, 'error': f"Unknown method: {params['method']}"}

        aux_list = params['mem_list']
        aux_list.append(inputs[0])
        if len(aux_list) > params['mem_len']:
            aux_list = aux_list[-5:]
        params['mem_list'] = aux_list

        result = {0: params['mem'], 'E': False} if params['add_in_memory'] else {'E': False}
        return result
