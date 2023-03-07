import numpy as N


class LinearizeModelMock:
    def __init__(self):
        self.states_list = []
        self.input_list = []
        self.output_list = []
        self.derivative_values = []
        self.ss_matrix = (N.array([]), N.array([]), N.array([]), N.array([]))
        self.variable_data = {}
        self.solved_at_time_called = None

    def set(self, key, value):
        pass

    def get_state_space_representation(self, use_structure_info=False):
        return self.ss_matrix

    def solve_at_time(self, t_linearize):
        self.solved_at_time_called = t_linearize

    def get_state_vars(self):
        return self.states_list

    def get_input_vars(self):
        return self.input_list

    def get_output_vars(self):
        return self.output_list

    def get(self, variable_names):
        return [self.variable_data[variable_name] for variable_name in variable_names]

    def get_derivatives(self):
        return self.derivative_values
