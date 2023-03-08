import tempfile
import numpy as N
import os
from tests.utils import capture
from tests.mocks import LinearizeModelMock

from linearize import linearize

TEST_WS_ID = 'TEST_WS'
TEST_EXP_ID = 'TEST_EXP'
TEST_CASE_ID = 'TEST_CASE'


def _assert_file_equal(result, ref_result_file):
    ref_result_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'files',ref_result_file
    )
    with open(ref_result_file_path) as f1:
        for l1, l2 in zip(f1, result.split('\n')):
            assert l1.strip() == l2.strip()


def get_params_base():
    def upload_custom_artifact(artifact_id, local_path):
        workspace_id = TEST_WS_ID
        experiment_id = TEST_EXP_ID
        case_id = TEST_CASE_ID
        artifact_route = (
            'api/workspace/{}/experiments/{}/cases/{}/custom-artifacts/{}'
        ).format(workspace_id, experiment_id, case_id, artifact_id)
        return artifact_route

    params = {}
    params["t_linearize"] = 0
    params['environment'] = {}
    params['environment']["result_folder_path"] = tempfile.gettempdir()
    params['environment']["result_file_name"] = 'test.suffix'
    params['upload_custom_artifact'] = upload_custom_artifact
    params['parametrization'] = dict()
    params['print_to_log'] = True
    return params


def set_fmu_data(model: LinearizeModelMock):
    A = N.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    B = N.array([[0.0], [0.0], [0.0], [0.0], [0.0], [-0.33333337], [0.0], [0.0]])
    C = N.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    D = N.array([[0.0]])
    model.ss_matrix = (A, B, C, D)

    model.derivative_values = [-0.00181693, -0.0163713, -0.0190693, -0.054883]
    model.variable_data = {'d.a': 1, 'e.b': 2, 'f.c': 3, 'g.d': 4}
    model.states_list = list(model.variable_data.keys())


def test_linearize_at_init():
    params = get_params_base()
    model = LinearizeModelMock()
    set_fmu_data(model)
    params['model'] = model
    params["t_linearize"] = 0
    linearize(**params)
    assert model.solved_at_time_called == 0


def test_linearize_at_1():
    params = get_params_base()
    model = LinearizeModelMock()
    set_fmu_data(model)
    params['model'] = model
    params["t_linearize"] = 1
    with capture() as out:
        linearize(**params)
    _assert_file_equal(out[0], 'result_regular.txt')
    assert model.solved_at_time_called == 1


def test_linearize_at_1_no_print():
    params = get_params_base()
    model = LinearizeModelMock()
    set_fmu_data(model)
    params['model'] = model
    params["t_linearize"] = 1
    params['print_to_log'] = False
    with capture() as out:
        linearize(**params)
    _assert_file_equal(out[0], 'result_no_print.txt')
    assert model.solved_at_time_called == 1


def test_linearize_at_1_only_A():
    params = get_params_base()
    model = LinearizeModelMock()
    model.ss_matrix = (
        N.array([[0.0, 1.0], [0.0, 0.0]]),
        N.array([]),
        N.array([]),
        N.array([]),
    )
    params['model'] = model
    params["t_linearize"] = 1
    with capture() as out:
        linearize(**params)
    _assert_file_equal(out[0], "result_only_A.txt")
    assert model.solved_at_time_called == 1


def test_linearize_at_1_no_ss():
    params = get_params_base()
    model = LinearizeModelMock()
    params['model'] = model
    params["t_linearize"] = 1
    with capture() as out:
        linearize(**params)
    _assert_file_equal(out[0], "result_no_ss.txt")
    assert model.solved_at_time_called == 1


def test_linearize_with_inputs():
    params = get_params_base()
    model = LinearizeModelMock()
    set_fmu_data(model)
    params['model'] = model
    params["t_linearize"] = 1
    input_data = {'input1': 1, 'input2': 0}
    model.variable_data.update(input_data)
    model.input_list = list(input_data.keys())
    with capture() as out:
        linearize(**params)
    _assert_file_equal(out[0], "result_with_inputs.txt")
    assert model.solved_at_time_called == 1


def test_linearize_with_outputs():
    params = get_params_base()
    model = LinearizeModelMock()
    set_fmu_data(model)
    params['model'] = model
    params["t_linearize"] = 1
    output_data = {'output1': 1, 'output2': 0}
    model.variable_data.update(output_data)
    model.output_list = list(output_data.keys())
    with capture() as out:
        linearize(**params)
    _assert_file_equal(out[0], "result_with_outputs.txt")
    assert model.solved_at_time_called == 1


def test_linearize_with_io():
    params = get_params_base()
    model = LinearizeModelMock()
    set_fmu_data(model)
    params['model'] = model
    params["t_linearize"] = 1
    input_data = {'input1': 1, 'input2': 0, 'input3': 1, 'input4': 0}
    model.variable_data.update(input_data)
    model.input_list = list(input_data.keys())

    output_data = {'output1': 1, 'output2': 0, 'output3': 1, 'output4': 0}
    model.variable_data.update(output_data)
    model.output_list = list(output_data.keys())

    with capture() as out:
        linearize(**params)
    _assert_file_equal(out[0], "result_with_io.txt")
    assert model.solved_at_time_called == 1
