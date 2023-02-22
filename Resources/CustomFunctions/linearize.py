#
# Copyright (c) 2017 Modelon AB
#
import os
import numbers
import tempfile

import pandas
import numpy as np
import scipy


def signature():
    """
    The signature function specifying how the custom function is presented to the user
    in Modelon Impact.
    :return: A dictionary on the format:
                "version" - The version number of the custom function
                "name" - The name of the custom function to appear in Modelon Impact
                "description" - A description of what the custom function does.
                "parameters" - A list of parameters to be set by the user via the
                Modelon Impact simulation browser (optional).
                Each parameter is specified by a dictionary on the format (all
                optional):
                   "name" - the name of the parameter to appear in Modelon Impact
                   "type" - The type of the parameter: String, Number, Boolean or
                   Enumeration
                   "description" - A description of the parameter
                   "defaultValue" - The default value of the parameter
    """
    return {
        "version": "0.0.1",
        "name": "linearize",
        "description": "Linearize the model and compute its state space representation"
        "(matrices A, B, C and D).",
        "parameters": [
            {
                "name": "t_linearize",
                "type": "Number",
                "description": "Time (in seconds) at which to perform linearization."
                "To linearize at initialization,"
                "set t=0.",
                "defaultValue": 1,
            },
            {
                "name": "print_to_log",
                "type": "Boolean",
                "description": "Linearized model statistics are printed in the log, "
                "if this option is set to True",
                "defaultValue": True,
            },
        ],
    }


def run(
    get_fmu,
    environment,
    parametrization,
    upload_custom_artifact,
    t_linearize,
    print_to_log,
):
    """
    The run function, defining the operation or computation of the custom function.
    :param get_fmu: A function returning an FMU object for the model the custom
    function is applied on, with applied
    non-structural parameters as set in Modelon Impact.
    :param environment: A dictionary specifying environment variables:
                          "result_folder_path" - The path to the folder where the
                          result is to be saved
                          "result_file_name" - The name of the file where the
                          results are to be saved
                          "workspace_id" - The workspace ID
                          "case_id" - The case ID
                          "experiment_id" - The experiment ID
                          "log_file_name" - The name of the log file
    :param upload_custom_artifact: Function for uploading a custom artifact to the
    storage. Takes arguments:
                          "artifact_id": The artifact ID.
                          "local_file_path": Path to the local custom artifact to upload
            Returns the route for downloading the artifact.
    :param parametrization: The parametrization of the model as set in Modelon Impact
    experiment mode.
    :param t_linearize: Time (in seconds) at which to perform linearization.
    :param print_to_log: Toggle weather the linearized model statistics should be shown
                          in the simulation log.
    """

    # In this case, the linearization is packaged into a separate function. This
    # enables to use it outside of Modelon Impact and thereby also makes
    # it convenient to test.
    model = get_fmu()
    for key, value in parametrization.items():
        model.set(key, value)
    return linearize(
        model, environment, upload_custom_artifact, t_linearize, print_to_log
    )


def linearize(model, environment, upload_custom_artifact, t_linearize, print_to_log):
    """
    Compute the ABCD state space representation for a model and write the result to
    a .csv-file. Also save the result as a .mat-file and upload it as a custom artifact.
    The .mat-file can be used to load the result into MATLAB.
    :param model: An FMU object for the model to linearize.
    :param environment: A dictionary specifying environment variables:
                          "result_folder_path" - The path to the folder where the
                          result is to be saved
                          "result_file_name" - The name of the file where the results
                          are to be saved
                          "workspace_id" - The workspace ID
                          "case_id" - The case ID
                          "experiment_id" - The experiment ID
                          "log_file_name" - The name of the log file
    :param upload_custom_artifact: Function for uploading a custom artifact to the
    storage. Takes arguments:
                          "artifact_id": The artifact ID.
                          "local_file_path": Path to the local custom artifact to upload
            Returns the route for downloading the artifact.
    :param t_linearize: The time to simulate the model before linearizing.
    :param print_to_log: Toggle weather the linearized model statistics should be shown
                          in the simulation log
    """

    # Start by type checking the parameter, in case an invalid entry is given by
    # the user
    if not isinstance(t_linearize, numbers.Number) or t_linearize < 0:
        raise ValueError("The parameter t_linearize needs to be a non-negative number.")
    if t_linearize == 0:
        model.initialize()
    else:
        model.simulate(final_time=t_linearize)

    # Retrieve the state space representation of the linearized model
    result = model.get_state_space_representation(use_structure_info=False)

    ss = {matrix_name: result[i] for i, matrix_name in enumerate(["A", "B", "C", "D"])}

    # Pretty print the matrices to the simulation log
    if print_to_log:
        for matrix_name, result in ss.items():
            print('\n' + matrix_name + ' = [')
            matrix_shape = result.shape
            if not (matrix_shape[0] == 0 or matrix_shape[1] == 0):
                max_len = min(len(str(e)) for row in result for e in row)
                for row in result:
                    print(
                        '['
                        + ", ".join(
                            ['{:<{max_len}}'.format(e, max_len=max_len) for e in row]
                        )
                        + '],'
                    )
            print(']')

    # Scalarize the state space matrices
    scalarized_ss = {
        "{}[{},{}]".format(matrix_name, index[0], index[1]): [x]
        for matrix_name, matrix in ss.items()
        for index, x in np.ndenumerate(matrix)
    }

    # Write the matrices to a csv file in the prescribed path
    csv_file_path = os.path.join(
        environment["result_folder_path"], environment["result_file_name"]
    )
    df = pandas.DataFrame(data=scalarized_ss)
    df.to_csv(csv_file_path, index=False)

    # Add variable names
    state_names = list(model.get_states_list().keys())
    input_names = list(model.get_input_list().keys())
    output_names = list(model.get_output_list().keys())

    ss['state_names'] = state_names
    ss['input_names'] = input_names
    ss['output_names'] = output_names

    # Add operating point
    operating_point_time = t_linearize
    operating_point_states = [x[0] for x in model.get(state_names)]
    operating_point_derivatives = list(model.get_derivatives())
    operating_point_inputs = [x[0] for x in model.get(input_names)]
    operating_point_outputs = [x[0] for x in model.get(output_names)]

    ss['operating_point_time'] = operating_point_time
    ss['operating_point_states'] = np.array(operating_point_states)
    ss['operating_point_derivatives'] = np.array(operating_point_derivatives)
    ss['operating_point_inputs'] = np.array(operating_point_inputs)
    ss['operating_point_outputs'] = np.array(operating_point_outputs)

    # Pretty print the linearization statistics to the simulation log
    if print_to_log:
        if state_names:
            print('\n' + "# At operating point {}s :".format(str(t_linearize)) + '\n')
            print(f'state_names = {state_names}\n')
            print(f'operating_point_states = {operating_point_states}\n')
            print(f'operating_point_derivatives = {operating_point_derivatives}\n')
        if input_names:
            print(f'input_names = {input_names}\n')
            print(f'operating_point_inputs = {operating_point_inputs}\n')
        if output_names:
            print(f'output_names = {output_names}\n')
            print(f'operating_point_outputs = {operating_point_outputs}\n')

    # Write result to a .mat file that can be imported in MATLAB
    # First write the result to a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_mat_file = os.path.join(temp_dir, "result.mat")
    scipy.io.savemat(temp_mat_file, ss)

    # Now upload the result to the server as a custom artifact
    artifact_id = "ABCD"
    artifact_route = upload_custom_artifact(artifact_id, temp_mat_file)

    # Finally print the route where the artifact can be accessed
    print('Stored artifact with ID: {}'.format(artifact_id))
    print('')
    print('Artifact can be downloaded from @artifact[here]({})'.format(artifact_route))
