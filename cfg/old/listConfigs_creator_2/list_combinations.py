from importeur import *
import itertools

list_all_variables = list()
list_all_variables.append(
    [
        "orga.save.chain",
        "True",
    ]
)

list_all_variables.append(
    [
        "model.name",
        "multiFields_CNN",
        "multiFields_vector_CNN",
    ]
)

list_all_variables.append(
    [
        "model.optimiser",
        "adam",
    ]
)

list_all_variables.append(
    [
        "model.learning_rate",
        "1e-3",
    ]
)

list_all_variables.append(["model.epochs.number", "250"])

list_all_variables.append(
    [
        "model.batch_size",
        "64",
    ]
)

list_all_variables.append(
    [
        "data.directory.name",
        "CO2_pPGI_nBB_fullday",
        "CO2_pPGI_nBBO_fullday",
        "CO2_pPG_nBBOI_fullday",
    ]
)

list_all_variables.append(
    [
        "data.input.time",
        "False",
        "True",
    ]
)

list_all_variables.append(
    [
        "data.input.scale",
        "False",
        "True",
    ]
)

list_all_variables.append(
    [
        "data.input.winds.format",
        "none",
        "summary",
        "fields",
    ]
)

list_all_variables.append(
    [
        "data.input.winds.fields.number.current",
        "2",
        "6",
    ]
)

list_all_variables.append(
    [
        "data.input.dynamics.format",
        "none",
        "fields",
    ]
)

list_all_variables.append(
    [
        "data.input.dynamics.fields.number.current",
        "1",
        "3",
    ]
)

list_together_values_to_destroy = list()

list_together_values_to_destroy.append(
    [["model.name", "multiFields_CNN"], ["data.input.time", "True"]]
)
list_together_values_to_destroy.append(
    [["model.name", "multiFields_CNN"], ["data.input.scale", "True"]]
)
list_together_values_to_destroy.append(
    [["model.name", "multiFields_CNN"], ["data.input.winds.format", "summary"]]
)

list_together_values_to_destroy.append(
    [
        ["model.name", "multiFields_vector_CNN"],
        ["data.input.winds.format", "fields"],
        ["data.input.time", "False"],
    ]
)
list_together_values_to_destroy.append(
    [
        ["model.name", "multiFields_vector_CNN"],
        ["data.input.winds.format", "none"],
        ["data.input.time", "False"],
    ]
)

list_together_values_to_destroy.append(
    [
        ["data.input.winds.format", "none"],
        ["data.input.winds.fields.number.current", "6"],
    ]
)
list_together_values_to_destroy.append(
    [
        ["data.input.dynamics.format", "none"],
        ["data.input.dynamics.fields.number.current", "3"],
    ]
)

# ------------------------------------------------------------------------------------
# build_list_combinations
def build_list_combinations():

    dicts = {}
    values = range(len(list_all_variables))
    keys = [item[0] for item in list_all_variables]
    for i in values:
        dicts[keys[i]] = values[i]

    # create all combinations
    list_variables_names = list()
    list_variables_values = list()

    for i in range(len(list_all_variables)):
        list_variables_names.append(list_all_variables[i][0])
        list_variables_values.append(list_all_variables[i][1:])

    list_combinations = list(itertools.product(*list_variables_values))

    # destroy impossible combinations
    new_list_combinations = []

    initial_number_combinations = len(list_combinations)
    for combination in range(initial_number_combinations):

        destroy = "no"
        for index_destruction in range(len(list_together_values_to_destroy)):

            N_together_values = len(list_together_values_to_destroy[index_destruction])

            if all(
                list_together_values_to_destroy[index_destruction][i][1]
                == list_combinations[combination][
                    dicts[list_together_values_to_destroy[index_destruction][i][0]]
                ]
                for i in range(N_together_values)
            ):
                destroy = "yes"

        if destroy == "no":
            new_list_combinations.append(list_combinations[combination])

    return new_list_combinations


# ------------------------------------------------------------------------------------
