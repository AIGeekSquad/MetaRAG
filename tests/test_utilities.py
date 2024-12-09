# Description: This file contains the unit tests utilities of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

def prune_none_values_from_model(dictionary: dict):
    result = {}
    for k, v in dictionary.items():
        if v is not None:
            if isinstance(v, dict):
                result[k] = prune_none_values_from_model(v)
            else:
                result[k] = v
    return result