import json
from typing import (List, Dict, Any, NoReturn)


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(path: str, data: Dict[str, Any]) -> NoReturn:
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f)


def update_parameters(fn: str, parameter_name: str,
                      new_value: Any) -> NoReturn:
    curr_params = dict(read_json(fn))
    curr_params.update({parameter_name: new_value})
    write_json(fn, curr_params)