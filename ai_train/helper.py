from typing import Dict, List, Tuple, Any


def triangular_mf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.
    a: start (0 membership)
    b: peak (1 membership)
    c: end (0 membership)
    """
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a)
    if b < x < c:
        return (c - x) / (c - b)
    return 0.0


def trapezoidal_mf(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.
    a: start (0 membership)
    b: start of plateau (1 membership)
    c: end of plateau (1 membership)
    d: end (0 membership)
    """
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b <= x <= c:
        return 1.0
    if c < x < d:
        return (d - x) / (d - c)
    return 0.0


def left_shoulder_mf(x: float, a: float, b: float) -> float:
    """
    Left shoulder membership function (starts from plateau).
    a: end of plateau (membership 1)
    b: end of slope (membership 0)
    """
    if x <= a:
        return 1.0
    if a < x < b:
        return (b - x) / (b - a)
    return 0.0


def right_shoulder_mf(x: float, a: float, b: float) -> float:
    """
    Right shoulder membership function (ends at plateau).
    a: start of slope (membership 0)
    b: start of plateau (membership 1)
    """
    if x <= a:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    return 1.0


def fuzzify(value: float, mf_config: Dict[str, Tuple[str, List[float]]]) -> Dict[str, float]:
    """
    Fuzzification: Convert a crisp value into membership degrees for multiple sets.
    mf_config: Dict mapping set names to (type, params)
    Supported types: 'triangular', 'trapezoidal', 'left_shoulder', 'right_shoulder'
    """
    results: Dict[str, float] = {}
    for set_name, (mf_type, params) in mf_config.items():
        if mf_type == 'triangular':
            results[set_name] = triangular_mf(value, *params)
        elif mf_type == 'trapezoidal':
            results[set_name] = trapezoidal_mf(value, *params)
        elif mf_type == 'left_shoulder':
            results[set_name] = left_shoulder_mf(value, *params)
        elif mf_type == 'right_shoulder':
            results[set_name] = right_shoulder_mf(value, *params)
    return results


def evaluate_rule(
    inputs_memberships: Dict[str, Dict[str, float]], 
    rule_definition: Dict[str, Any]
) -> Tuple[float, Any]:
    """
    Rule Evaluation (Sugeno logic):
    inputs_memberships: dict of dicts { 'input_name': { 'set_name': degree } }
    rule_definition: dict { 'if': [('input_name', 'set_name'), ...], 'then': constant_value }
    Returns: (firing_strength, output_value)
    """
    # Using 'min' operator for AND logic
    degrees: List[float] = []
    for var_name, set_name in rule_definition['if']:
        degrees.append(inputs_memberships[var_name][set_name])
    
    firing_strength = min(degrees) if degrees else 0.0
    return firing_strength, rule_definition['then']


def sugeno_defuzzification(firing_strengths: List[float], rule_outputs: List[float]) -> float:
    """
    Zero-order Sugeno Defuzzification: Weighted Average.
    firing_strengths: list of w_i
    rule_outputs: list of z_i (constants)
    """
    total_weight = sum(firing_strengths)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(w * z for w, z in zip(firing_strengths, rule_outputs))
    return weighted_sum / total_weight

