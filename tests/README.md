# Test Guidelines for bioNC

This document provides guidelines for writing and running tests in the bioNC project.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_natural_segment.py

# Run tests with verbose output
pytest tests/ -v

# Run tests matching a pattern
pytest tests/ -k "natural_segment"
```

## Test Structure

### File Naming Convention
- Test files should be named `test_<feature>.py`
- Example: `test_natural_segment.py`, `test_joints.py`, `test_maths.py`

### Test Organization
Tests are organized by feature/module:

| Test File | Coverage Area |
|-----------|---------------|
| `test_natural_segment.py` | NaturalSegment class functionality |
| `test_natural_coordinates.py` | Natural coordinates for numpy |
| `test_natural_coordinates_casadi.py` | Natural coordinates for casadi |
| `test_biomech_model.py` | BiomechanicalModel operations |
| `test_joints.py` | Joint constraints and kinematics |
| `test_forward_dynamics.py` | Forward dynamics computations |
| `test_inverse_dynamics.py` | Inverse dynamics computations |
| `test_repr_str.py` | String representations (__repr__, __str__) |
| `test_maths.py` | Mathematical utilities |

## Testing Both Backends (numpy and casadi)

**Important:** bioNC has two computational backends: `numpy` and `casadi`. Most tests should verify both backends produce consistent results.

### Using `@pytest.mark.parametrize`

Use the parametrize decorator to test both backends in a single test function:

```python
import pytest
import numpy as np
from .utils import TestUtils

@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_my_feature(bionc_type):
    # Import the appropriate backend
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            NaturalSegment,
            SegmentNaturalCoordinates,
        )
    else:
        from bionc.bionc_numpy import (
            NaturalSegment,
            SegmentNaturalCoordinates,
        )
    
    # Write test code that works with both backends
    segment = NaturalSegment(name="test", length=0.4, mass=1.5)
    # ... assertions ...
```

### Multiple Parametrizations

You can combine multiple parametrizations:

```python
@pytest.mark.parametrize("joint_type", JointType)
@pytest.mark.parametrize("bionc_type", ["numpy", "casadi"])
def test_joints(bionc_type, joint_type):
    # Test all combinations of backends and joint types
    ...
```

## Test Utilities

The `utils.py` file provides helper classes and functions:

### `TestUtils` Class

```python
from .utils import TestUtils

# Assert numerical equality (works with both numpy arrays and casadi MX)
TestUtils.assert_equal(actual, expected, decimal=6)

# Convert casadi MX to numpy array for comparison
array = TestUtils.mx_to_array(mx_variable)

# Convert either type to numpy array
array = TestUtils.to_array(value)

# Get the path to the bionc folder
path = TestUtils.bionc_folder()
```

### Key Parameters for `assert_equal`:
- `decimal`: Number of decimal places for comparison (default: 6)
- `squeeze`: Whether to squeeze single-dimensional entries (default: True)
- `expand`: Whether to expand casadi functions (default: True)

## Best Practices

### 1. Always Test Both Backends
```python
@pytest.mark.parametrize("bionc_type", ["numpy", "casadi"])
def test_feature(bionc_type):
    ...
```

### 2. Use TestUtils for Comparisons
This handles the differences between numpy arrays and casadi MX types:
```python
# Good
TestUtils.assert_equal(result, expected, decimal=6)

# Avoid (unless you know the type)
np.testing.assert_almost_equal(result, expected)
```

### 3. Document Edge Cases
When testing error conditions, use `pytest.raises`:
```python
with pytest.raises(ValueError, match="expected error message"):
    function_that_should_raise()
```

### 4. Use Descriptive Test Names
Test names should describe what is being tested:
- `test_marker_features` ✓
- `test_angle_sanity_check` ✓
- `test_center_of_mass` ✓
- `test_1` ✗

## Adding New Tests

1. Create a new file `tests/test_<feature>.py` or add to an existing file
2. Import pytest and necessary modules
3. Use `@pytest.mark.parametrize` for backend testing
4. Use `TestUtils.assert_equal()` for numerical comparisons
5. Run your tests: `pytest tests/test_<feature>.py -v`

### Template for New Test File

```python
import numpy as np
import pytest
from .utils import TestUtils


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_new_feature(bionc_type):
    """Test description here."""
    if bionc_type == "casadi":
        from bionc.bionc_casadi import MyClass
    else:
        from bionc.bionc_numpy import MyClass
    
    # Arrange
    obj = MyClass(...)
    
    # Act
    result = obj.some_method()
    
    # Assert
    TestUtils.assert_equal(result, expected_value)
```

## Continuous Integration

Tests are automatically run on pull requests. Ensure all tests pass before submitting:

```bash
pytest tests/ -v
```
