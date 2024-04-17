import jax

# TODO: i really hate that i cannot just attach everything from jax.array to array with basis :///
def forward_dunders(cls):
    dunder_methods = [
        '__eq__', '__add__', '__sub__', '__mul__', '__matmul__',
        '__truediv__', '__floordiv__', '__mod__', '__pow__',
        '__radd__', '__rsub__', '__rmul__', '__rmatmul__',
        '__rtruediv__', '__rfloordiv__', '__rmod__', '__rpow__',
        '__neg__', '__pos__', '__abs__', '__invert__'
    ]

    def make_method(name):
        def method(self, *args, **kwargs):
            # Convert ArrayWithBasis args to their underlying arrays for operation
            new_args = [(arg.array if isinstance(arg, ArrayWithBasis) else arg) for arg in args]
            new_kwargs = {k: (v.array if isinstance(v, ArrayWithBasis) else v) for k, v in kwargs.items()}
            result = getattr(self.array, name)(*new_args, **new_kwargs)
            if isinstance(result, jax.Array):
                return result
            return result
        return method

    for name in dunder_methods:
        if hasattr(jax.Array, name):
            setattr(cls, name, make_method(name))

    return cls

@forward_dunders
class ArrayWithBasis:
    def __init__(self, array, is_in_site_basis):
        self.array = array
        self.is_in_site_basis = is_in_site_basis

    def __getattr__(self, name):
        attr = getattr(self.array, name)
        if callable(attr):
            def method(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, jax.Array):
                    return result
                return result
            return method
        return attr

    def __getitem__(self, key):
        result = self.array[key]
        if isinstance(result, jax.Array):
            return result
        return result

    def __array__(self):
        return self.array

    def __repr__(self):
        return f"ArrayWithBasis(array={self.array}, is_in_site_basis={self.is_in_site_basis})"
