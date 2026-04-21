import numpy as np
from typing import Self, Callable

# to open up the dual number to more functions, observe the definition of the dual number:
# f(a+b*eps) = f(a) + b * f'(a) * eps
# example:
# f(a) = arcsin(a)
# f'(a) = 1 / np.sqrt(1 - a ** 2.0)
# f(a + b*eps) = arcsin(a) + b / np.sqrt( 1 - a ** 2.0) * eps

# Then, to calculate the partial with respect to a variable, create a function f()
# f(dual [, dual, dual, ...])
# Have all dual numbers have an epsilon value of 0.0 except the target

# Example:
# partial f(x,y)/ partial(x) for f(x,y) = x * y + x  at (2,3)
# def f(x,y):
#    return x * y + x
#
# x = Dual_Number(real=2.0, eps=1.0)
# y = Dual_Number(real=3.0, eps=0.0)
# print(f(x,y))

class Dual_Number:
    def __init__(self, real: float = 0.0, eps: float = 0.0):
        self.real = real
        self.eps = eps

    def __add__(self, other: Self):
        if isinstance(other, Dual_Number):
            return Dual_Number(self.real + other.real, self.eps + other.eps)
        elif isinstance(other, float) or isinstance(other, int):
            return Dual_Number(self.real + other, self.eps)
        else:
            raise ValueError(f"Unknown input {other} to dual number addition")

    def __sub__(self, other: Self):
        if isinstance(other, Dual_Number):
            return Dual_Number(self.real - other.real, self.eps - other.eps)
        elif isinstance(other, float) or isinstance(other, int):
            return Dual_Number(self.real - other, self.eps)
        else:
            raise ValueError(f"Unknown input {other} to dual number subtraction")

    def __mul__(self, other: Self):
        if isinstance(other, Dual_Number):
            return Dual_Number(self.real * other.real, self.eps * other.real + self.real * other.eps)
        elif isinstance(other, float) or isinstance(other, int):
            return Dual_Number(self.real * other.real, self.eps * other.real)
        else:
            raise ValueError(f"Unknown input {other} to dual number multiplication")

    def __truediv__(self, other: Self):
        if isinstance(other, Dual_Number):
            return Dual_Number(self.real / other.real,
                               (self.eps * other.real - self.real * other.eps) / (other.real ** 2.0))
        elif isinstance(other, float) or isinstance(other, int):
            return Dual_Number(self.real / other.real, self.eps / other.real)
        else:
            raise ValueError(f"Unknown input {other} to dual number multiplication")

    def __repr__(self):
        return f'{self.real} + {self.eps}\u03B5'

    @property
    def sin(self):
        return Dual_Number(np.sin(self.real), self.eps * np.cos(self.real))

    @property
    def cos(self):
        return Dual_Number(np.cos(self.real), -self.eps * np.sin(self.real))

    @property
    def exp(self):
        return Dual_Number(np.exp(self.real), self.eps * np.exp(self.real))

    @property
    def ln(self):
        if self.real <= 0.0:
            raise ValueError(f"Can't take logarithm of negative dual number: {self}")
        return Dual_Number(np.log(self.real), self.eps / self.real)

    def __pow__(self, power: float, modulo=None):
        if np.abs(self.real) < 0.0000001 and power <= 0.0:
            raise ValueError(
                f"Can't raise negative number to negative power with this class: \nDual: {self}\nPower: {power}")
        else:
            return Dual_Number(self.real ** power, self.eps * power * (self.real ** (power - 1)))

    def __abs__(self):
        if np.abs(self.real) < 0.0000001:
            raise ValueError(f'No derivative of the absolute value at 0.')
        else:
            return Dual_Number(np.abs(self.real), self.eps * np.sign(self.real))

    @property
    def arcsin(self):
        return Dual_Number(np.arcsin(self.real), self.eps / np.sqrt(1.0 - self.real ** 2.0))

    @property
    def arccos(self):
        return Dual_Number(np.arccos(self.real), -self.eps / np.sqrt(1.0 - self.real ** 2.0))


class dn:
    @staticmethod
    def sin(a: Dual_Number):
        if isinstance(a, Dual_Number):
            return a.sin
        else:
            return np.sin(a)

    @staticmethod
    def cos(a: Dual_Number):
        if isinstance(a, Dual_Number):
            return a.cos
        else:
            return np.cos(a)

    @staticmethod
    def arcsin(a: Dual_Number):
        if isinstance(a, Dual_Number):
            return a.arcsin
        else:
            return np.arcsin(a)

    @staticmethod
    def arccos(a: Dual_Number):
        if isinstance(a, Dual_Number):
            return a.arccos
        else:
            return np.arccos(a)

    @staticmethod
    def ln(a: Dual_Number):
        if isinstance(a, Dual_Number):
            return a.ln
        else:
            return np.ln(a)

    @staticmethod
    def exp(a: Dual_Number):
        if isinstance(a, Dual_Number):
            return a.exp
        else:
            return np.exp(a)


def f(x: Dual_Number | float, y: Dual_Number | float):
    return dn.exp(dn.arccos(dn.sin(x * y) ** 2.0))


def g(x: Dual_Number | float):
    return dn.sin(x * 2.0)


def h(a: Dual_Number | float, b: Dual_Number | float, c: Dual_Number | float):
    return (a ** 2.0 + b ** 2.0 + c ** 2.0) / (a * b + c)


def deriv(func: Callable, inputs: list[float] | float, idx: int = None):
    if isinstance(inputs, float):
        dual_input = Dual_Number(inputs, 1.0)
        return func(dual_input)

    dual_inputs = []
    for input in inputs:
        dual_inputs.append(Dual_Number(input, 0.0))

    if idx is not None and idx < len(dual_inputs):
        dual_inputs[idx].eps = 1.0
        return func(*dual_inputs)

    if idx is not None:
        raise ValueError(f"Invalid idx ({idx}) for derivative; outside number of inputs {len(inputs)}.")

    all_duals = []
    for idx, dual in enumerate(dual_inputs):
        dual_inputs[idx].eps = 1.0
        all_duals.append(func(*dual_inputs))
        dual_inputs[idx].eps = 0.0

    return all_duals


def main():
    x = 2.0
    y = 3.0
    break_str = "~" * 50 + "\n"
    print("e^[ arccos( sin(x*y)^2 ) ]")
    print(f"{x=}, {y=}")
    print("Function value:")
    print(f(x, y))
    print()
    print("Partial with respect to x:")
    print(2.0 * np.sin(x * y) * y * np.cos(x * y) * -1.0 / np.sqrt(1.0 - np.sin(x * y) ** 4.0) * np.exp(
        np.arccos(np.sin(x * y) ** 2.0)))
    print("Dual Result:")
    print(deriv(f, [x, y], 0))
    print()
    print("Partial with respect to y:")
    print(2.0 * np.sin(x * y) * x * np.cos(x * y) * -1.0 / np.sqrt(1.0 - np.sin(x * y) ** 4.0) * np.exp(
        np.arccos(np.sin(x * y) ** 2.0)))
    print("Dual Result:")
    print(deriv(f, [x, y], 1))
    print()
    print("All possible duals:")
    [print(dual) for dual in deriv(f, [x, y])]
    print(break_str)

    z = 0.5
    print("sin(2z):")
    print(f'{z=}')
    print("Function value:")
    print(g(z))
    print("Partial with respect to z:")
    print(2.0 * np.cos(2.0 * z))
    print("Dual Result:")
    print(deriv(g, z))
    print(break_str)

    a, b, c = 1.0, 3.0, 2.0
    print("(a ** 2.0 + b ** 2.0 + c ** 2.0) / (a * b + c)")
    print(f"{a=}, {b=}, {c=}")
    print("Function Value:")
    print(h(a, b, c))
    print("a-Partial")
    print((2.0 * a * (a * b + c) - b * (a ** 2.0 + b ** 2.0 + c ** 2.0))/(a*b + c) ** 2)
    print("b-Partial")
    print((2.0 * b * (a * b + c) - a * (a ** 2.0 + b ** 2.0 + c ** 2.0))/(a*b + c) ** 2)
    print("c-Partial")
    print((2.0 * c * (a * b + c) - (a ** 2.0 + b ** 2.0 + c ** 2.0))/(a*b + c) ** 2)
    print("All Duals:")
    [print(dual) for dual in deriv(h, [a, b, c])]



if __name__ == '__main__':
    main()
