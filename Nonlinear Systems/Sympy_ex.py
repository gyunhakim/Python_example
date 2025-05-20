"""
Some examples using sympy
"""

import sympy
from sympy import oo

t, w, s = sympy.symbols("t, omega, s")
a = sympy.Symbol("a", positive=True)
n = sympy.Symbol("n", integer=True)

f = sympy.sin(t)
g = f/t

f_diff = f.diff()
f_int = f.integrate()
f_series = f.series()

limit = sympy.limit(g, t, 0)

print("\nf(t) = ", f, "\n\nDerivative of f(t) = ", f_diff, "\nIntegral of f(t) = ", f_int)
print("\nSeries expansion of f(t) = ", f_series)
sympy.pprint(f_series)

print("\nLimit of f(t)/t = ", limit)

ex_series = sympy.Sum((-1)**(n-1)/n, (n, 1, oo))
print("\nExample series: ", ex_series, " = ", ex_series.doit())
sympy.pprint(ex_series)

x = sympy.exp(-a*t)*sympy.Heaviside(t)
Xf = sympy.fourier_transform(x, t, w)
Xs = sympy.laplace_transform(x, t, s, noconds=True)

x1 = sympy.inverse_fourier_transform(Xf, w, t)
x2 = sympy.inverse_laplace_transform(Xs, s, t, noconds=True)

print("\nx(t) = ", x)
sympy.pprint(x)
print("\nFourier transform of x(t): ", Xf)
sympy.pprint(Xf)
print("Laplace transform of x(t): ", Xs)
sympy.pprint(Xs)
print("\nInverse Fourier transform of X(w): ", x1)
sympy.pprint(x1)
print("\nInverse Laplace transform of X(s): ", x2)
sympy.pprint(x2)
