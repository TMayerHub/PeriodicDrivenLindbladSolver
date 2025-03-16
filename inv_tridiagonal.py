import sympy as sp

def tridiagonal_inverse_symbolic(n):
    """Computes the symbolic inverse of a symmetric tridiagonal matrix with varying diagonal elements."""
    
    # Define symbolic variables for the diagonal and off-diagonal elements
    d = sp.symbols(f'd1:{n+1}')  # Diagonal elements (d1, d2, ..., dn)
    c = sp.symbols('c')  # Constant off-diagonal element
    
    # Construct the tridiagonal matrix
    T = sp.zeros(n, n)
    for i in range(n):
        T[i, i] = d[i]  # Main diagonal
        if i < n - 1:
            T[i, i+1] = c  # Upper diagonal
            T[i+1, i] = c  # Lower diagonal
    sp.pprint(T)
    # Compute the inverse symbolically
    T_inv = T.inv()
    
    return T_inv

# Example: Compute the inverse for a 4x4 symbolic tridiagonal matrix
n = 4
T_inverse = tridiagonal_inverse_symbolic(n)
sp.pprint(T_inverse)