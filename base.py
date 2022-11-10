import polynomial
import lagrange_psr
import constant
import numpy as np
import qiskit

def zz_measure(qc: qiskit.QuantumCircuit):
    _00 = np.asarray([1, 0, 0, 0])
    _01 = np.asarray([0, 1, 0, 0])
    _10 = np.asarray([0, 0, 1, 0])
    _11 = np.asarray([0, 0, 0, 1])
    psi = qiskit.quantum_info.Statevector(qc)
    return np.abs(np.inner(_00, psi))**2 - np.abs(np.inner(_01, psi))**2 - np.abs(np.inner(_10, psi))**2 + np.abs(np.inner(_11, psi))**2

def measure(qc: qiskit.QuantumCircuit, qubits, cbits=[]):
    """Measuring the quantu circuit which fully measurement gates
    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (np.ndarray): List of measured qubit

    Returns:
        - float: Frequency of 00.. cbit
    """
    n = len(qubits)
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, n):
        qc.measure(qubits[i], cbits[i])

    counts = qiskit.execute(
        qc, backend=constant.backend,
        shots=constant.num_shots).result().get_counts()

    return counts.get("0" * len(qubits), 0) / constant.num_shots


def upper_matrix(M):
    """Example:

    [[1,2,3,4],

    [5,6,7,8],

    [9,10,12,13],

    [14,15,16,17]]

    Return: 
        [4, 7, 13]


    Args:
        M (np.ndarray): non-zero square matrix  

    Returns:
        np.ndarray: zic-zac elements except the first
    """
    upper_elements = []
    for i in range(0, M.shape[0]):
        for j in range(i + 1, M.shape[0], 2):
            upper_elements.append(M[i, j])
    return np.expand_dims(np.asarray(upper_elements[1:]), 1)


def calculate_Lambda(lambdas: list, x):
    """Convert the exponential expression to polynomial expression

    Args:
        lambdas (list): assume that we have already the eigenvalues = {\lambda_i}
        x: parameter value of quantum gate

    Returns:
        polynomial.Polynomial: polynomial expression of quantum gate
    """
    #
    n = len(lambdas)
    Ss = []
    for k in range(0, n):
        P = np.exp(-1j * x * lambdas[k])
        Vs = []
        for l in range(0, n):
            if l != k:
                P = P / (lambdas[k] - lambdas[l])
                Vs.append(polynomial.Polymonial([-lambdas[l], 1]))
        V = polynomial.multiXPoly(Vs)
        Ss.append(V.multiX(P))
    S = polynomial.addXPoly(Ss)
    return S


def calculate_Lambda_matrix(lambdas: np.ndarray, x: float):
    """Return square matrix which contains Lambda_i * Lambda_j at each element

    Args:
        lambdas (np.ndarray): eigenvalues of quantum gate
        x (float): phase

    Returns:
        np.ndarray: square matrix
    """
    Lambdas = calculate_Lambda(lambdas, x).coeff
    M = np.zeros([len(Lambdas), len(Lambdas)], dtype=np.complex128)
    for i in range(0, len(Lambdas)):
        for j in range(0, len(Lambdas)):
            M[i, j] = np.conjugate(Lambdas[i]) * Lambdas[j]
    return M


def calculate_Tau_matrix(B, G, n):
    Tau = np.zeros([n, n], dtype=np.complex128)
    for i in range(0, n):
        for j in range(0, n):
            Tau[i, j] = np.linalg.matrix_power(
                G, i) @ B @ np.linalg.matrix_power(G, j)
    return Tau


def check_symmetric(matrix, rtol=1e-05, atol=1e-08):
    return np.allclose(matrix, np.conjugate(matrix.T), rtol=rtol, atol=atol)


def unit_vector(i, length):
    unit_vector = np.zeros((length))
    unit_vector[i] = 1.0
    return unit_vector

def create_log_step_sizes(low, high, size):
    steps = []
    step = low
    while (step < high):
        steps.append(step)
        step = step + size
        size = size * 1.01
    return steps

def create_logsin_step_sizes(low, high, size):
    steps = []
    step = low
    while (step < high):
        if step < 0.1:
            steps.append(np.sin(step))
        else:
            steps.append((step))
        step = step + size
        size = size * 1.01
    return np.round(steps, 2)

def second_derivative_2psr(f, thetas, i, j, alpha=np.pi/3):
    length = thetas.shape[0]
    k1 = f(thetas + alpha*(unit_vector(i, length) + unit_vector(j, length)))
    k2 = -f(thetas + alpha * (unit_vector(i, length) - unit_vector(j, length)))
    k3 = -f(thetas - alpha * (unit_vector(i, length) - unit_vector(j, length)))
    k4 = f(thetas - alpha*(unit_vector(i, length) + unit_vector(j, length)))
    return (1/(4*(np.sin(alpha))**2))*(k1 + k2 + k3 + k4)


def second_derivative_4psr(f, thetas, i, j):
    alpha1 = np.pi/2
    alpha2 = np.pi
    d1 = 1j
    d2 = 1j*(-1 + np.sqrt(2)) / 2
    length = thetas.shape[0]

    k1A = -1*(f(thetas + alpha1*unit_vector(i, length) + alpha1*unit_vector(j, length))
              - f(thetas + alpha1*unit_vector(i, length) - alpha1*unit_vector(j, length)))
    k1B = -(1-np.sqrt(2))/2*(f(thetas + alpha1*unit_vector(i, length) + alpha2*unit_vector(j, length))
                             - f(thetas + alpha1*unit_vector(i, length) - alpha2*unit_vector(j, length)))

    k2A = 1*(f(thetas - alpha1*unit_vector(i, length) + alpha1*unit_vector(j, length))
             - f(thetas - alpha1*unit_vector(i, length) - alpha1*unit_vector(j, length)))
    k2B = (1-np.sqrt(2))/2*(f(thetas - alpha1*unit_vector(i, length) + alpha2*unit_vector(j, length))
                            - f(thetas - alpha1*unit_vector(i, length) - alpha2*unit_vector(j, length)))

    k3A = -(1-np.sqrt(2))/2*(f(thetas + alpha2*unit_vector(i, length) + alpha1*unit_vector(j, length))
                             - f(thetas + alpha2*unit_vector(i, length) - alpha1*unit_vector(j, length)))
    k3B = -(1-np.sqrt(2))**2/4*(f(thetas + alpha2*unit_vector(i, length) + alpha2*unit_vector(j, length))
                                + f(thetas + alpha2*unit_vector(i, length) - alpha2*unit_vector(j, length)))

    k4A = (1-np.sqrt(2))/2*(f(thetas - alpha2*unit_vector(i, length) + alpha1*unit_vector(j, length))
                            - f(thetas - alpha2*unit_vector(i, length) - alpha1*unit_vector(j, length)))
    k4B = (1-np.sqrt(2))**2/4*(f(thetas - alpha2*unit_vector(i, length) + alpha2*unit_vector(j, length))
                               + f(thetas - alpha2*unit_vector(i, length) - alpha2*unit_vector(j, length)))

    return (-1j/2)**2*(k1A + k1B + k2A + k2B + k3A + k3B + k4A + k4B)


# def two_prx(f, thetas, j):
#     length = thetas.shape[0]
#     return constant.two_term_psr['r'] * (
#         f(thetas + constant.two_term_psr['s'] * unit_vector(j, length)) -
#         f(thetas - constant.two_term_psr['s'] * unit_vector(j, length))
#     )


def two_prx_hLMG(f, thetas, h):
    lambdas = constant.lambdas
    length = thetas.shape[0]
    alphas, d = lagrange_psr.lagrange_psr(lambdas)
    grad = np.zeros(length, dtype = np.complex128)
    for i in range(0, length):
        for j in range(0, len(d)):
            grad[i] += d[j] * (
                f(thetas + alphas[j]* unit_vector(i, length), h) -
                f(thetas - alphas[j]* unit_vector(i, length), h)
            )
    return np.real((-1j)*grad)

def pseudo_two_prx(f, thetas, j, step_size):
    length = thetas.shape[0]
    return (1/(2*np.sin(step_size))) * (
        f(thetas + step_size * unit_vector(j, length)) -
        f(thetas - step_size * unit_vector(j, length))
    )

def a_pseudo_two_prx(f_left, f_right, step_size):
    return (1/(2*np.sin(step_size))) * (
        f_left - f_right
    )

def a_two_finite_diff(f_left, f_right, step_size):
    return (1/(2*(step_size))) * (
        f_left - f_right
    )


def four_prx(f, thetas, j):
    length = thetas.shape[0]

    return - (constant.four_term_psr['d_plus'] * (
        f(thetas + constant.four_term_psr['alpha'] * unit_vector(j, length)) -
        f(thetas - constant.four_term_psr['alpha'] * unit_vector(j, length))
        - constant.four_term_psr['d_minus'] * (
            f(thetas + constant.four_term_psr['beta'] * unit_vector(j, length)) -
            f(thetas - constant.four_term_psr['beta'] * unit_vector(j, length))
        )
    ))


def two_finite_diff(f, thetas, j, step_size):
    length = thetas.shape[0]
    return (1 / (2*step_size))*(
        f(thetas + step_size * unit_vector(j, length)) -
        f(thetas - step_size * unit_vector(j, length)))

def true_grad(thetas):
    """
        df value in fig 2. paper
    """
    derivate_x = -(np.cos(thetas[1]/2)**2)*np.sin(thetas[0])
    derivate_y = (np.sin(thetas[0]/2)**2)*np.sin(thetas[1])
    derivate_z = 0
    return np.asarray([derivate_x, derivate_y, derivate_z])

def f_analytic(thetas):
    """
        f value in fig 2. paper
    """

    return 1/2*(1 + np.cos(thetas[0]) + (-1 + np.cos(thetas[0])*np.cos(thetas[2])))

def pseudo_four_prx(f, thetas, j):
    length = thetas.shape[0]

    return np.real(- 1j/2*(constant.four_term_psr['d_plus'] * (
        f(thetas + constant.four_term_psr['alpha'] * unit_vector(j, length)) -
        f(thetas - constant.four_term_psr['alpha'] * unit_vector(j, length))
        - constant.four_term_psr['d_minus'] * (
            f(thetas + constant.four_term_psr['beta'] * unit_vector(j, length)) -
            f(thetas - constant.four_term_psr['beta'] * unit_vector(j, length))
        )
    )))
