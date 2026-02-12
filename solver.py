import random
from typing import List


def parse_float(token: str):
    token = token.strip().replace(",", ".")
    if not token:
        raise ValueError("Пустое значение числа.")
    return float(token)


def norm_inf(a: List[List[float]]):
    return max(sum(abs(x) for x in row) for row in a)


def is_diag_dominant(a: List[List[float]]):
    n = len(a)
    for i in range(n):
        diag = abs(a[i][i])
        off = sum(abs(a[i][j]) for j in range(n) if j != i)
        if diag < off:
            return False
    return True


def _build_dominance_graph(a: List[List[float]]):
    n = len(a)
    graph: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        row_sum = sum(abs(x) for x in a[i])
        for j in range(n):
            diag = abs(a[i][j])
            if diag >= row_sum - diag:
                graph[i].append(j)
    return graph


def _find_dominance_matching(
    a: List[List[float]],
):
    n = len(a)
    graph = _build_dominance_graph(a)
    match_to_col = [-1] * n

    def dfs(v: int, seen: List[bool]):
        for col in graph[v]:
            if seen[col]:
                continue
            seen[col] = True
            if match_to_col[col] == -1 or dfs(match_to_col[col], seen):
                match_to_col[col] = v
                return True
        return False

    for v in range(n):
        seen = [False] * n
        if not dfs(v, seen):
            return None

    return match_to_col


def _find_row_perm_for_dominance(
    a: List[List[float]],
):
    match_to_col = _find_dominance_matching(a)
    if match_to_col is None:
        return None
    row_perm = match_to_col[:]
    if any(r == -1 for r in row_perm):
        return None
    return row_perm


def _find_col_perm_for_dominance(
    a: List[List[float]],
):
    match_to_col = _find_dominance_matching(a)
    if match_to_col is None:
        return None
    n = len(a)
    perm = [-1] * n
    for col, row in enumerate(match_to_col):
        perm[row] = col
    if any(p == -1 for p in perm):
        return None
    return perm


def make_diag_dominant(
    a: List[List[float]],
    b: List[float],
):
    if is_diag_dominant(a):
        return a, b, None, None

    row_perm = _find_row_perm_for_dominance(a)
    if row_perm is not None:
        a2 = [a[i][:] for i in row_perm]
        b2 = [b[i] for i in row_perm]
        return a2, b2, None, row_perm

    col_perm = _find_col_perm_for_dominance(a)
    if col_perm is None:
        return a, b, None, None
    n = len(a)
    a2 = [[a[i][col_perm[j]] for j in range(n)] for i in range(n)]
    return a2, b[:], col_perm, None


def gauss_seidel(
    a: List[List[float]],
    b: List[float],
    eps: float,
    max_iter: int = 10000,
    ):
    n = len(a)
    x = [0.0] * n
    prev = [0.0] * n
    errors = [0.0] * n

    for k in range(1, max_iter + 1):
        for i in range(n):
            if a[i][i] == 0.0:
                raise ZeroDivisionError(
                    f"Нулевой диагональный элемент в строке {i + 1}."
                )
            s1 = sum(a[i][j] * x[j] for j in range(i))
            s2 = sum(a[i][j] * prev[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / a[i][i]

        errors = [abs(x[i] - prev[i]) for i in range(n)]
        if max(errors) < eps:
            return x, k, errors
        prev = x[:]

    return x, max_iter, errors


def generate_random_system(
    n: int,
    low,
    high):
    
    def rand_val():
        digits = random.randint(0, 10)
        return round(random.uniform(low, high), digits)

    a = []
    
    for i in range(n):
        row = [rand_val() for _ in range(n)]
        row[i] = 0
        
        s = sum(abs(x) for x in row)
        digits = random.randint(0, 10)
        diag = round(random.uniform(s + 1, s + 5), digits)
        row[i] = diag
        a.append([float(x) for x in row])
    b = [float(rand_val()) for _ in range(n)]
    return a, b


def apply_inverse_permutation(x_perm: List[float], perm: List[int]) -> List[float]:
    n = len(x_perm)
    x = [0.0] * n
    for new_pos, old_col in enumerate(perm):
        x[old_col] = x_perm[new_pos]
    return x
