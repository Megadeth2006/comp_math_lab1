import math
import sys
from typing import List

from solver import (
    ensure_nonzero_diagonal,
    gauss_seidel,
    generate_random_system,
    is_diag_dominant,
    make_diag_dominant,
    norm_inf,
    parse_float,
)


def format_power10(value: float, precision: int = 6):
    if value == 0.0:
        return "0"

    exponent = int(math.floor(math.log10(abs(value))))
    if -2 <= exponent <= 2:
        return f"{value:.{precision}f}".rstrip("0").rstrip(".")

    mantissa = value / (10 ** exponent)
    mantissa = round(mantissa, precision)
    if abs(mantissa) >= 10:
        mantissa /= 10
        exponent += 1

    mantissa_str = f"{mantissa:.{precision}f}".rstrip("0").rstrip(".")
    if mantissa_str == "1":
        return f"10^{exponent}"
    if mantissa_str == "-1":
        return f"-10^{exponent}"
    return f"{mantissa_str}*10^{exponent}"


def read_numbers(count: int):
    while True:
        line = input().strip()
        if not line:
            print("ошибка: пустая строка. Повторите ввод.")
            continue
        try:
            tokens = line.split()
            if len(tokens) != count:
                print(f"ошибка: нужно {count} чисел, получено {len(tokens)}.")
                continue
            return [parse_float(t) for t in tokens]
        except ValueError as exc:
            print(f"ошибка ввода числа: {exc}. Повторите ввод.")


def read_matrix_from_keyboard():
    while True:
        try:
            raw = input("Введите n (<=20, одно число): ").strip()
            if len(raw.split()) != 1:
                print("ошибка: введите ровно одно число для n.")
                continue
            n = int(parse_float(raw))
            if n <= 0 or n > 20:
                print("ошибка: n должно быть в диапазоне 1..20.")
                continue
            break
        except ValueError as exc:
            print(f"ошибка: {exc}. Повторите ввод.")

    a: List[List[float]] = []
    for i in range(n):
        print(f"Введите строку {i + 1} из {n} чисел:")
        a.append(read_numbers(n))

    zeros_b = [0.0] * n
    print(f"Введите вектор b из {n} чисел:")
    b = read_numbers(n)
    return a, b


def read_matrix_from_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as exc:
        raise OSError(f"Не удалось открыть файл: {exc}") from exc

    tokens = [t for t in raw.replace(",", ".").split()]
    if not tokens:
        raise ValueError("Файл пустой.")
    
    try:
        n = int(float(tokens[0]))
    except ValueError as exc:
        
        raise ValueError("Первое значение в файле должно быть n.") from exc
    if n <= 0 or n > 20:
        raise ValueError("n должно быть в диапазоне 1..20.")

    expected = 1 + n * n + n + 1
    if len(tokens) != expected:
        
        raise ValueError(
            f"Ожидалось {expected} чисел (n + матрица + b + eps), получено {len(tokens)}."
        )

    nums = [parse_float(t) for t in tokens[1:]]
    a = []
    
    idx = 0
    
    for _ in range(n):
        row = nums[idx : idx + n]
        a.append(row)
        idx += n
    b = nums[idx : idx + n]
    idx += n
    eps = nums[idx]
    if eps <= 0:
        raise ValueError("eps должно быть > 0.")
    return a, b, eps


def select_input():
    print("Выберите способ ввода:\n")
    print("1) с клавиатуры")
    print("2) из файла")
    print("3) случайная матрица")
    choice = input("Ваш выбор: ").strip()

    if choice == "1":
        a, b = read_matrix_from_keyboard()
        return a, b, None
    if choice == "2":
        path = input("Введите путь к файлу: ").strip()
        a, b, eps = read_matrix_from_file(path)
        return a, b, eps
    
    if choice == "3":
        while True:
            try:
                raw = input("Введите n меньше/равно 20: ").strip()
                if len(raw.split()) != 1:
                    print("ошибка: введите ровно одно число для n")
                    continue
                n = int(parse_float(raw))
                if n <= 0 or n > 20:
                    print("ошибка: n должно быть в диапазоне от 1 до 20")
                    continue
                break
            
            except ValueError as exc:
                print(f"ошибка: {exc} Повторите ввод")
        while True:
            try:
                low = parse_float(input("Нижняя граница элементов: ").strip())
                high = parse_float(input("Верхняя граница элементов: ").strip())
                if low > high:
                    print("ошибка: нижняя граница больше верхней")
                    continue
                break
            except ValueError as exc:
                print(f"ошибка: {exc} Повторите ввод")
        a, b = generate_random_system(n, low=low, high=high)
        print("Сгенерированная матрица A:")
        for row in a:
            print(" ".join(format_power10(x) for x in row))
        print("Сгенерированный вектор b:")
        print(" ".join(format_power10(x) for x in b))
        return a, b, None

    print("ошибка: неизвестный вариант")
    return select_input()


def read_eps():
    while True:
        try:
            eps = parse_float(input("Введите точность eps: ").strip())
            if eps <= 0:
                print("ошибка: eps должно быть > 0.")
                continue
            return eps
        except ValueError as exc:
            print(f"ошибка: {exc}. Повторите ввод.")


def main():
    try:
        a, b, eps = select_input()

        a2, b2, col_perm, row_perm = make_diag_dominant(a, b)
        if col_perm is None and row_perm is None and not is_diag_dominant(a2):
            print(
                "Невозможно добиться диагонального преобладания "
                "перестановкой столбцов/строк."
            )
            sys.exit(0)

        if row_perm is not None:
            print("Диагональное преобладание достигнуто перестановкой строк.")
        if col_perm is not None:
            print("Диагональное преобладание достигнуто перестановкой столбцов.")

        nz_result = ensure_nonzero_diagonal(a2, b2)
        if nz_result is None:
            print("Нулевой диагональный элемент, перестановка невозможна.")
            sys.exit(0)
        a3, b3, row_perm_nz, col_perm_nz = nz_result

        if row_perm is None:
            row_perm = row_perm_nz
        elif row_perm_nz is not None:
            row_perm = [row_perm[i] for i in row_perm_nz]

        if col_perm is None:
            col_perm = col_perm_nz
        elif col_perm_nz is not None:
            col_perm = [col_perm[i] for i in col_perm_nz]

        if eps is None:
            eps = read_eps()

        norm = norm_inf(a3)
        print(f"Норма матрицы (inf): {format_power10(norm)}")

        x_perm, iters, errors = gauss_seidel(a3, b3, eps)
        
        x = x_perm

        print("Вектор неизвестных:")
        for i, val in enumerate(x, 1):
            print(f"x{i} = {format_power10(val)}")

        print(f"Количество итераций: {iters}")
        print("Вектор погрешностей:")
        for i, val in enumerate(errors, 1):
            print(f"|x{i}^k - x{i}^(k-1)| = {format_power10(val)}")
    except Exception as exc:
        print(f"ошибка: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()


