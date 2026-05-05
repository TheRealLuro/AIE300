import torch


# =========================================================
# PYTORCH FUNDAMENTALS DEMO
# =========================================================

print("\n===== PYTORCH BASICS =====\n")

# ---------------------------------------------------------
# 1. CREATE TENSORS
# ---------------------------------------------------------

tensor_from_list = torch.tensor([1.0, 2.0, 3.0])

random_tensor = torch.randn(3, 3)

zeros_tensor = torch.zeros(2, 2)

ones_tensor = torch.ones(2, 2)

print("Tensor from list:")
print(tensor_from_list)

print("\nRandom tensor:")
print(random_tensor)

print("\nZeros tensor:")
print(zeros_tensor)

print("\nOnes tensor:")
print(ones_tensor)


# ---------------------------------------------------------
# 2. BASIC OPERATIONS
# ---------------------------------------------------------

a = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])

b = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])

# Addition
addition = a + b

# Matrix multiplication
matrix_mult = torch.matmul(a, b)

print("\nMatrix A:")
print(a)

print("\nMatrix B:")
print(b)

print("\nA + B:")
print(addition)

print("\nA x B:")
print(matrix_mult)


# ---------------------------------------------------------
# 3. AUTOGRAD DEMO
# ---------------------------------------------------------

x = torch.tensor(3.0, requires_grad=True)

# y = x^2 + 2x + 1
y = x ** 2 + 2 * x + 1

# Compute gradient
y.backward()

print("\nAutograd Example:")
print(f"x = {x.item()}")
print(f"y = {y.item()}")
print(f"dy/dx = {x.grad.item()}")  # Expected = 8.0


print("\n===== END OF PYTORCH BASICS =====\n")